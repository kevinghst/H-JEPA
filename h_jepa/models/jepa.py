"""JEPA Model Implementation"""

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

def detach_clone(v):
    return v.detach().clone() if torch.is_tensor(v) else v


def _run_encoder(encoder, x):
    if x.ndim >= 4:
        return encoder(x, interpolate_pos_encoding=True)
    return encoder(x)


def _split_encoder_output(
    encoder_output,
    allow_cls_token: bool = True,
) -> torch.Tensor:
    hidden = (
        encoder_output.last_hidden_state
        if hasattr(encoder_output, "last_hidden_state")
        else encoder_output
    )

    if hidden.ndim == 2:
        return hidden
    if hidden.ndim == 3 and allow_cls_token:
        return hidden[:, 0]

    if hidden.ndim == 3:
        raise ValueError(
            "Chunked temporal encoder inputs must be collapsed by the encoder "
            "to rank-2 shape (B*T, D); got rank-3 output "
            f"with shape {tuple(hidden.shape)}"
        )

    raise ValueError(
        "Encoder output must be rank-2 or rank-3, "
        f"got shape {tuple(hidden.shape)}"
    )


class ProjectedEncoder(nn.Module):
    """Encoder followed by its projection head."""

    def __init__(self, encoder, projector=None):
        super().__init__()
        self.encoder = encoder
        self.projector = projector or nn.Identity()

    def forward(self, x, *, allow_cls_token: bool = True):
        output = _run_encoder(self.encoder, x)
        hidden = _split_encoder_output(output, allow_cls_token=allow_cls_token)
        return self.projector(hidden)

    def encode_info(self, info, *, key: str, allow_cls_token: bool = True):
        x = rearrange(info[key].float(), "b t ... -> (b t) ...")
        return {"embed": self(x, allow_cls_token=allow_cls_token)}


class FusionEncoder(nn.Module):
    """Encode pixel/proprio streams and fuse them into the model embedding."""

    def __init__(
        self,
        pixel_encoder,
        proprio_encoder,
        projector=None,
        proprio_key: str = "proprio",
    ):
        super().__init__()
        self.pixel_encoder = pixel_encoder
        self.proprio_encoder = proprio_encoder
        self.projector = projector or nn.Identity()
        self.proprio_key = proprio_key

    def encode_info(self, info, *, key: str, allow_cls_token: bool = True):
        if self.proprio_key not in info:
            raise KeyError(
                "FusionEncoder requires a proprio stream, but "
                f"'{self.proprio_key}' is missing from input"
            )

        pixel_embed = self.pixel_encoder.encode_info(
            info,
            key=key,
            allow_cls_token=allow_cls_token,
        )["embed"]
        proprio_embed = self.proprio_encoder.encode_info(
            info,
            key=self.proprio_key,
            allow_cls_token=allow_cls_token,
        )["embed"]
        embed = self.projector(torch.cat([pixel_embed, proprio_embed], dim=-1))
        return {
            "embed": embed,
            "pixel_embed": pixel_embed,
            "proprio_embed": proprio_embed,
        }


class ProjectedPredictor(nn.Module):
    """Predictor followed by its projection head."""

    def __init__(self, predictor, projector=None):
        super().__init__()
        self.predictor = predictor
        self.projector = projector or nn.Identity()

    def forward(self, emb, act_emb):
        preds = self.predictor(emb, act_emb)
        preds = self.projector(rearrange(preds, "b t d -> (b t) d"))
        return rearrange(preds, "(b t) ... -> b t ...", b=emb.size(0))


class JEPA(nn.Module):
    def __init__(
        self,
        encoder,
        predictor,
        action_embed,
        action_encoder=None,
        level:int=1,
        action_queue_size: int = 0,
        temporal_stride: int = 1,
        temporal_window_size: int = 1,
        target_length: int | None = None,
    ):
        super().__init__()

        self.encoder = encoder
        self.predictor = predictor
        self.action_embed = action_embed
        self.action_encoder = action_encoder or nn.Identity()
        self.level = level
        self.temporal_stride = int(temporal_stride)
        self.temporal_window_size = int(temporal_window_size)
        self.target_length = None if target_length is None else int(target_length)
        self.configure_action_queue(action_queue_size)

    def _state_windows(
        self,
        x: torch.Tensor,
        starts: torch.Tensor,
        window_size: int,
    ) -> torch.Tensor:
        """Subsample or chunk a time sequence using full state windows only."""
        if window_size == 1:
            return x.index_select(1, starts)

        offsets = torch.arange(window_size, device=x.device)
        indices = starts[:, None] + offsets[None, :]
        flat = indices.reshape(-1)
        chunks = x.index_select(1, flat)
        return chunks.reshape(x.size(0), starts.numel(), window_size, *x.shape[2:])

    def _action_windows(
        self,
        x: torch.Tensor,
        starts: torch.Tensor,
        state_window_size: int,
        stride: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Chunk actions with zero padding.

        The first action in chunk T aligns with the last state in chunk T.
        The returned mask is True for padded positions.
        """
        seq_len = x.size(1)
        action_starts = starts + state_window_size - 1
        offsets = torch.arange(stride, device=x.device)
        indices = action_starts[:, None] + offsets[None, :]
        valid = indices < seq_len
        padding_mask = ~valid
        indices = indices.clamp(max=seq_len - 1)

        flat = indices.reshape(-1)
        chunks = x.index_select(1, flat)
        chunks = chunks.reshape(x.size(0), starts.numel(), stride, *x.shape[2:])

        valid = valid.unsqueeze(0)
        while valid.ndim < chunks.ndim:
            valid = valid.unsqueeze(-1)
        mask = padding_mask.unsqueeze(0).expand(x.size(0), -1, -1)
        return chunks * valid.to(dtype=chunks.dtype), mask

    def _chunk_temporal_info(
        self,
        info: dict,
        key: str,
        stride: int,
        window_size: int,
    ) -> dict:
        stride = int(stride)
        window_size = int(window_size)
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")

        if key not in info:
            raise KeyError(f"Input key '{key}' not found in info")

        ref = info[key]
        if not torch.is_tensor(ref) or ref.ndim < 3:
            raise ValueError(
                f"Expected '{key}' to be a tensor with shape (B, T, ...), "
                f"got {type(ref)} with shape {getattr(ref, 'shape', None)}"
            )

        seq_len = ref.size(1)
        if seq_len < window_size:
            raise ValueError(
                f"Sequence length {seq_len} is shorter than window_size={window_size}"
            )
        starts = torch.arange(0, seq_len - window_size + 1, stride, device=ref.device)
        chunked = dict(info)

        for info_key, value in info.items():
            if not torch.is_tensor(value) or value.ndim < 3:
                continue
            if value.size(0) != ref.size(0) or value.size(1) != seq_len:
                continue

            if info_key == "action":
                action_chunks, action_mask = self._action_windows(
                    value,
                    starts=starts,
                    state_window_size=window_size,
                    stride=stride,
                )
                chunked[info_key] = action_chunks
                chunked["action_mask"] = action_mask
            elif not info_key.startswith("action"):
                chunked[info_key] = self._state_windows(
                    value,
                    starts=starts,
                    window_size=window_size,
                )

        return chunked

    def configure_action_queue(self, queue_size: int = 0) -> None:
        """Configure latent action queue used to initialize planner priors."""
        queue_size = int(queue_size)

        self.action_queue_size = queue_size
        if queue_size == 0:
            self.latent_action_queue = None
            return

        if "latent_action_queue" not in self._buffers:
            self.register_buffer(
                "latent_action_queue", torch.empty(0, 0), persistent=True
            )
        elif self.latent_action_queue is None:
            self._buffers["latent_action_queue"] = torch.empty(0, 0)

        queue = self.latent_action_queue
        if torch.is_tensor(queue) and queue.ndim == 2 and queue.numel() > 0:
            self._buffers["latent_action_queue"] = queue[-queue_size:]

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # latent_action_queue grows during training, so a fresh model's empty (0,0)
        # buffer can't receive a checkpoint's (queue_size, A_emb) tensor under strict
        # load. Resize it to the saved shape before the parent copies into it.
        key = prefix + "latent_action_queue"
        if key in state_dict and "latent_action_queue" in self._buffers:
            saved = state_dict[key]
            self._buffers["latent_action_queue"] = saved.new_empty(saved.shape)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def _update_latent_action_queue(self, latent_actions: torch.Tensor) -> None:
        """latent_actions: (B, T, A_emb)"""

        if not self.training or self.action_queue_size <= 0:
            return

        assert latent_actions.ndim == 3, "latent_actions must be (B, T, A_emb)"

        samples = latent_actions[:, 0].detach()

        queue = self.latent_action_queue
        queue_size = int(self.action_queue_size)

        if queue.numel() == 0:
            updated = samples
        else:
            updated = torch.cat(
                [queue.to(device=samples.device, dtype=samples.dtype), samples],
                dim=0,
            )

        self._buffers["latent_action_queue"] = updated[-queue_size:]

    def get_latent_action_queue(self) -> torch.Tensor | None:
        """Return latent action queue, or None if disabled/empty."""
        queue = getattr(self, "latent_action_queue", None)
        if not torch.is_tensor(queue) or queue.numel() == 0:
            return None

        return queue

    def encode(
        self,
        info,
        key="pixels",
        temporal_stride: int | None = None,
        chunk_temporal_inputs: bool = False,
    ):
        """Encode observations and actions into embeddings.
        info: dict with pixels and action keys
        """
        chunked_state_input = False
        if chunk_temporal_inputs:
            if temporal_stride is None:
                temporal_stride = getattr(self, "temporal_stride", 1)
            temporal_window_size = getattr(self, "temporal_window_size", 1)
            chunked_state_input = int(temporal_window_size) > 1

            info = self._chunk_temporal_info(
                info,
                key=key,
                stride=temporal_stride,
                window_size=temporal_window_size,
            )

        B = info[key].size(0)
        encoder_outputs = self.encoder.encode_info(
            info,
            key=key,
            allow_cls_token=not chunked_state_input,
        )
        for output_key, value in encoder_outputs.items():
            info[f"{output_key}_0"] = rearrange(
                value,
                "(b t) ... -> b t ...",
                b=B,
            )

        if "action" in info:
            action_mask = info.get("action_mask")
            if action_mask is None:
                latent_actions = self.action_encoder(info["action"])
            else:
                latent_actions = self.action_encoder(info["action"], action_mask)
            self._update_latent_action_queue(latent_actions)
            info["action_0"] = self.action_embed(latent_actions)

        return info

    def predict(self, emb, act_emb):
        """Predict next state embedding
        emb: (B, T, D)
        act_emb: (B, T, A_emb)
        """
        return self.predictor(emb, act_emb)

    def parallel_unroll(self, emb, act_emb, nsteps):
        """nsteps passes over the band: each pass predicts frames 1..T-1 from the previous pass (the
        encoder band on pass 0, detached after), with ground-truth frame 0 re-injected on the left.
        Returns every pass stacked: (nsteps, B, T, D), aligned with emb.
        """
        T = emb.size(1) - 1
        pred_input = emb[:, :T]
        passes = []
        for _ in range(nsteps):
            predicted = torch.cat([emb[:, :1], self.predict(pred_input, act_emb[:, :T])], dim=1)
            passes.append(predicted)
            pred_input = torch.cat([emb[:, :1], predicted[:, 1:T].detach()], dim=1)
        return torch.stack(passes)

    def rollout(self, info, action_sequence, history_size: int = 1):
        """Rollout the model given an initial info dict and action sequence."""

        assert "pixels" in info, "pixels not in info_dict"
        H = info["pixels"].size(2)
        B, S, T = action_sequence.shape[:3]
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        info["action"] = act_0
        n_steps = T - H

        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        if "embed_0" not in _init:
            with torch.no_grad():
                _init = self.encode(_init)
        emb = info["embed_0"] = _init["embed_0"].unsqueeze(1).expand(B, S, -1, -1)
        _init = {k: detach_clone(v) for k, v in _init.items()}

        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        act = rearrange(act_0, "b s ... -> (b s) ...")
        act_future = rearrange(act_future, "b s ... -> (b s) ...")

        HS = history_size
        for t in range(n_steps):
            act_emb = self.action_embed(act)
            emb_trunc = emb[:, -HS:]  # (BS, HS, D)
            act_trunc = act_emb[:, -HS:]  # (BS, HS, A_emb)
            pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)
            emb = torch.cat([emb, pred_emb], dim=1)  # (BS, T+1, D)

            next_act = act_future[:, t : t + 1, :]  # (BS, 1, action_dim)
            act = torch.cat([act, next_act], dim=1)  # (BS, T+1, action_dim)

        # predict the last state
        act_emb = self.action_embed(act)  # (BS, T, A_emb)
        emb_trunc = emb[:, -HS:]  # (BS, HS, D)
        act_trunc = act_emb[:, -HS:]  # (BS, HS, A_emb)
        pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)
        emb = torch.cat([emb, pred_emb], dim=1)

        pred_rollout = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        info["predicted_embed_0"] = pred_rollout

        return info

    def criterion(self, info_dict: dict):
        """Planning cost: MSE between the last cost_last_n predicted embeddings and the goal."""
        pred_emb = info_dict["predicted_embed_0"][:, :, 1:, :]
        cost_last_n = max(1, int(info_dict.get("cost_last_n", 1)))
        cost_window = min(cost_last_n, pred_emb.shape[-2])

        pred_emb = pred_emb[:, :, -cost_window:, :]
        goal_emb = info_dict["goal_embed_0"].expand_as(pred_emb).detach()
        cost = F.mse_loss(pred_emb, goal_emb, reduction="none")

        intermediate_weight = float(info_dict.get("intermediate_cost_weight", 1.0))
        if intermediate_weight != 1.0 and cost.shape[2] > 1:
            weights = cost.new_ones(cost.shape[2])
            weights[:-1] = intermediate_weight
            cost = cost * weights.view(1, 1, -1, 1)

        return cost.sum(dim=tuple(range(2, cost.ndim)))

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        if "goal_embed_0" not in info_dict:
            assert "goal" in info_dict, "goal not in info_dict"

        device = next(self.parameters()).device
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)

        if "goal_embed_0" not in info_dict:
            _goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
            _goal["pixels"] = _goal["goal"]

            for k in info_dict:
                if k.startswith("goal_"):
                    _goal[k[len("goal_") :]] = _goal.pop(k)

            _goal.pop("action")
            with torch.no_grad():
                _goal = self.encode(_goal)

            # keep the sample axis, (B, 1, T, D): expand_as against (B, S, W, D) is wrong for B > 1
            info_dict["goal_embed_0"] = _goal["embed_0"].unsqueeze(1)

        info_dict = self.rollout(info_dict, action_candidates)

        return self.criterion(info_dict)
