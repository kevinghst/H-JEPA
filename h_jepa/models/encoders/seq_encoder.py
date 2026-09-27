import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x: Tensor of shape (B, L, D)
        mask: Bool tensor of shape (B, L), where True means pad / ignore

    Returns:
        Tensor of shape (B, D)
    """
    valid = (~mask).float()
    summed = (x * valid.unsqueeze(-1)).sum(dim=1)
    count = valid.sum(dim=1).clamp_min(1.0).unsqueeze(-1)
    return summed / count


class PerStepMLP(nn.Module):
    """
    Effectively similar to a per-step Conv1d projection used in DINO-WM.
    Maps each token vector independently from input_dim -> d_model.
    """

    def __init__(self, input_dim: int, d_model: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, L, input_dim)

        Returns:
            Tensor of shape (B, L, d_model)
        """
        return self.fc1(x)


class SequenceEncoder(nn.Module):
    """
    Encode a fixed or variable-length sequence chunk into one vector.

    Inputs:
        inputs:       (B, T, L, input_dim)
        padding_mask: (B, T, L), where True = padding (ignore)

    Outputs:
        If stochastic:
            mu, std: each (B, T, output_dim)
        Else:
            z: (B, T, output_dim)
    """

    def __init__(
        self,
        output_dim: int = 4,
        input_dim: int = 10,
        d_model: int = 64,
        step_mlp: bool = True,
        nhead: int = 2,
        num_layers: int = 1,
        ff_mult: int = 2,
        dropout: float = 0.1,
        use_cls: bool = True,
        stochastic: bool = False,
        max_chunk: int = 15,
        norm_first: bool = True,
        uniform_input: bool = False,
    ):
        super().__init__()

        self.stochastic = stochastic
        self.use_cls = use_cls
        self.max_chunk = max_chunk
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.use_step_mlp = step_mlp
        self.uniform_input = uniform_input

        if self.use_step_mlp:
            self.step_mlp = PerStepMLP(input_dim=input_dim, d_model=d_model)
        else:
            if d_model != input_dim:
                raise ValueError(
                    "When step_mlp=False, d_model must equal input_dim."
                )
            self.step_mlp = nn.Identity()

        self.pos = nn.Parameter(torch.zeros(max_chunk + 1, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_mult * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=norm_first,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        out_dim = 2 * output_dim if stochastic else output_dim
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, out_dim),
        )

        self.final_ln = nn.LayerNorm(output_dim, elementwise_affine=True, eps=1e-6)

    def forward(
        self,
        inputs: torch.Tensor,
        padding_mask: torch.Tensor = None,
        *,
        action_masks: torch.Tensor = None,
    ):
        """
        Args:
            inputs: Tensor of shape (B, T, L, input_dim)
            padding_mask: Bool tensor of shape (B, T, L), where True = padding
            action_masks: Deprecated alias for padding_mask.

        Returns:
            If self.stochastic:
                mu, std: each (B, T, output_dim)
            Else:
                z: (B, T, output_dim)
        """
        if action_masks is not None:
            if padding_mask is not None:
                raise ValueError("Pass only one of padding_mask or action_masks.")
            padding_mask = action_masks

        if padding_mask is None and not self.uniform_input:
            raise ValueError(
                "padding_mask is required when uniform_input=False. "
                "Set uniform_input=True to omit masks for all-valid chunks."
            )

        B, T, L, C = inputs.shape
        assert C == self.input_dim, (
            f"Last dim of inputs must be {self.input_dim}, got {C}"
        )

        if padding_mask is None:
            padding_mask = torch.zeros(B, T, L, dtype=torch.bool, device=inputs.device)

        if L > self.max_chunk:
            warnings.warn(
                f"chunk_t={L} exceeds max_chunk={self.max_chunk}; truncating to max_chunk"
            )
            inputs = inputs[:, :, : self.max_chunk]
            padding_mask = padding_mask[:, :, : self.max_chunk]
            L = self.max_chunk

        x = inputs.reshape(B * T, L, C)
        m = padding_mask.reshape(B * T, L)

        x = self.step_mlp(x)

        pos = self.pos[1 : L + 1].unsqueeze(0)
        x = x + pos

        if self.use_cls:
            cls_tok = self.pos[0].unsqueeze(0).unsqueeze(0).expand(B * T, 1, -1)
            x = torch.cat([cls_tok, x], dim=1)

            pad_cls = torch.zeros(B * T, 1, dtype=torch.bool, device=x.device)
            src_key_padding_mask = torch.cat([pad_cls, m], dim=1)
        else:
            src_key_padding_mask = m

        y = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        if self.use_cls:
            h = y[:, 0, :]
        else:
            h = masked_mean(y, src_key_padding_mask)

        out = self.head(h)

        if self.stochastic:
            mu, log_std = torch.chunk(out, 2, dim=-1)
            std = F.softplus(log_std) + 1e-5

            mu = mu.view(B, T, -1)
            std = std.view(B, T, -1)
            return mu, std

        z = self.final_ln(out)
        z = z.view(B, T, -1)
        return z


class FlattenedSequenceEncoder(nn.Module):
    """Adapt SequenceEncoder to JEPA's flattened state-encoder interface."""

    def __init__(self, **kwargs):
        super().__init__()
        if bool(kwargs.get("stochastic", False)):
            raise ValueError("FlattenedSequenceEncoder does not support stochastic=True")
        self.encoder = SequenceEncoder(**kwargs)
        self.output_dim = self.encoder.output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                "FlattenedSequenceEncoder expects shape (B*T, K, D), "
                f"got {tuple(x.shape)}"
            )
        return self.encoder(x[:, None])[:, 0]
