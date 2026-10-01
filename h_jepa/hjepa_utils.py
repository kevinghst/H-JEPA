import torch
from torch.nn import functional as F


def _loss_components(loss_cfg):
    components = {name: loss_cfg[name] for name in ("embed", "pixel", "proprio") if name in loss_cfg}
    return components or {"embed": loss_cfg}


def _loss_term_enabled(component_cfg, name):
    if name == "pred" and name not in component_cfg:
        return True
    if name not in component_cfg:
        return False
    return bool(component_cfg[name].get("enabled", False))


def _loss_term_weight(component_cfg, name):
    if name == "pred" and name not in component_cfg:
        return 1.0
    return float(component_cfg[name].get("weight", 1.0))


def _component_key(component, level):
    if component == "embed":
        return f"embed_{level}"
    return f"{component}_embed_{level}"


def _component_loss_key(loss_name, component, level_suffix):
    if component == "embed":
        return f"{loss_name}_loss{level_suffix}"
    return f"{loss_name}_loss_{component}{level_suffix}"


def _sigreg_module_name(component, level):
    if component == "embed":
        return f"sigreg_level{level}"
    return f"sigreg_{component}_level{level}"


def _pred_component(pred, output, level, component):
    # Fusion-level predictions are [pixel | proprio] along the channel dim (2 for patch
    # tokens, last otherwise).
    if component == "embed":
        return pred
    pixel_embed = output[f"pixel_embed_{level}"]
    dim = 2 if pixel_embed.ndim >= 5 else -1
    pixel_size = pixel_embed.size(dim)
    if component == "pixel":
        return pred.narrow(dim, 0, pixel_size)
    return pred.narrow(dim, pixel_size, pred.size(dim) - pixel_size)


def _pred_mse(pred, target):
    # Computed in the inputs' dtype (bf16 under autocast), as in the paper runs; F.mse_loss
    # would be autocast to fp32 and change the training numerics.
    return (pred - target).square().mean()


def hjepa_forward(self, batch, stage, cfg, *, normalize_batch):
    """Encode HJEPA inputs, predict next states, and compute per-level losses."""
    normalize_batch(batch)

    output = self.model.encode_hierarchical(batch)

    total_loss = None
    for level in range(1, int(cfg.num_levels) + 1):
        level_suffix = f"_level{level}"
        level_cfg = cfg[f"level{level}"]
        history_size = int(level_cfg.wm.history_size)
        rollout_n = int(level_cfg.wm.get("rollout_n", 1))
        rollout_loss_weight = float(level_cfg.wm.get("rollout_loss_weight", 1.0))
        jepa = self.model.get_level(level)
        emb = output[f"embed_{level}"]
        act_emb = output[f"action_{level}"]

        nsteps = level_cfg.wm.get("nsteps", None)
        if nsteps is not None:
            pred_passes = jepa.parallel_unroll(emb, act_emb, int(nsteps))
            pred_emb = pred_passes.mean(dim=0)
            tgt_emb = emb
        elif rollout_n == 1:
            teacher_pred = jepa.predict(emb[:, :history_size], act_emb[:, :history_size])
            pred_emb = teacher_pred
            tgt_emb = emb[:, 1 : history_size + 1]
        else:
            teacher_pred = jepa.predict(emb[:, :history_size], act_emb[:, :history_size])
            rollout = emb[:, :history_size]
            for step in range(rollout_n):
                pred_step = jepa.predict(
                    rollout[:, -history_size:], act_emb[:, step : step + history_size]
                )[:, -1:]
                rollout = torch.cat([rollout, pred_step], dim=1)
            pred_emb = rollout[:, history_size:]
            tgt_emb = emb[:, history_size : history_size + rollout_n]

        with torch.no_grad():
            output[f"mse_loss{level_suffix}"] = F.mse_loss(pred_emb, tgt_emb)
            output[f"l1_loss{level_suffix}"] = F.smooth_l1_loss(pred_emb, tgt_emb)
            output[f"dim_mean_loss{level_suffix}"] = tgt_emb.mean()
            output[f"dim_std_loss{level_suffix}"] = tgt_emb.std()
            output[f"dim_min_loss{level_suffix}"] = tgt_emb.min()
            output[f"dim_max_loss{level_suffix}"] = tgt_emb.max()

        level_loss = None
        for component, component_cfg in _loss_components(level_cfg.loss).items():
            component_emb = output[_component_key(component, level)]
            component_loss = None

            if _loss_term_enabled(component_cfg, "pred") and nsteps is not None:
                pred_component = _pred_component(pred_passes, output, level, component)
                tgt_component = (
                    component_emb.detach() if level_cfg.wm.get("detach_pred_target") else component_emb
                )
                pred_loss = (
                    (pred_component - tgt_component.unsqueeze(0)).square().flatten(1).mean(dim=1).mean()
                )
                output[_component_loss_key("pred", component, level_suffix)] = pred_loss
                component_loss = _loss_term_weight(component_cfg, "pred") * pred_loss
            elif _loss_term_enabled(component_cfg, "pred"):
                teacher_forcing_loss = _pred_mse(
                    _pred_component(teacher_pred, output, level, component),
                    component_emb[:, 1 : history_size + 1],
                )
                output[_component_loss_key("teacher_forcing", component, level_suffix)] = (
                    teacher_forcing_loss
                )
                if rollout_n == 1:
                    pred_loss = teacher_forcing_loss
                else:
                    rollout_loss = _pred_mse(
                        _pred_component(pred_emb, output, level, component),
                        component_emb[:, history_size : history_size + rollout_n],
                    )
                    output[_component_loss_key("rollout", component, level_suffix)] = rollout_loss
                    if history_size > 1:
                        pred_loss = teacher_forcing_loss + rollout_loss_weight * rollout_loss
                    else:
                        pred_loss = rollout_loss_weight * rollout_loss
                output[_component_loss_key("pred", component, level_suffix)] = pred_loss
                component_loss = _loss_term_weight(component_cfg, "pred") * pred_loss

            if _loss_term_enabled(component_cfg, "sigreg"):
                sigreg = getattr(self, _sigreg_module_name(component, level))
                sigreg_loss = sigreg(component_emb.flatten(2).transpose(0, 1))
                output[_component_loss_key("sigreg", component, level_suffix)] = sigreg_loss
                weighted_sigreg_loss = _loss_term_weight(component_cfg, "sigreg") * sigreg_loss
                component_loss = (
                    weighted_sigreg_loss
                    if component_loss is None
                    else component_loss + weighted_sigreg_loss
                )

            if component_loss is None:
                continue
            output[f"{component}_loss{level_suffix}"] = component_loss
            level_loss = component_loss if level_loss is None else level_loss + component_loss

        action_sigreg_coeff = float(level_cfg.get("action_sigreg_coeff", 0.0))
        if action_sigreg_coeff > 0:
            action_sigreg_loss = getattr(self, _sigreg_module_name("action", level))(
                act_emb[:, : emb.size(1) - 1].transpose(0, 1)
            )
            output[_component_loss_key("sigreg", "action", level_suffix)] = action_sigreg_loss
            level_loss = level_loss + action_sigreg_coeff * action_sigreg_loss

        idm_coeff = float(level_cfg.get("idm_coeff", 0.0))
        if idm_coeff > 0:
            idm_loss = jepa.idm_loss_fn(emb, act_emb)
            output[f"idm_loss{level_suffix}"] = idm_loss
            level_loss = level_loss + idm_coeff * idm_loss

        if level_loss is None:
            continue
        output[f"loss{level_suffix}"] = level_loss
        if level_loss.isnan():
            raise ValueError(f"NaN loss encountered at level {level}!")
        total_loss = level_loss if total_loss is None else total_loss + level_loss

    output["loss"] = total_loss

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)

    return output


def create_world_model(cfg):
    from loss import InverseDynamicsLoss, InverseDynamicsModel, SIGReg
    from models.encoders.build_encoder import build_encoder
    from models.encoders.seq_encoder import SequenceEncoder, SequenceMLPEncoder
    from models.hjepa import HJEPA
    from models.jepa import FusionEncoder, JEPA, ProjectedEncoder, ProjectedPredictor
    from models.module import Embedder, build_projector
    from models.predictors.predictors import ARPredictor, CausalTransformerPredictor

    jepas = []
    embed_dims = {}
    pixel_embed_dims = {}
    proprio_embed_dims = {}

    for level in range(1, int(cfg.num_levels) + 1):
        level_cfg = cfg[f"level{level}"]
        target_length = int(level_cfg.wm.history_size) + int(level_cfg.wm.get("rollout_n", 1))
        encoder_cfg = dict(level_cfg.encoder)
        encoder_projector_cfg = encoder_cfg.pop("projector")

        if level_cfg.wm.get("use_proprio", False):
            pixel_spec = encoder_cfg.pop("pixel_encoder")
            proprio_spec = encoder_cfg.pop("proprio_encoder")

            pixel_base_encoder, hidden_dim = build_encoder(
                dict(pixel_spec["encoder"]),
                default_patch_size=cfg.patch_size,
                default_image_size=cfg.img_size,
            )
            pixel_output_dim = int(pixel_spec["projector"].get("output_dim", hidden_dim))
            pixel_encoder = ProjectedEncoder(
                pixel_base_encoder,
                build_projector(
                    pixel_spec["projector"],
                    input_dim=hidden_dim,
                    output_dim=pixel_output_dim,
                ),
            )

            proprio_encoder_cfg = dict(proprio_spec["encoder"])
            proprio_encoder_cfg.setdefault("input_dim", int(level_cfg.wm.proprio_dim))
            proprio_encoder_cfg.setdefault("output_dim", int(level_cfg.wm.proprio_emb_dim))
            proprio_base_encoder, proprio_hidden_dim = build_encoder(
                proprio_encoder_cfg,
                default_patch_size=cfg.patch_size,
                default_image_size=cfg.img_size,
            )
            proprio_output_dim = int(
                proprio_spec["projector"].get("output_dim", proprio_encoder_cfg["output_dim"])
            )
            proprio_encoder = ProjectedEncoder(
                proprio_base_encoder,
                build_projector(
                    proprio_spec["projector"],
                    input_dim=proprio_hidden_dim,
                    output_dim=proprio_output_dim,
                ),
            )

            hidden_dim = int(hidden_dim)
            embed_dim = int(level_cfg.wm.get("embed_dim", hidden_dim))
            pixel_embed_dims[level] = pixel_output_dim
            proprio_embed_dims[level] = proprio_output_dim
            encoder = FusionEncoder(
                pixel_encoder=pixel_encoder,
                proprio_encoder=proprio_encoder,
                projector=build_projector(
                    encoder_projector_cfg,
                    input_dim=pixel_output_dim + proprio_output_dim,
                    output_dim=embed_dim,
                ),
                proprio_key="proprio" if level == 1 else "proprio_input",
            )
        else:
            base_encoder, hidden_dim = build_encoder(
                encoder_cfg,
                default_patch_size=cfg.patch_size,
                default_image_size=cfg.img_size,
            )
            embed_dim = int(level_cfg.wm.get("embed_dim", hidden_dim))
            encoder = ProjectedEncoder(
                base_encoder,
                build_projector(
                    encoder_projector_cfg,
                    input_dim=hidden_dim,
                    output_dim=embed_dim,
                ),
            )
        embed_dims[level] = embed_dim

        predictor_cfg = dict(level_cfg.predictor)
        if predictor_cfg.pop("type", None) == "causal_transformer":
            predictor = CausalTransformerPredictor(input_dim=embed_dim, **predictor_cfg)
        else:
            predictor_projector_cfg = predictor_cfg.pop("projector")
            predictor = ProjectedPredictor(
                ARPredictor(
                    num_frames=level_cfg.wm.history_size,
                    input_dim=embed_dim,
                    output_dim=hidden_dim,
                    **predictor_cfg,
                ),
                build_projector(
                    predictor_projector_cfg,
                    input_dim=hidden_dim,
                    output_dim=embed_dim,
                ),
            )

        action_encoder = None
        queue_size = 0
        if "action_encoder" in level_cfg:
            action_encoder_kwargs = dict(level_cfg.action_encoder)
            queue_size = int(action_encoder_kwargs.pop("queue_size", 0))
            if action_encoder_kwargs.pop("type", "sequence") == "mlp":
                action_encoder = SequenceMLPEncoder(
                    temporal_stride=int(level_cfg.get("stride", 1)), **action_encoder_kwargs
                )
            else:
                action_encoder = SequenceEncoder(**action_encoder_kwargs)

        action_embed_kwargs = dict(level_cfg.get("action_embed", {}))
        if action_embed_kwargs.pop("type", None) == "identity":
            action_embed = torch.nn.Identity()
        else:
            if level == 1:
                action_embed_kwargs["input_dim"] = (
                    cfg.data.dataset.level1.frameskip * cfg.level1.wm.action_dim
                )
            action_embed_kwargs["emb_dim"] = embed_dim
            action_embed = Embedder(**action_embed_kwargs)

        jepa = JEPA(
            encoder=encoder,
            predictor=predictor,
            action_embed=action_embed,
            action_encoder=action_encoder,
            level=level,
            action_queue_size=queue_size,
            temporal_stride=int(level_cfg.get("stride", 1)),
            temporal_window_size=int(level_cfg.get("window_size", 1)),
            target_length=target_length,
        )

        if float(level_cfg.get("idm_coeff", 0.0)) > 0:
            jepa.idm_loss_fn = InverseDynamicsLoss(
                InverseDynamicsModel(
                    state_dim=embed_dim,
                    hidden_dim=int(level_cfg.idm.hidden_dim),
                    action_dim=int(level_cfg.predictor.action_dim),
                )
            )

        jepas.append(jepa)

    world_model = HJEPA(jepas=jepas)
    world_model.embed_dims = embed_dims
    world_model.pixel_embed_dims = pixel_embed_dims
    world_model.proprio_embed_dims = proprio_embed_dims

    losses = {}
    for level in range(1, int(cfg.num_levels) + 1):
        level_cfg = cfg[f"level{level}"]
        for component, component_cfg in _loss_components(level_cfg.loss).items():
            if not _loss_term_enabled(component_cfg, "sigreg"):
                continue
            sigreg_cfg = component_cfg.sigreg
            sigreg_kwargs = {
                k: v for k, v in sigreg_cfg.items() if k not in {"enabled", "weight"}
            }
            losses[_sigreg_module_name(component, level)] = SIGReg(**sigreg_kwargs)
        if float(level_cfg.get("action_sigreg_coeff", 0.0)) > 0:
            losses[_sigreg_module_name("action", level)] = SIGReg()

    return world_model, losses
