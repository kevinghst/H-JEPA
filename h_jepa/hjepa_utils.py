import torch
from loguru import logger as logging
from torch.nn import functional as F


def probe_target_key(col: str, level: int) -> str:
    return f"{col}_level{level}_probe_target"


def probe_targets(target):
    if target.ndim >= 4:
        return target[:, :, -1].contiguous()
    return target.contiguous()


def add_probe_targets(output, cfg) -> None:
    for level in range(1, int(cfg.num_levels) + 1):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels") or col == "action":
                continue
            target_key = f"{col}_level{level}"
            if target_key in output:
                output[probe_target_key(col, level)] = probe_targets(output[target_key])


def _loss_components(loss_cfg):
    component_names = ("embed", "pixel", "proprio")
    configured_components = {
        name: loss_cfg[name] for name in component_names if name in loss_cfg
    }
    legacy_losses = {
        name: cfg for name, cfg in loss_cfg.items() if name not in component_names
    }

    if configured_components and legacy_losses:
        raise ValueError(
            "Loss config cannot mix component sections "
            f"{sorted(configured_components)} with legacy flat losses "
            f"{sorted(legacy_losses)}. Put flat losses under loss.embed."
        )

    if configured_components:
        return configured_components

    return {"embed": loss_cfg}


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
    if component in {"pixel", "proprio"}:
        return f"{component}_embed_{level}"
    raise ValueError(f"Unsupported loss component '{component}'.")


def _component_loss_key(loss_name, component, level_suffix):
    if component == "embed":
        return f"{loss_name}_loss{level_suffix}"
    return f"{loss_name}_loss_{component}{level_suffix}"


def _sigreg_module_name(component, level):
    if component == "embed":
        return f"sigreg_level{level}"
    return f"sigreg_{component}_level{level}"


def _split_pred_component(pred, output, level, component):
    if component == "embed":
        return pred
    if component not in {"pixel", "proprio"}:
        raise ValueError(f"Unsupported loss component '{component}'.")

    pixel_key = f"pixel_embed_{level}"
    proprio_key = f"proprio_embed_{level}"
    if pixel_key not in output or proprio_key not in output:
        raise KeyError(
            f"Cannot compute {component} prediction loss for level {level}: "
            f"missing '{pixel_key}' or '{proprio_key}'."
        )

    pixel_embed = output[pixel_key]
    proprio_embed = output[proprio_key]
    if pixel_embed.ndim >= 5:
        pixel_channels = pixel_embed.size(2)
        proprio_channels = proprio_embed.size(2)
        expected_channels = pixel_channels + proprio_channels
        channel_dim = 2
        if pred.size(channel_dim) != expected_channels:
            raise ValueError(
                f"Cannot split level {level} prediction into pixel/proprio "
                f"components: expected channel dim {expected_channels}, "
                f"got {pred.size(channel_dim)}."
            )
        if component == "pixel":
            return pred.narrow(channel_dim, 0, pixel_channels)
        return pred.narrow(channel_dim, pixel_channels, proprio_channels)

    pixel_dim = pixel_embed.size(-1)
    proprio_dim = proprio_embed.size(-1)
    expected_dim = pixel_dim + proprio_dim
    if pred.size(-1) != expected_dim:
        raise ValueError(
            f"Cannot split level {level} prediction into pixel/proprio "
            f"components: expected last dim {expected_dim}, got {pred.size(-1)}."
        )
    if component == "pixel":
        return pred[..., :pixel_dim]
    return pred[..., pixel_dim:]


def hjepa_forward(
    self,
    batch,
    stage,
    cfg,
    *,
    normalize_batch=None,
):
    """Encode HJEPA inputs, predict next states, and compute per-level losses."""
    if normalize_batch is not None:
        normalize_batch(batch)

    output = self.model.encode_hierarchical_per_level_inputs(
        batch,
        key="pixels",
        levels_to_encode=int(cfg.num_levels),
    )
    add_probe_targets(output, cfg)

    total_loss = None
    for level in range(1, int(cfg.num_levels) + 1):
        level_suffix = f"_level{level}"
        level_name = f"level{level}"
        if level_name not in cfg:
            raise KeyError(f"Level config '{level_name}' not found in cfg")

        level_cfg = cfg[level_name]
        wm_cfg = level_cfg.wm
        loss_cfg = level_cfg.loss
        emb_key = f"embed_{level}"
        if emb_key not in output:
            raise KeyError(f"Missing '{emb_key}' in hierarchical output")

        emb = output[emb_key]
        level_loss = None

        act_key = f"action_{level}"
        if act_key not in output:
            raise KeyError(f"Missing '{act_key}' in hierarchical output")

        act_emb = output[act_key]

        history_size = int(wm_cfg.history_size)
        rollout_n = int(wm_cfg.get("rollout_n", 1))
        if rollout_n <= 0:
            raise ValueError(f"level{level}.wm.rollout_n must be positive.")
        rollout_loss_weight = float(wm_cfg.get("rollout_loss_weight", 1.0))
        if rollout_loss_weight < 0:
            raise ValueError(
                f"level{level}.wm.rollout_loss_weight must be non-negative."
            )

        jepa = self.model.get_level(level)

        def _teacher_forcing_prediction():
            pred = jepa.predict(emb[:, :history_size], act_emb[:, :history_size])
            target = emb[:, 1 : history_size + 1]
            return pred, target

        def _full_history_rollout():
            rollout = emb[:, :history_size]
            action_history = act_emb[:, :history_size]
            preds = []

            for step in range(rollout_n):
                emb_trunc = rollout[:, -history_size:]
                act_trunc = action_history[:, -emb_trunc.size(1) :]
                pred_step = jepa.predict(emb_trunc, act_trunc)[:, -1:]
                preds.append(pred_step)
                rollout = torch.cat([rollout, pred_step], dim=1)

                next_action_idx = history_size + step
                if step + 1 < rollout_n:
                    action_history = torch.cat(
                        [
                            action_history,
                            act_emb[:, next_action_idx : next_action_idx + 1],
                        ],
                        dim=1,
                    )

            pred = torch.cat(preds, dim=1)
            target = emb[:, history_size : history_size + rollout_n]
            pred_timeline = torch.cat([emb[:, :history_size], pred], dim=1)
            return pred, target, pred_timeline

        def _predict_rollouts():
            teacher_pred, teacher_target = _teacher_forcing_prediction()
            if rollout_n == 1:
                pred_timeline = torch.cat([emb[:, :1], teacher_pred], dim=1)
                return (
                    teacher_pred,
                    teacher_target,
                    pred_timeline,
                    teacher_pred,
                    teacher_target,
                    None,
                    None,
                )

            rollout_pred, rollout_target, pred_timeline = _full_history_rollout()
            return (
                rollout_pred,
                rollout_target,
                pred_timeline,
                teacher_pred,
                teacher_target,
                rollout_pred,
                rollout_target,
            )

        (
            pred_emb,
            tgt_emb,
            pred_timeline,
            teacher_pred_emb,
            teacher_tgt_emb,
            rollout_pred_emb,
            rollout_tgt_emb,
        ) = _predict_rollouts()

        # Keep prediction probes aligned with the original timeline.
        output[f"pred_embed_{level}"] = pred_timeline

        with torch.no_grad():
            output[f"mse_loss{level_suffix}"] = F.mse_loss(pred_emb, tgt_emb)
            output[f"l1_loss{level_suffix}"] = F.smooth_l1_loss(pred_emb, tgt_emb)
            output[f"dim_mean_loss{level_suffix}"] = tgt_emb.mean()
            output[f"dim_std_loss{level_suffix}"] = tgt_emb.std()
            output[f"dim_min_loss{level_suffix}"] = tgt_emb.min()
            output[f"dim_max_loss{level_suffix}"] = tgt_emb.max()

        for component, component_cfg in _loss_components(loss_cfg).items():
            component_key = _component_key(component, level)
            if component_key not in output:
                raise KeyError(
                    f"Cannot compute {component} losses for level {level}: "
                    f"missing '{component_key}'."
                )

            component_emb = output[component_key]
            component_loss = None

            if _loss_term_enabled(component_cfg, "pred"):
                teacher_pred_component = _split_pred_component(
                    teacher_pred_emb, output, level, component
                )
                if component == "embed":
                    teacher_tgt_component = teacher_tgt_emb
                else:
                    teacher_tgt_component = component_emb[:, 1 : history_size + 1]

                teacher_forcing_loss = F.mse_loss(
                    teacher_pred_component,
                    teacher_tgt_component,
                )
                output[
                    _component_loss_key("teacher_forcing", component, level_suffix)
                ] = teacher_forcing_loss

                if rollout_n == 1:
                    pred_loss = teacher_forcing_loss
                else:
                    rollout_pred_component = _split_pred_component(
                        rollout_pred_emb, output, level, component
                    )
                    if component == "embed":
                        rollout_tgt_component = rollout_tgt_emb
                    else:
                        rollout_tgt_component = component_emb[
                            :, history_size : history_size + rollout_n
                        ]
                    rollout_loss = F.mse_loss(
                        rollout_pred_component,
                        rollout_tgt_component,
                    )
                    output[_component_loss_key("rollout", component, level_suffix)] = (
                        rollout_loss
                    )
                    if history_size > 1:
                        pred_loss = (
                            teacher_forcing_loss
                            + rollout_loss_weight * rollout_loss
                        )
                    else:
                        pred_loss = rollout_loss_weight * rollout_loss

                output[_component_loss_key("pred", component, level_suffix)] = (
                    pred_loss
                )
                component_loss = (
                    _loss_term_weight(component_cfg, "pred") * pred_loss
                )

            if _loss_term_enabled(component_cfg, "sigreg"):
                sigreg_emb = (
                    component_emb.flatten(2)
                    if component_emb.ndim > 3
                    else component_emb
                )
                sigreg_loss = getattr(
                    self, _sigreg_module_name(component, level)
                )(sigreg_emb.transpose(0, 1))
                output[_component_loss_key("sigreg", component, level_suffix)] = (
                    sigreg_loss
                )
                weighted_sigreg_loss = (
                    _loss_term_weight(component_cfg, "sigreg") * sigreg_loss
                )
                component_loss = (
                    weighted_sigreg_loss
                    if component_loss is None
                    else component_loss + weighted_sigreg_loss
                )

            if component_loss is None:
                continue

            output[f"{component}_loss{level_suffix}"] = component_loss
            level_loss = (
                component_loss
                if level_loss is None
                else level_loss + component_loss
            )

        if level_loss is None:
            continue

        output[f"loss{level_suffix}"] = level_loss

        if level_loss.isnan():
            logging.info(f"NaN loss encountered at level {level}!")
            raise ValueError(f"NaN loss encountered at level {level}!")

        total_loss = level_loss if total_loss is None else total_loss + level_loss

    output["loss"] = total_loss

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)

    return output


def create_world_model(cfg):
    from loss import SIGReg
    from models.encoders.build_encoder import build_encoder
    from models.encoders.seq_encoder import SequenceEncoder
    from models.hjepa import HJEPA
    from models.jepa import FusionEncoder, JEPA, ProjectedEncoder, ProjectedPredictor
    from models.module import Embedder, build_projector
    from models.predictors.predictors import ARPredictor

    jepas = []
    embed_dims = {}
    pixel_embed_dims = {}
    proprio_embed_dims = {}

    def _plain_cfg(node):
        if node is None:
            return {}
        return {k: v for k, v in node.items()}

    def _split_projector_cfg(encoder_cfg):
        cfg = _plain_cfg(encoder_cfg)
        projector_cfg = cfg.pop("projector", None)
        return cfg, projector_cfg

    def _projector_output_dim(projector_cfg, default_dim):
        return int(projector_cfg.get("output_dim", default_dim))

    def _build_projected_encoder(encoder_cfg, projector_cfg, output_dim):
        base_encoder, hidden_dim = build_encoder(
            encoder_cfg,
            default_patch_size=cfg.patch_size,
            default_image_size=cfg.img_size,
        )
        projector = build_projector(
            projector_cfg,
            input_dim=hidden_dim,
            output_dim=output_dim,
        )
        return ProjectedEncoder(base_encoder, projector), int(hidden_dim)

    for level in range(1, int(cfg.num_levels) + 1):
        level_name = f"level{level}"
        level_cfg = cfg[level_name]
        action_encoder_cfg = level_cfg.get("action_encoder", {})
        queue_size = int(action_encoder_cfg.get("queue_size", 0))
        temporal_stride = int(level_cfg.get("stride", 1))
        temporal_window_size = int(level_cfg.get("window_size", 1))
        rollout_n = int(level_cfg.wm.get("rollout_n", 1))
        if rollout_n <= 0:
            raise ValueError(f"level{level}.wm.rollout_n must be positive.")
        rollout_loss_weight = float(level_cfg.wm.get("rollout_loss_weight", 1.0))
        if rollout_loss_weight < 0:
            raise ValueError(
                f"level{level}.wm.rollout_loss_weight must be non-negative."
            )
        target_length = int(level_cfg.wm.history_size) + rollout_n

        raw_encoder_cfg = _plain_cfg(level_cfg.encoder)
        use_proprio = bool(level_cfg.wm.get("use_proprio", False))

        if use_proprio:
            fusion_cfg, fusion_projector_cfg = _split_projector_cfg(raw_encoder_cfg)
            pixel_spec = fusion_cfg.pop("pixel_encoder")
            proprio_spec = fusion_cfg.pop("proprio_encoder")

            pixel_encoder_cfg = _plain_cfg(pixel_spec["encoder"])
            pixel_projector_cfg = pixel_spec["projector"]
            pixel_base_encoder, pixel_hidden_dim = build_encoder(
                pixel_encoder_cfg,
                default_patch_size=cfg.patch_size,
                default_image_size=cfg.img_size,
            )
            pixel_output_dim = _projector_output_dim(
                pixel_projector_cfg,
                pixel_hidden_dim,
            )
            pixel_encoder = ProjectedEncoder(
                pixel_base_encoder,
                build_projector(
                    pixel_projector_cfg,
                    input_dim=pixel_hidden_dim,
                    output_dim=pixel_output_dim,
                ),
            )

            proprio_dim = level_cfg.wm.proprio_dim
            proprio_emb_dim = level_cfg.wm.proprio_emb_dim
            proprio_encoder_cfg = _plain_cfg(proprio_spec["encoder"])
            proprio_projector_cfg = proprio_spec["projector"]
            proprio_encoder_cfg.setdefault("input_dim", int(proprio_dim))
            proprio_encoder_cfg.setdefault("output_dim", int(proprio_emb_dim))
            proprio_output_dim = _projector_output_dim(
                proprio_projector_cfg,
                proprio_encoder_cfg["output_dim"],
            )
            proprio_encoder, _ = _build_projected_encoder(
                proprio_encoder_cfg,
                proprio_projector_cfg,
                proprio_output_dim,
            )

            hidden_dim = int(pixel_hidden_dim)
            embed_dim = int(level_cfg.wm.get("embed_dim", hidden_dim))
            embed_dims[level] = embed_dim
            pixel_embed_dims[level] = int(pixel_output_dim)
            proprio_embed_dims[level] = int(proprio_output_dim)

            encoder = FusionEncoder(
                pixel_encoder=pixel_encoder,
                proprio_encoder=proprio_encoder,
                projector=build_projector(
                    fusion_projector_cfg,
                    input_dim=int(pixel_output_dim) + int(proprio_output_dim),
                    output_dim=embed_dim,
                ),
                proprio_key="proprio" if level == 1 else "proprio_input",
            )
        else:
            encoder_cfg, encoder_projector_cfg = _split_projector_cfg(
                raw_encoder_cfg
            )
            encoder, hidden_dim = build_encoder(
                encoder_cfg,
                default_patch_size=cfg.patch_size,
                default_image_size=cfg.img_size,
            )
            embed_dim = int(level_cfg.wm.get("embed_dim", hidden_dim))
            embed_dims[level] = embed_dim
            encoder = ProjectedEncoder(
                encoder,
                build_projector(
                    encoder_projector_cfg,
                    input_dim=hidden_dim,
                    output_dim=embed_dim,
                ),
            )


        predictor_cfg = _plain_cfg(level_cfg.predictor)
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

        action_pooler = None
        if "action_pooler" in level_cfg:
            action_pooler_kwargs = {k: v for k, v in level_cfg.action_pooler.items()}
            action_pooler_type = str(
                action_pooler_kwargs.pop("type", "sequence")
            ).lower()
            if action_pooler_type == "sequence":
                action_pooler = SequenceEncoder(**action_pooler_kwargs)
            else:
                raise ValueError(
                    f"Unsupported action_pooler type {action_pooler_type!r}"
                )

        action_encoder_kwargs = {k: v for k, v in action_encoder_cfg.items()}
        action_encoder_kwargs.pop("queue_size", None)
        if level == 1:
            effective_act_dim = cfg.data.dataset.level1.frameskip * cfg.level1.wm.action_dim
            action_encoder_kwargs["input_dim"] = effective_act_dim
        action_encoder_kwargs["emb_dim"] = embed_dim
        action_encoder = Embedder(**action_encoder_kwargs)

        jepa = JEPA(
            encoder=encoder,
            predictor=predictor,
            action_encoder=action_encoder,
            action_pooler=action_pooler,
            level=level,
            action_queue_size=queue_size,
            temporal_stride=temporal_stride,
            temporal_window_size=temporal_window_size,
            target_length=target_length,
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

    return world_model, losses, embed_dims
