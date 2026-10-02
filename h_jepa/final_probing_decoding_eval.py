from __future__ import annotations

import math
import os
import re
from pathlib import Path

import lightning as pl
import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger as logging
from omegaconf import OmegaConf
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from data import (
    Compose,
    NORMALIZER_ARTIFACT_FILENAME,
    build_hdf5_dataset,
    get_column_normalizer_from_artifact,
    load_normalizer_artifact,
)
from models.module import CLSDecoder
from models.probers import build_prober
from utils import (
    load_training_config_for_checkpoint,
    resolve_model_checkpoint_path,
)


FINAL_PROBING_DECODING_EVAL_DIR = "final_probing_decoding_eval"
FINAL_PROBING_DECODING_EVAL_PREFIX = "final_probing_decoding_eval"
DECODINGS_DIRNAME = "decodings"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_probing_config(config_name):
    return OmegaConf.load(Path(__file__).resolve().parent / "config" / "probing" / f"{config_name}.yaml")


def _plain_container(node) -> dict:
    if node is None:
        return {}
    if OmegaConf.is_config(node):
        return OmegaConf.to_container(node, resolve=True)
    return dict(node)


def _cache_dir(cfg) -> str | None:
    cache_dir = cfg.get("cache_dir", None)
    if cache_dir not in (None, "", "null"):
        return str(cache_dir)
    return os.environ.get("HJEPA_HOME", None)


def _normalizer_path_for_policy(policy_path: str | Path) -> Path:
    return Path(policy_path).expanduser().parent / NORMALIZER_ARTIFACT_FILENAME


def _load_policy(policy, cache_dir):
    ckpt_path = resolve_model_checkpoint_path(str(policy), cache_dir)
    model = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    train_cfg = load_training_config_for_checkpoint(ckpt_path)
    return model, ckpt_path, train_cfg


def _freeze_model(model: nn.Module) -> nn.Module:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def _normalize_uint8_images_on_device(batch):
    for key, value in list(batch.items()):
        if not key.startswith("pixels"):
            continue
        if not torch.is_tensor(value) or value.dtype != torch.uint8:
            continue
        if value.ndim < 4 or value.size(-3) != 3:
            raise ValueError(
                f"Expected uint8 pixel tensor '{key}' to have channel-first RGB "
                f"shape (..., 3, H, W), got {tuple(value.shape)}"
            )

        view_shape = [1] * value.ndim
        view_shape[-3] = 3
        mean = value.new_tensor(IMAGENET_MEAN, dtype=torch.float32).view(*view_shape)
        std = value.new_tensor(IMAGENET_STD, dtype=torch.float32).view(*view_shape)
        batch[key] = value.float().div(255.0).sub(mean).div(std)


def _build_datasets(cfg, normalizer_artifact):
    train_dataset_cfg = _plain_container(cfg.train_dataset)

    cache_dir = _cache_dir(cfg)
    train_dataset = build_hdf5_dataset(train_dataset_cfg, cache_dir=cache_dir)
    eval_dataset = build_hdf5_dataset(_plain_container(cfg.eval_dataset), cache_dir=cache_dir)

    # Stored images already match img_size; uint8 pixels are normalized on device.
    extra_transforms = [
        get_column_normalizer_from_artifact(normalizer_artifact, col, col, fill_nan=False)
        for col in train_dataset_cfg["keys_to_load"]
        if not col.startswith("pixels")
    ]

    transform = Compose(*extra_transforms)
    train_dataset.transform = transform
    eval_dataset.transform = transform
    return train_dataset, eval_dataset


def _loader_cfg(cfg, *, train: bool) -> dict:
    base = _plain_container(cfg.get("loader", {}))
    base.setdefault("batch_size", 128)
    base.setdefault("pin_memory", True)
    base.setdefault("persistent_workers", base.get("num_workers", 0) > 0)
    base.setdefault("drop_last", train)
    base.setdefault("shuffle", train)
    if not train:
        base["drop_last"] = False
        base["shuffle"] = False
    if base.get("num_workers", 0) <= 0:
        base.pop("persistent_workers", None)
        base.pop("prefetch_factor", None)
    return base


def _target_columns(cfg) -> list[str]:
    dataset_cfg = _plain_container(cfg.train_dataset)
    columns = []
    for col in dataset_cfg.get("keys_to_load", []) or []:
        if col.startswith("pixels") or col == "action":
            continue
        columns.append(str(col))
    return columns


def _level_cfg(cfg, level: int) -> dict:
    return _plain_container(cfg.get(f"level{level}", {}))


def _probe_enabled(cfg, level: int) -> bool:
    level_cfg = _level_cfg(cfg, level)
    probes_cfg = level_cfg.get("probes", {}) or {}
    if "enabled" in probes_cfg:
        return bool(probes_cfg["enabled"])
    return True


def _train_decoder_enabled(cfg, level: int) -> bool:
    level_cfg = _level_cfg(cfg, level)
    return bool(level_cfg.get("train_decoder", False))


def _embed_dim(model, train_cfg, level: int) -> int:
    dims = getattr(model, "embed_dims", None)
    if isinstance(dims, dict) and level in dims:
        return int(dims[level])
    return int(train_cfg[f"level{level}"].wm.embed_dim)


def _validate_dense_assumptions(model) -> None:
    for level in range(1, int(model.num_levels) + 1):
        window_size = int(getattr(model.get_level(level), "temporal_window_size", 1))
        if window_size != 1:
            raise ValueError(
                "Dense final probing/decoding eval currently requires "
                f"window_size=1 for every evaluated level; level {level} has "
                f"window_size={window_size}."
            )


def _dense_encode_no_stride(model, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Encode every level-1 frame through all levels at stride 1 (one embedding per frame)."""
    info = {"pixels": batch["pixels_level1"]}
    if "proprio_level1" in batch:
        info["proprio"] = batch["proprio_level1"]
    output = {"embed_1": model.get_level(1).encode(info, key="pixels")["embed_0"]}

    for level in range(2, int(model.num_levels) + 1):
        input_key = f"embed_{level - 1}"
        level_out = model.get_level(level).encode({input_key: output[input_key]}, key=input_key)
        output[f"embed_{level}"] = level_out["embed_0"]
    return output


def _flatten_time(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.size(0) * x.size(1), *x.shape[2:])


def _optimizer(heads, cfg):
    param_groups = []
    for module in (heads.probers, heads.decoders):
        parameters = [param for param in module.parameters() if param.requires_grad]
        if parameters:
            param_groups.append({"params": parameters})
    return torch.optim.AdamW(
        param_groups,
        lr=float(cfg.optimizer.lr),
        weight_decay=float(cfg.optimizer.weight_decay),
    )


class FinalProbeDecodeHeads(nn.Module):
    def __init__(self, model, train_dataset, cfg, train_cfg, normalizer_artifact):
        super().__init__()
        self.model = _freeze_model(model)
        self.target_columns = _target_columns(cfg)
        self.probers = nn.ModuleDict()
        self.decoders = nn.ModuleDict()
        self.prober_metadata = {}
        self.decoder_metadata = {}
        self.prober_target_std = {}
        normalizer_stats = normalizer_artifact.get("stats", {})

        for level in range(1, int(self.model.num_levels) + 1):
            input_dim = _embed_dim(self.model, train_cfg, level)
            if _probe_enabled(cfg, level):
                for col in self.target_columns:
                    name = f"level{level}_probe_{col}"
                    output_dim = int(train_dataset.get_dim(col))
                    self.probers[name] = build_prober(
                        input_dim=input_dim,
                        output_dim=output_dim,
                    )
                    self.prober_metadata[name] = {"level": level, "target": col}
                    if col not in normalizer_stats:
                        raise KeyError(
                            f"Normalizer artifact is missing probe target {col!r}."
                        )
                    self.prober_target_std[name] = torch.as_tensor(
                        normalizer_stats[col]["std"],
                        dtype=torch.float32,
                    ).reshape(1, -1)

            if _train_decoder_enabled(cfg, level):
                decoder_kwargs = _plain_container(cfg.decoder)
                decoder_name = f"decoder_level{level}"
                self.decoders[decoder_name] = CLSDecoder(
                    cls_dim=input_dim,
                    img_size=int(train_cfg.img_size),
                    patch_size=int(train_cfg.patch_size),
                    **decoder_kwargs,
                )
                self.decoder_metadata[decoder_name] = {
                    "level": level,
                    "cls_dim": input_dim,
                    "img_size": int(train_cfg.img_size),
                    "patch_size": int(train_cfg.patch_size),
                    "decoder_kwargs": decoder_kwargs,
                }

    def _batch_losses(self, batch):
        """Training loss, plus per-metric (summed error, count) for this batch.

        Probe errors are de-normalized squared errors summed per target dimension;
        `_run_epoch` turns them into macro NMSE (mean over dims of MSE_d / Var_d).
        """
        with torch.no_grad():
            encoded = _dense_encode_no_stride(self.model, batch)

        stats = {}
        total_loss = None

        for name, prober in self.probers.items():
            meta = self.prober_metadata[name]
            level = int(meta["level"])
            pred = prober(_flatten_time(encoded[f"embed_{level}"]))
            target = _flatten_time(batch[f"{meta['target']}_level1"]).float()
            valid = ~torch.isnan(target).any(dim=-1)
            if not bool(valid.any()):
                continue
            pred = pred[valid]
            target = target[valid]
            loss = F.mse_loss(pred, target)
            std = self.prober_target_std[name].to(device=pred.device, dtype=pred.dtype)
            squared_error = torch.square((pred - target).detach() * std)
            stats[f"probe/{name}"] = (squared_error.sum(dim=0), squared_error.size(0))
            total_loss = loss if total_loss is None else total_loss + loss

        pixels = _flatten_time(batch["pixels_level1"]).float()
        for name, decoder in self.decoders.items():
            level = int(self.decoder_metadata[name]["level"])
            loss = F.mse_loss(decoder(_flatten_time(encoded[f"embed_{level}"])), pixels)
            stats[f"decoder/{name}"] = (loss.detach() * pixels.numel(), pixels.numel())
            total_loss = loss if total_loss is None else total_loss + loss

        return total_loss, stats


def _move_batch(batch, device):
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def _dataset_col_variance(dataset, col: str, device) -> torch.Tensor:
    data = np.asarray(dataset.get_col_data(col))
    flat = data.reshape(data.shape[0], -1)
    valid_mask = ~np.isnan(flat).any(axis=1)
    valid = data[valid_mask].astype(np.float64, copy=False)
    if valid.shape[0] == 0:
        raise ValueError(f"Column {col!r} has no finite rows.")
    std = valid.std(
        axis=0,
        ddof=1 if valid.shape[0] > 1 else 0,
        keepdims=True,
    )

    return torch.as_tensor(std, dtype=torch.float32, device=device).reshape(-1).square()


def _probe_variances(heads, dataset, device) -> dict[str, torch.Tensor]:
    variances = {}
    for name, meta in heads.prober_metadata.items():
        variances[name] = _dataset_col_variance(dataset, meta["target"], device).clamp_min(1e-12)
    return variances


def _log_metrics(logger, wandb_run, metrics: dict[str, float], step: int | None):
    if logger is not None:
        logger.log_metrics(metrics, step=step)
    if wandb_run is not None:
        wandb_run.log(metrics, step=step)


def _prefixed(prefix: str, key: str) -> str:
    return f"{prefix}/{key}" if prefix else key


def _build_unroll_figure(frames_gt, frames_decoded, decoded_label):
    num_clips = frames_gt.shape[0]
    num_cols_per_clip = frames_gt.shape[1]
    clips_per_block = 8
    spacer_cols = 1
    num_blocks = max((num_clips + clips_per_block - 1) // clips_per_block, 1)
    num_rows = min(num_clips, clips_per_block) * 2
    num_cols = num_cols_per_clip * num_blocks + spacer_cols * (num_blocks - 1)
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(max(2 * num_cols, 1), max(2 * num_rows, 1)),
        squeeze=False,
    )

    for row in range(num_rows):
        for col in range(num_cols):
            axes[row, col].axis("off")

    for clip_idx in range(num_clips):
        block_idx = clip_idx // clips_per_block
        row_offset = clip_idx % clips_per_block
        col_offset = block_idx * (num_cols_per_clip + spacer_cols)

        gt_row = 2 * row_offset
        pred_row = gt_row + 1

        for t in range(num_cols_per_clip):
            col = col_offset + t
            axes[gt_row, col].imshow(np.transpose(frames_gt[clip_idx, t], (1, 2, 0)))
            axes[pred_row, col].imshow(
                np.transpose(frames_decoded[clip_idx, t], (1, 2, 0))
            )
            axes[gt_row, col].axis("off")
            axes[pred_row, col].axis("off")
            if gt_row == 0:
                axes[gt_row, col].set_title(f"t={t}")

        axes[gt_row, col_offset].set_ylabel(
            f"Clip {clip_idx} GT",
            rotation=0,
            labelpad=40,
            va="center",
        )
        axes[pred_row, col_offset].set_ylabel(
            f"Clip {clip_idx} {decoded_label}",
            rotation=0,
            labelpad=40,
            va="center",
        )

    fig.tight_layout()
    return fig


def _unnormalize_image_tensor(images: torch.Tensor) -> torch.Tensor:
    view_shape = [1] * images.ndim
    view_shape[-3] = 3
    mean = images.new_tensor(IMAGENET_MEAN).view(*view_shape)
    std = images.new_tensor(IMAGENET_STD).view(*view_shape)
    return torch.clamp(images * std + mean, 0, 1)


def _collate_visualization_batch(dataset, num_frames=32):
    num_frames = min(num_frames, len(dataset))
    if num_frames <= 0:
        return None
    if num_frames == 1:
        indices = [0]
    else:
        indices = torch.linspace(0, len(dataset) - 1, steps=num_frames).long().tolist()
    samples = [dataset[int(idx)] for idx in indices]
    return torch.utils.data.default_collate(samples)


@torch.no_grad()
def _decoder_visualization_figures(heads, dataset, device):
    if not heads.decoders:
        return {}

    batch = _collate_visualization_batch(dataset)
    if batch is None:
        return {}

    batch = _move_batch(batch, device)
    _normalize_uint8_images_on_device(batch)
    encoded = _dense_encode_no_stride(heads.model, batch)
    pixels = batch.get("pixels_level1", None)
    if pixels is None:
        return {}

    pixels = pixels.float()

    payload = {}
    for name, decoder in heads.decoders.items():
        meta = heads.decoder_metadata[name]
        level = int(meta["level"])
        embed = _flatten_time(encoded[f"embed_{level}"])
        decoded = decoder(embed)
        decoded = decoded.reshape(
            pixels.size(0),
            pixels.size(1),
            *decoded.shape[1:],
        )

        frames = (_unnormalize_image_tensor(pixels).detach().cpu() * 255).byte().numpy()
        frames_decoded = (
            _unnormalize_image_tensor(decoded).detach().cpu() * 255
        ).byte().numpy()
        fig = _build_unroll_figure(frames, frames_decoded, "Enc")
        payload[
            _prefixed(
                FINAL_PROBING_DECODING_EVAL_PREFIX,
                f"eval/level{level}_embed_unroll",
            )
        ] = fig

    return payload


def _limit_batches(cfg, key: str) -> int | None:
    value = cfg.get(key, None)
    return None if value is None else int(value)


def _run_epoch(heads, loader, device, probe_variances, limit_batches=None, optimizer=None):
    """One pass over `loader`; trains the heads when an optimizer is given."""
    training = optimizer is not None
    heads.train(training)
    heads.model.eval()
    losses = []
    totals = {}
    counts = {}

    with torch.set_grad_enabled(training):
        for batch_idx, batch in enumerate(loader):
            if limit_batches is not None and batch_idx >= limit_batches:
                break
            batch = _move_batch(batch, device)
            _normalize_uint8_images_on_device(batch)
            if training:
                optimizer.zero_grad(set_to_none=True)
            loss, stats = heads._batch_losses(batch)
            if training:
                loss.backward()
                optimizer.step()

            losses.append(float(loss.detach().cpu()))
            for key, (total, count) in stats.items():
                totals[key] = totals[key] + total if key in totals else total
                counts[key] = counts.get(key, 0) + count

    results = {"loss": _mean(losses)}
    for key, total in totals.items():
        mean = total / counts[key]
        if key.startswith("probe/"):
            mean = (mean / probe_variances[key.removeprefix("probe/")]).mean()
        results[key] = float(mean)
    return results


def _metric_name(raw_key: str) -> str:
    if raw_key.startswith("probe/"):
        name = raw_key.removeprefix("probe/")
        return f"{name}_nmse"
    if raw_key.startswith("decoder/"):
        name = raw_key.removeprefix("decoder/")
        level = name.removeprefix("decoder_level")
        return f"decoder_loss_level{level}"
    return raw_key


def _format_epoch_metrics(train_metrics, eval_metrics, prefix):
    formatted = {}
    formatted[_prefixed(prefix, "train/loss_epoch")] = train_metrics.get("loss", math.nan)

    for key, value in train_metrics.items():
        if key != "loss":
            formatted[_prefixed(prefix, f"train/{_metric_name(key)}")] = value

    formatted[_prefixed(prefix, "eval/loss")] = eval_metrics.get("loss", math.nan)
    for key, value in eval_metrics.items():
        if key != "loss":
            formatted[_prefixed(prefix, f"eval/{_metric_name(key)}")] = value

    return formatted


def _save_decoder_visualizations(output_dir: Path, heads, dataset, device) -> None:
    figures = _decoder_visualization_figures(heads, dataset, device)
    if not figures:
        return

    decodings_dir = output_dir / DECODINGS_DIRNAME
    decodings_dir.mkdir(parents=True, exist_ok=True)
    for key, fig in figures.items():
        filename = key.removeprefix(f"{FINAL_PROBING_DECODING_EVAL_PREFIX}/")
        filename = re.sub(r"[^A-Za-z0-9_.-]+", "_", filename).strip("_")
        try:
            fig.savefig(decodings_dir / f"{filename}.png", dpi=150, bbox_inches="tight")
        finally:
            plt.close(fig)


def _save_artifacts(heads, output_dir: Path, cfg, metrics) -> None:
    for name, module in heads.decoders.items():
        meta = heads.decoder_metadata[name]
        level = int(meta["level"])
        torch.save(
            {
                "format": "final_probing_decoding_eval_decoder_v1",
                "level": level,
                "decoder_state_dict": module.state_dict(),
                "decoder_kwargs": meta["decoder_kwargs"],
                "cls_dim": meta["cls_dim"],
                "img_size": meta["img_size"],
                "patch_size": meta["patch_size"],
            },
            output_dir / f"decoder_level{level}.ckpt",
        )
    OmegaConf.save(cfg, output_dir / "config.yaml")
    OmegaConf.save(OmegaConf.create(metrics), output_dir / "metrics.yaml")


def _init_wandb(cfg, output_dir: Path):
    wandb_cfg = cfg.get("wandb", {}) or {}
    if not bool(wandb_cfg.get("enabled", False)):
        return None

    import wandb

    init_cfg = _plain_container(wandb_cfg.get("config", {}))
    # Prefer an explicit configured name/id;
    # fall back to the output dir name for standalone one-off runs.
    run_name = init_cfg.get("name") or output_dir.name
    init_cfg["name"] = run_name
    init_cfg["id"] = init_cfg.get("id") or run_name
    init_cfg.setdefault("resume", "allow")
    return wandb.init(**init_cfg)


def run_final_probing_decoding_eval(
    cfg,
    *,
    model=None,
    policy_path: str | Path | None = None,
    train_cfg=None,
    normalizer_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    logger=None,
    log_step: int | None = None,
):
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cache_dir = _cache_dir(cfg)

    if model is None:
        if cfg.get("policy", None) in (None, "", "null"):
            raise ValueError("cfg.policy must be set when no in-memory model is provided.")
        model, resolved_policy_path, train_cfg = _load_policy(cfg.policy, cache_dir)
        policy_path = resolved_policy_path
    else:
        if train_cfg is None:
            if policy_path is None:
                raise ValueError("train_cfg or policy_path is required for in-memory eval.")
            train_cfg = load_training_config_for_checkpoint(policy_path)

    policy_path = None if policy_path is None else Path(policy_path).expanduser()
    if normalizer_path is None:
        if policy_path is None:
            raise ValueError("normalizer_path is required for in-memory eval.")
        normalizer_path = _normalizer_path_for_policy(policy_path)
    normalizer_path = Path(normalizer_path).expanduser()
    if not normalizer_path.exists():
        raise FileNotFoundError(
            "Training normalizer artifact not found for final probing/decoding "
            f"eval. Expected: {normalizer_path}"
        )
    normalizer_artifact = load_normalizer_artifact(normalizer_path)

    model = _freeze_model(model)
    _validate_dense_assumptions(model)

    if output_dir is None and cfg.get("output_dir", None) not in (None, "", "null"):
        output_dir = cfg.output_dir

    if output_dir is None:
        if policy_path is None:
            raise ValueError("output_dir is required when policy_path is not set.")
        output_subdir = str(cfg.get("output_subdir", FINAL_PROBING_DECODING_EVAL_DIR))
        output_dir = policy_path.parent / output_subdir
    output_dir = Path(output_dir).expanduser()

    seed = int(cfg.get("seed", 42))
    pl.seed_everything(seed, workers=True)

    train_dataset, eval_dataset = _build_datasets(cfg, normalizer_artifact)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        **_loader_cfg(cfg, train=True),
        generator=generator,
    )
    eval_loader = DataLoader(eval_dataset, **_loader_cfg(cfg, train=False))

    device_cfg = str(cfg.get("device", "auto"))
    if device_cfg == "auto":
        device_cfg = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_cfg)
    heads = FinalProbeDecodeHeads(
        model,
        train_dataset,
        cfg,
        train_cfg,
        normalizer_artifact,
    ).to(device)
    optimizer = _optimizer(heads, cfg)
    train_probe_variances = _probe_variances(heads, train_dataset, device)
    eval_probe_variances = _probe_variances(heads, eval_dataset, device)

    wandb_run = None
    if logger is None:
        wandb_run = _init_wandb(cfg, output_dir)

    epochs = int(cfg.epochs)

    train_batch_limit = _limit_batches(cfg, "limit_train_batches")
    eval_batch_limit = _limit_batches(cfg, "limit_eval_batches")
    final_metrics = {}
    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(
            heads,
            train_loader,
            device,
            train_probe_variances,
            limit_batches=train_batch_limit,
            optimizer=optimizer,
        )
        eval_metrics = _run_epoch(
            heads,
            eval_loader,
            device,
            eval_probe_variances,
            limit_batches=eval_batch_limit,
        )
        final_metrics = _format_epoch_metrics(
            train_metrics,
            eval_metrics,
            FINAL_PROBING_DECODING_EVAL_PREFIX,
        )
        step = None if log_step is None else int(log_step) + epoch
        _log_metrics(logger, wandb_run, final_metrics, step)

        logging.info(
            "final_probing_decoding_eval epoch "
            f"{epoch}/{epochs}: train_loss={train_metrics.get('loss'):.6f}, "
            f"eval_loss={eval_metrics.get('loss'):.6f}"
        )

    if bool(cfg.get("save_artifacts", True)):
        output_dir.mkdir(parents=True, exist_ok=True)
        _save_decoder_visualizations(output_dir, heads, eval_dataset, device)
        _save_artifacts(heads, output_dir, cfg, final_metrics)
        logging.info(f"Saved final probing/decoding eval artifacts to {output_dir}")
    if wandb_run is not None:
        wandb_run.finish()
    return final_metrics


class FinalProbingDecodingEvalCallback(pl.Callback):
    def __init__(self, *, eval_cfg, run_dir, output_subdir=FINAL_PROBING_DECODING_EVAL_DIR):
        super().__init__()
        self.eval_cfg = eval_cfg
        self.run_dir = Path(run_dir)
        self.output_subdir = output_subdir
        self._ran = False

    @property
    def enabled(self) -> bool:
        return bool(self.eval_cfg.get("enabled", False))

    def on_train_end(self, trainer, pl_module):
        if not self.enabled or self._ran:
            return

        trainer.strategy.barrier("final_probing_decoding_eval_start")
        if trainer.is_global_zero:
            cfg = load_probing_config(self.eval_cfg.config_name)
            cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
            overrides = self.eval_cfg.get("overrides", None)
            if overrides is not None:
                cfg = OmegaConf.merge(cfg, overrides)

            output_model_name = getattr(pl_module, "_output_model_name", "model")
            policy_path = self.run_dir / f"{output_model_name}_object.ckpt"

            run_final_probing_decoding_eval(
                cfg,
                model=pl_module.model,
                policy_path=policy_path,
                train_cfg=getattr(pl_module, "_train_cfg", None),
                normalizer_path=self.run_dir / NORMALIZER_ARTIFACT_FILENAME,
                output_dir=self.run_dir / self.output_subdir,
                logger=trainer.logger,
                log_step=trainer.global_step,
            )
            self._ran = True
        trainer.strategy.barrier("final_probing_decoding_eval_end")
