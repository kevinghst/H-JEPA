from __future__ import annotations

import csv
import math
import os
import re
from pathlib import Path
from typing import Any

import lightning as pl
import matplotlib.pyplot as plt
import numpy as np
import stable_pretraining as spt
import torch
from loguru import logger as logging
from omegaconf import OmegaConf
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from data import (
    NORMALIZER_ARTIFACT_FILENAME,
    build_hdf5_dataset,
    get_column_normalizer_from_artifact,
    get_hdf5_image_shape,
    get_img_preprocessor,
    image_shape_matches_size,
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
HEADS_FILENAME = "heads.ckpt"
MANIFEST_FILENAME = "manifest.yaml"
EVAL_METRICS_FILENAME = "eval_metrics.yaml"
EVAL_METRICS_HISTORY_FILENAME = "eval_metrics_history.yaml"
PROBE_DIM_METRICS_FILENAME = "probe_dim_metrics.csv"
PROBE_SUMMARY_METRICS_FILENAME = "probe_summary_metrics.csv"
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
    return os.environ.get("STABLEWM_HOME", None)


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


def _can_skip_cpu_image_preprocess(train_dataset, eval_datasets, col, img_size):
    train_shape = get_hdf5_image_shape(train_dataset, col)
    if not image_shape_matches_size(train_shape, img_size):
        return False

    for eval_dataset in eval_datasets:
        eval_shape = get_hdf5_image_shape(eval_dataset, col)
        if train_shape != eval_shape or not image_shape_matches_size(eval_shape, img_size):
            return False
    return True


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


def _build_datasets(cfg, normalizer_artifact, train_cfg):
    train_dataset_cfg = _plain_container(cfg.train_dataset)
    eval_dataset_cfgs = _plain_container(cfg.eval_datasets)

    cache_dir = _cache_dir(cfg)
    train_dataset = build_hdf5_dataset(
        train_dataset_cfg,
        cache_dir=cache_dir,
    )

    eval_datasets = {}
    for eval_key, eval_dataset_cfg in eval_dataset_cfgs.items():
        eval_dataset = build_hdf5_dataset(
            eval_dataset_cfg,
            cache_dir=cache_dir,
        )
        eval_datasets[eval_key] = eval_dataset

    img_size = int(train_cfg.img_size)
    extra_transforms = []
    for col in train_dataset_cfg.get("keys_to_load", []) or []:
        if col.startswith("pixels"):
            if _can_skip_cpu_image_preprocess(
                train_dataset,
                eval_datasets.values(),
                col,
                img_size,
            ):
                logging.info(
                    f"Skipping CPU image preprocessing for '{col}' because stored "
                    f"images already match img_size={img_size}."
                )
                continue
            extra_transforms.append(get_img_preprocessor(col, col, img_size))
            continue
        extra_transforms.append(
            get_column_normalizer_from_artifact(
                normalizer_artifact, col, col, fill_nan=False
            )
        )

    transform = spt.data.transforms.Compose(*extra_transforms)
    train_dataset.transform = transform
    for eval_dataset in eval_datasets.values():
        eval_dataset.transform = transform
    return train_dataset, eval_datasets


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
    output = {}
    info = {
        "pixels": batch["pixels_level1"],
    }
    if "proprio_level1" in batch:
        info["proprio"] = batch["proprio_level1"]

    level1_out = model.get_level(1).encode(info, key="pixels")
    for name in ("embed", "pixel_embed", "proprio_embed"):
        key = f"{name}_0"
        if key in level1_out:
            output[f"{name}_1"] = level1_out[key]

    for level in range(2, int(model.num_levels) + 1):
        jepa = model.get_level(level)
        encoder = getattr(jepa, "encoder", None)
        uses_proprio = (
            hasattr(encoder, "pixel_encoder") and hasattr(encoder, "proprio_encoder")
        )
        if uses_proprio:
            input_key = f"pixel_embed_{level - 1}"
            proprio_key = f"proprio_embed_{level - 1}"
            model_proprio_key = getattr(encoder, "proprio_key", "proprio")
            level_info = {
                input_key: output[input_key],
                model_proprio_key: output[proprio_key],
            }
        else:
            input_key = f"embed_{level - 1}"
            level_info = {input_key: output[input_key]}

        level_out = jepa.encode(level_info, key=input_key, chunk_temporal_inputs=False)
        for name in ("embed", "pixel_embed", "proprio_embed"):
            key = f"{name}_0"
            if key in level_out:
                output[f"{name}_{level}"] = level_out[key]

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
                    self.prober_metadata[name] = {
                        "level": level,
                        "target": col,
                        "input": f"embed_{level}",
                        "input_dim": input_dim,
                        "output_dim": output_dim,
                    }
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
                    "input": f"embed_{level}",
                    "cls_dim": input_dim,
                    "img_size": int(train_cfg.img_size),
                    "patch_size": int(train_cfg.patch_size),
                    "decoder_kwargs": decoder_kwargs,
                }

    def _batch_losses(self, batch, probe_metric_denominators=None):
        with torch.no_grad():
            encoded = _dense_encode_no_stride(self.model, batch)

        losses = {}
        metrics = {}
        metric_weights = {}
        total_loss = None

        for name, prober in self.probers.items():
            meta = self.prober_metadata[name]
            level = int(meta["level"])
            target_col = meta["target"]
            pred = prober(_flatten_time(encoded[f"embed_{level}"]))
            target = _flatten_time(batch[f"{target_col}_level1"]).float()
            valid = ~torch.isnan(target).any(dim=-1)
            if not bool(valid.any()):
                continue
            pred = pred[valid]
            target = target[valid]
            loss = F.mse_loss(pred, target)
            losses[f"probe/{name}"] = loss
            if probe_metric_denominators is None:
                metrics[f"probe/{name}"] = loss
                metric_weights[f"probe/{name}"] = target.numel()
            else:
                std = self.prober_target_std[name].to(
                    device=pred.device,
                    dtype=pred.dtype,
                )
                raw_squared_error = torch.square((pred - target) * std)
                denominator = probe_metric_denominators[name].to(
                    device=pred.device,
                    dtype=pred.dtype,
                )
                metrics[f"probe/{name}"] = raw_squared_error.mean() / denominator
                metric_weights[f"probe/{name}"] = raw_squared_error.numel()
            total_loss = loss if total_loss is None else total_loss + loss

        if "pixels_level1" in batch:
            pixels = _flatten_time(batch["pixels_level1"]).float()
            for name, decoder in self.decoders.items():
                meta = self.decoder_metadata[name]
                level = int(meta["level"])
                embed = _flatten_time(encoded[f"embed_{level}"])
                if embed.ndim != 2:
                    raise ValueError(
                        f"{name} expects rank-2 flattened embeddings, got "
                        f"{tuple(embed.shape)}."
                    )
                pred_pixels = decoder(embed)
                loss = F.mse_loss(pred_pixels, pixels)
                losses[f"decoder/{name}"] = loss
                metrics[f"decoder/{name}"] = loss
                metric_weights[f"decoder/{name}"] = pixels.numel()
                total_loss = loss if total_loss is None else total_loss + loss

        if total_loss is None:
            raise ValueError("No final probing/decoding heads are configured.")

        return total_loss, losses, metrics, metric_weights


def _move_batch(batch, device):
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def _weighted_metric_means(
    metric_totals: dict[str, float],
    metric_weights: dict[str, int],
) -> dict[str, float]:
    return {
        key: total / metric_weights[key]
        for key, total in metric_totals.items()
        if metric_weights.get(key, 0) > 0
    }


def _dataset_col_variance(dataset, col: str, device) -> torch.Tensor:
    if hasattr(dataset, "get_col_stats"):
        _, std = dataset.get_col_stats(col)
    else:
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


def _probe_metric_denominators(heads, dataset, device) -> dict[str, torch.Tensor]:
    denominators = {}
    cache = {}

    for name, meta in heads.prober_metadata.items():
        target_col = meta["target"]
        if target_col not in cache:
            variance = _dataset_col_variance(dataset, target_col, device)
            cache[target_col] = variance.mean().clamp_min(1e-12)
        denominators[name] = cache[target_col]

    return denominators


def _probe_metric_variances(heads, dataset, device) -> dict[str, torch.Tensor]:
    variances = {}
    cache = {}

    for name, meta in heads.prober_metadata.items():
        target_col = meta["target"]
        if target_col not in cache:
            cache[target_col] = _dataset_col_variance(
                dataset,
                target_col,
                device,
            ).clamp_min(1e-12)
        variances[name] = cache[target_col]

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


def _train_one_epoch(
    heads,
    loader,
    optimizer,
    device,
    probe_metric_denominators,
    limit_batches: int | None = None,
):
    heads.train()
    heads.model.eval()
    losses_by_batch: dict[str, list[float]] = {}
    metric_totals: dict[str, float] = {}
    metric_weights: dict[str, int] = {}

    for batch_idx, batch in enumerate(loader):
        if limit_batches is not None and batch_idx >= limit_batches:
            break
        batch = _move_batch(batch, device)
        _normalize_uint8_images_on_device(batch)
        optimizer.zero_grad(set_to_none=True)
        loss, _, metrics, weights = heads._batch_losses(
            batch,
            probe_metric_denominators=probe_metric_denominators,
        )
        loss.backward()
        optimizer.step()

        losses_by_batch.setdefault("loss", []).append(float(loss.detach().cpu()))
        for key, value in metrics.items():
            weight = int(weights[key])
            metric_totals[key] = metric_totals.get(key, 0.0) + (
                float(value.detach().cpu()) * weight
            )
            metric_weights[key] = metric_weights.get(key, 0) + weight

    results = {key: _mean(values) for key, values in losses_by_batch.items()}
    results.update(_weighted_metric_means(metric_totals, metric_weights))
    return results


@torch.no_grad()
def _evaluate(
    heads,
    loader,
    device,
    probe_metric_denominators,
    limit_batches: int | None = None,
):
    heads.eval()
    losses_by_batch: dict[str, list[float]] = {}
    metric_totals: dict[str, float] = {}
    metric_weights: dict[str, int] = {}

    for batch_idx, batch in enumerate(loader):
        if limit_batches is not None and batch_idx >= limit_batches:
            break
        batch = _move_batch(batch, device)
        _normalize_uint8_images_on_device(batch)
        loss, _, metrics, weights = heads._batch_losses(
            batch,
            probe_metric_denominators=probe_metric_denominators,
        )
        losses_by_batch.setdefault("loss", []).append(float(loss.detach().cpu()))
        for key, value in metrics.items():
            weight = int(weights[key])
            metric_totals[key] = metric_totals.get(key, 0.0) + (
                float(value.detach().cpu()) * weight
            )
            metric_weights[key] = metric_weights.get(key, 0) + weight

    results = {key: _mean(values) for key, values in losses_by_batch.items()}
    results.update(_weighted_metric_means(metric_totals, metric_weights))
    return results


def _metric_name(raw_key: str) -> str:
    if raw_key.startswith("probe/"):
        name = raw_key.removeprefix("probe/")
        return f"{name}_mse"
    if raw_key.startswith("decoder/"):
        name = raw_key.removeprefix("decoder/")
        level = name.removeprefix("decoder_level")
        return f"decoder_loss_level{level}"
    return raw_key


def _mean_metrics(metrics_by_key: dict[str, dict[str, float]]) -> dict[str, float]:
    if not metrics_by_key:
        return {}

    metric_names = sorted(
        {metric for metrics in metrics_by_key.values() for metric in metrics}
    )
    mean_metrics = {}
    for metric in metric_names:
        values = [
            metrics[metric]
            for metrics in metrics_by_key.values()
            if metric in metrics and math.isfinite(float(metrics[metric]))
        ]
        if values:
            mean_metrics[metric] = _mean(values)
    return mean_metrics


def _format_epoch_metrics(train_metrics, eval_metrics_by_key, prefix):
    formatted = {}
    formatted[_prefixed(prefix, "train/loss_epoch")] = train_metrics.get("loss", math.nan)

    for key, value in train_metrics.items():
        if key.startswith("probe/"):
            formatted[_prefixed(prefix, f"train/{_metric_name(key)}")] = value
        elif key.startswith("decoder/"):
            formatted[_prefixed(prefix, f"train/{_metric_name(key)}")] = value

    for eval_key, eval_metrics in eval_metrics_by_key.items():
        formatted[_prefixed(prefix, f"eval/{eval_key}/loss")] = eval_metrics.get(
            "loss",
            math.nan,
        )
        for key, value in eval_metrics.items():
            if key == "loss":
                continue
            formatted[_prefixed(prefix, f"eval/{eval_key}/{_metric_name(key)}")] = value

    eval_mean_metrics = _mean_metrics(eval_metrics_by_key)
    formatted[_prefixed(prefix, "eval_mean/loss")] = eval_mean_metrics.get(
        "loss",
        math.nan,
    )
    for key, value in eval_mean_metrics.items():
        if key == "loss":
            continue
        formatted[_prefixed(prefix, f"eval_mean/{_metric_name(key)}")] = value

    return formatted


def _eval_only_metrics(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    eval_prefix = _prefixed(prefix, "eval/")
    eval_mean_prefix = _prefixed(prefix, "eval_mean/")
    return {
        key: value
        for key, value in metrics.items()
        if key.startswith(eval_prefix) or key.startswith(eval_mean_prefix)
    }


@torch.no_grad()
def _evaluate_probe_dim_metrics(
    heads,
    loader,
    device,
    probe_metric_variances,
    *,
    eval_key: str,
    limit_batches: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Per-dim probe rows, plus one summary row per probe.

    The summary carries, beside the pooled NMSE that `_batch_losses` reports
    (mean_d MSE_d / mean_d Var_d, where one wide high-variance dim can mask
    failure on the rest), the macro NMSE (mean_d of MSE_d / Var_d) and the NMSE
    a constant predict-the-eval-mean probe scores under either convention. The
    latter is the measured floor: it is not exactly 1.0, because the numerator
    de-normalises with the train normalizer std while the denominator variance
    is measured on the eval split.
    """
    heads.eval()
    squared_error_totals: dict[str, torch.Tensor] = {}
    target_totals: dict[str, torch.Tensor] = {}
    target_squared_totals: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}

    for batch_idx, batch in enumerate(loader):
        if limit_batches is not None and batch_idx >= limit_batches:
            break
        batch = _move_batch(batch, device)
        _normalize_uint8_images_on_device(batch)
        encoded = _dense_encode_no_stride(heads.model, batch)

        for name, prober in heads.probers.items():
            meta = heads.prober_metadata[name]
            level = int(meta["level"])
            target_col = meta["target"]
            pred = prober(_flatten_time(encoded[f"embed_{level}"]))
            target = _flatten_time(batch[f"{target_col}_level1"]).float()
            valid = ~torch.isnan(target).any(dim=-1)
            if not bool(valid.any()):
                continue
            pred = pred[valid]
            target = target[valid]
            std = heads.prober_target_std[name].to(
                device=pred.device,
                dtype=pred.dtype,
            )
            raw_squared_error = torch.square((pred - target) * std)
            if name not in squared_error_totals:
                squared_error_totals[name] = torch.zeros(
                    raw_squared_error.size(-1),
                    device=raw_squared_error.device,
                    dtype=raw_squared_error.dtype,
                )
                target_totals[name] = torch.zeros_like(squared_error_totals[name])
                target_squared_totals[name] = torch.zeros_like(
                    squared_error_totals[name]
                )
            squared_error_totals[name] += raw_squared_error.sum(dim=0)
            target_totals[name] += target.sum(dim=0)
            target_squared_totals[name] += target.square().sum(dim=0)
            counts[name] = counts.get(name, 0) + raw_squared_error.size(0)

    rows = []
    summary_rows = []
    for name, total in squared_error_totals.items():
        count = counts.get(name, 0)
        if count <= 0:
            continue
        meta = heads.prober_metadata[name]
        variance = probe_metric_variances[name].to(
            device=total.device,
            dtype=total.dtype,
        )
        raw_mse = total / count
        normalized_mse = raw_mse / variance
        # Floor: swap the probe for the constant per-dim eval-split mean and push
        # it through the identical ((pred - target) * std) ** 2 arithmetic.
        std = heads.prober_target_std[name].to(
            device=total.device,
            dtype=total.dtype,
        )
        target_mean = target_totals[name] / count
        floor_raw_mse = (
            target_squared_totals[name] / count - target_mean.square()
        ).clamp_min(0.0) * std.reshape(-1).square()
        summary_rows.append(
            {
                "eval_key": eval_key,
                "level": int(meta["level"]),
                "target": str(meta["target"]),
                "probe_name": name,
                "num_dims": int(raw_mse.numel()),
                "probe_nmse": float((raw_mse.mean() / variance.mean()).cpu()),
                "probe_nmse_macro": float(normalized_mse.mean().cpu()),
                "probe_nmse_floor": float(
                    (floor_raw_mse.mean() / variance.mean()).cpu()
                ),
                "probe_nmse_macro_floor": float((floor_raw_mse / variance).mean().cpu()),
            }
        )
        for dim_index in range(raw_mse.numel()):
            rows.append(
                {
                    "eval_key": eval_key,
                    "level": int(meta["level"]),
                    "target": str(meta["target"]),
                    "probe_name": name,
                    "dim_index": dim_index,
                    "raw_mse": float(raw_mse[dim_index].detach().cpu()),
                    "target_variance": float(variance[dim_index].detach().cpu()),
                    "normalized_mse": float(
                        normalized_mse[dim_index].detach().cpu()
                    ),
                }
            )
    return rows, summary_rows


def _write_probe_dim_metrics(output_dir: Path, rows: list[dict[str, Any]]) -> Path:
    path = output_dir / PROBE_DIM_METRICS_FILENAME
    fieldnames = [
        "eval_key",
        "level",
        "target",
        "probe_name",
        "dim_index",
        "raw_mse",
        "target_variance",
        "normalized_mse",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_probe_summary_metrics(output_dir: Path, rows: list[dict[str, Any]]) -> Path:
    path = output_dir / PROBE_SUMMARY_METRICS_FILENAME
    fieldnames = [
        "eval_key",
        "level",
        "target",
        "probe_name",
        "num_dims",
        "probe_nmse",
        "probe_nmse_macro",
        "probe_nmse_floor",
        "probe_nmse_macro_floor",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


# Logger suffixes for the summary columns, matching the existing `..._mse` keys
# that `_metric_name` produces for the pooled metric.
_PROBE_SUMMARY_LOG_SUFFIXES = {
    "probe_nmse_macro": "mse_macro",
    "probe_nmse_floor": "mse_floor",
    "probe_nmse_macro_floor": "mse_macro_floor",
}


def _probe_summary_log_metrics(
    rows: list[dict[str, Any]],
    prefix: str,
) -> dict[str, float]:
    """Flatten the macro/floor summary into keys sitting beside the pooled ones."""
    metrics = {}
    for row in rows:
        base = _prefixed(prefix, f"eval/{row['eval_key']}/{row['probe_name']}")
        for column, suffix in _PROBE_SUMMARY_LOG_SUFFIXES.items():
            metrics[f"{base}_{suffix}"] = float(row[column])
    return metrics


def _save_decoder_visualizations(
    output_dir: Path,
    heads,
    eval_datasets,
    device,
) -> list[str]:
    eval_key, dataset = next(iter(eval_datasets.items()))
    figures = _decoder_visualization_figures(heads, dataset, device)
    if not figures:
        return []

    decodings_dir = output_dir / DECODINGS_DIRNAME
    decodings_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for key, fig in figures.items():
        filename = key.removeprefix(f"{FINAL_PROBING_DECODING_EVAL_PREFIX}/")
        filename = re.sub(r"[^A-Za-z0-9_.-]+", "_", filename).strip("_")
        path = decodings_dir / f"{eval_key}_{filename}.png"
        try:
            fig.savefig(path, dpi=150, bbox_inches="tight")
        finally:
            plt.close(fig)
        saved.append(str(path.relative_to(output_dir)))
    return saved


def _save_artifacts(
    heads,
    output_dir: Path,
    cfg,
    train_cfg,
    metrics,
    eval_metrics,
    eval_metrics_history,
    probe_dim_metrics,
    probe_summary_metrics,
    decoder_visualization_files,
    policy_path,
    normalizer_path,
    eval_dataset_keys,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    heads_path = output_dir / HEADS_FILENAME
    payload = {
        "format": "final_probing_decoding_eval_heads_v1",
        "probers": {
            name: module.state_dict()
            for name, module in heads.probers.items()
        },
        "decoders": {
            name: module.state_dict()
            for name, module in heads.decoders.items()
        },
        "prober_metadata": heads.prober_metadata,
        "decoder_metadata": heads.decoder_metadata,
    }
    torch.save(payload, heads_path)

    decoder_files = {}
    for name, module in heads.decoders.items():
        meta = heads.decoder_metadata[name]
        level = int(meta["level"])
        path = output_dir / f"decoder_level{level}.ckpt"
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
            path,
        )
        decoder_files[f"level{level}"] = path.name


    OmegaConf.save(cfg, output_dir / "config.yaml")
    OmegaConf.save(train_cfg, output_dir / "policy_train_config.yaml")
    OmegaConf.save(OmegaConf.create(metrics), output_dir / "metrics.yaml")
    OmegaConf.save(OmegaConf.create(eval_metrics), output_dir / EVAL_METRICS_FILENAME)
    OmegaConf.save(
        OmegaConf.create(eval_metrics_history),
        output_dir / EVAL_METRICS_HISTORY_FILENAME,
    )
    _write_probe_dim_metrics(output_dir, probe_dim_metrics)
    _write_probe_summary_metrics(output_dir, probe_summary_metrics)

    manifest = {
        "format": "final_probing_decoding_eval_v1",
        "policy": None if policy_path is None else str(policy_path),
        "normalizer": None if normalizer_path is None else str(normalizer_path),
        "metric_prefix": FINAL_PROBING_DECODING_EVAL_PREFIX,
        "metrics": "metrics.yaml",
        "eval_metrics": EVAL_METRICS_FILENAME,
        "eval_metrics_history": EVAL_METRICS_HISTORY_FILENAME,
        "probe_dim_metrics": PROBE_DIM_METRICS_FILENAME,
        "probe_summary_metrics": PROBE_SUMMARY_METRICS_FILENAME,
        "eval_datasets": list(eval_dataset_keys),
        "heads": heads_path.name,
        "decoder_files": decoder_files,
        "decoder_visualizations": decoder_visualization_files,
        "probers": heads.prober_metadata,
        "decoders": heads.decoder_metadata,
    }
    OmegaConf.save(OmegaConf.create(manifest), output_dir / MANIFEST_FILENAME)
    return output_dir / MANIFEST_FILENAME


def _init_wandb(cfg, output_dir: Path):
    wandb_cfg = cfg.get("wandb", {}) or {}
    if not bool(wandb_cfg.get("enabled", False)):
        return None, {}

    import wandb

    init_cfg = _plain_container(wandb_cfg.get("config", {}))
    # Prefer an explicit configured name/id;
    # fall back to the output dir name for standalone one-off runs.
    run_name = init_cfg.get("name") or output_dir.name
    init_cfg["name"] = run_name
    init_cfg["id"] = init_cfg.get("id") or run_name
    init_cfg.setdefault("resume", "allow")
    run = wandb.init(**init_cfg)
    return run, init_cfg


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

    train_dataset, eval_datasets = _build_datasets(cfg, normalizer_artifact, train_cfg)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        **_loader_cfg(cfg, train=True),
        generator=generator,
    )
    eval_loaders = {
        key: DataLoader(dataset, **_loader_cfg(cfg, train=False))
        for key, dataset in eval_datasets.items()
    }

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
    train_probe_denominators = _probe_metric_denominators(heads, train_dataset, device)
    eval_probe_denominators = {
        key: _probe_metric_denominators(heads, dataset, device)
        for key, dataset in eval_datasets.items()
    }
    eval_probe_variances = {
        key: _probe_metric_variances(heads, dataset, device)
        for key, dataset in eval_datasets.items()
    }

    wandb_run = None
    wandb_config = {}
    if logger is None:
        wandb_run, wandb_config = _init_wandb(cfg, output_dir)

    epochs = int(cfg.epochs)

    train_batch_limit = _limit_batches(cfg, "limit_train_batches")
    eval_batch_limit = _limit_batches(cfg, "limit_eval_batches")
    final_metrics = {}
    eval_metrics_history = []
    for epoch in range(1, epochs + 1):
        train_metrics = _train_one_epoch(
            heads,
            train_loader,
            optimizer,
            device,
            train_probe_denominators,
            limit_batches=train_batch_limit,
        )
        eval_metrics_by_key = {
            key: _evaluate(
                heads,
                loader,
                device,
                eval_probe_denominators[key],
                limit_batches=eval_batch_limit,
            )
            for key, loader in eval_loaders.items()
        }
        final_metrics = _format_epoch_metrics(
            train_metrics,
            eval_metrics_by_key,
            FINAL_PROBING_DECODING_EVAL_PREFIX,
        )
        eval_metrics_history.append(
            {
                "epoch": epoch,
                "metrics": _eval_only_metrics(
                    final_metrics,
                    FINAL_PROBING_DECODING_EVAL_PREFIX,
                ),
            }
        )
        step = None if log_step is None else int(log_step) + epoch
        _log_metrics(logger, wandb_run, final_metrics, step)

        eval_mean_loss = final_metrics.get(
            f"{FINAL_PROBING_DECODING_EVAL_PREFIX}/eval_mean/loss",
            math.nan,
        )
        logging.info(
            "final_probing_decoding_eval epoch "
            f"{epoch}/{epochs}: train_loss={train_metrics.get('loss'):.6f}, "
            f"eval_mean_loss={eval_mean_loss:.6f}"
        )

    manifest_path = None
    if bool(cfg.get("save_artifacts", True)):
        output_dir.mkdir(parents=True, exist_ok=True)
        final_eval_metrics = _eval_only_metrics(
            final_metrics,
            FINAL_PROBING_DECODING_EVAL_PREFIX,
        )
        probe_dim_metrics = []
        probe_summary_metrics = []
        for key, loader in eval_loaders.items():
            dim_rows, summary_rows = _evaluate_probe_dim_metrics(
                heads,
                loader,
                device,
                eval_probe_variances[key],
                eval_key=key,
                limit_batches=eval_batch_limit,
            )
            probe_dim_metrics.extend(dim_rows)
            probe_summary_metrics.extend(summary_rows)
        _log_metrics(
            logger,
            wandb_run,
            _probe_summary_log_metrics(
                probe_summary_metrics,
                FINAL_PROBING_DECODING_EVAL_PREFIX,
            ),
            step,
        )
        decoder_visualization_files = _save_decoder_visualizations(
            output_dir,
            heads,
            eval_datasets,
            device,
        )
        manifest_path = _save_artifacts(
            heads,
            output_dir,
            cfg,
            train_cfg,
            final_metrics,
            final_eval_metrics,
            eval_metrics_history,
            probe_dim_metrics,
            probe_summary_metrics,
            decoder_visualization_files,
            policy_path,
            normalizer_path,
            eval_datasets.keys(),
        )
    if wandb_run is not None:
        wandb_run.finish()

    if manifest_path is None:
        logging.info("Skipped final probing/decoding eval artifact saving.")
    else:
        logging.info(f"Saved final probing/decoding eval artifacts to {output_dir}")
    return {
        "output_dir": str(output_dir),
        "manifest": None if manifest_path is None else str(manifest_path),
        "metrics": final_metrics,
    }


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
