from functools import partial
import os
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torchmetrics
from loguru import logger as logging
from omegaconf import OmegaConf, open_dict
from torch import nn
from torch.utils.data import DataLoader

from hjepa_utils import (
    create_world_model,
    hjepa_forward,
    probe_target_key as _probe_target_key,
)
from models.probers import build_prober
from utils import (
    ModelObjectCallBack,
    DebugArtifactCleanupCallback,
    PlanningEvalCallback,
    TrainBatchLimitCallback,
)
from final_probing_decoding_eval import FinalProbingDecodingEvalCallback
from data import (
    build_normalizer_artifact,
    build_hdf5_dataset,
    get_column_normalizer,
    get_hdf5_image_shape,
    get_img_preprocessor,
    image_shape_matches_size,
    save_normalizer_artifact,
)

# uuid
import uuid
from datetime import datetime


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _get_level_lr(level_cfg, default_lr):
    level_lr = level_cfg.get("lr", None)
    if level_lr is None:
        return default_lr
    return level_lr


def _optimizer_base_config(optimizer_cfg):
    return {
        key: value
        for key, value in optimizer_cfg.items()
        if key != "prober"
    }


def _optimizer_group_config(optimizer_cfg, group_name):
    group_cfg = optimizer_cfg.get(group_name, None)
    if group_cfg is None:
        return None
    base_cfg = _optimizer_base_config(optimizer_cfg)
    base_cfg.update(group_cfg)
    return base_cfg


def _probe_input_stream(level_cfg, col):
    probes_cfg = level_cfg.get("probes", {}) or {}
    inputs = probes_cfg.get("inputs", None)
    if inputs is None:
        return "embed"
    return inputs.get(col, None)


def _probes_enabled(level_cfg):
    probes_cfg = level_cfg.get("probes", {}) or {}
    return bool(probes_cfg.get("enabled", True))


def _probe_architecture(level_cfg, col):
    probes_cfg = level_cfg.get("probes", {}) or {}
    architectures = probes_cfg.get("architectures", {}) or {}
    return architectures.get(col, {}) or {}


def _probe_input_key(level_cfg, col, level, *, pred=False):
    stream = _probe_input_stream(level_cfg, col)
    if stream is None:
        return None
    stream = str(stream)
    valid_streams = {"embed", "pixel_embed", "proprio_embed"}
    if stream not in valid_streams:
        raise ValueError(
            f"level{level}.probes.inputs.{col} must be one of "
            f"{sorted(valid_streams)}, got '{stream}'."
        )

    prefix = "pred_" if pred else ""
    return f"{prefix}{stream}_{level}"


def _probe_source_model(world_model):
    return getattr(world_model, "model", world_model)


def _pred_stream_probe_available(world_model, level, input_key):
    if input_key == f"pred_embed_{level}":
        return True
    if input_key not in {
        f"pred_pixel_embed_{level}",
        f"pred_proprio_embed_{level}",
    }:
        return False

    source_model = _probe_source_model(world_model)
    encoder = getattr(source_model.get_level(level), "encoder", None)
    return (
        hasattr(encoder, "pixel_encoder")
        and hasattr(encoder, "proprio_encoder")
        and isinstance(getattr(encoder, "projector", None), nn.Identity)
    )


def _probe_input_dim(world_model, embed_dims, level, input_key):
    if input_key.startswith("pred_"):
        if not _pred_stream_probe_available(world_model, level, input_key):
            raise ValueError(
                f"Probe input '{input_key}' cannot be produced. Stream-specific "
                "prediction probes require a proprio fusion encoder with "
                "encoder.projector.type=identity."
            )
        input_key = input_key[len("pred_"):]

    if input_key == f"embed_{level}":
        return embed_dims[level]
    source_model = _probe_source_model(world_model)
    if input_key == f"pixel_embed_{level}":
        dims = getattr(source_model, "pixel_embed_dims", {})
        if level in dims:
            return dims[level]
    if input_key == f"proprio_embed_{level}":
        dims = getattr(source_model, "proprio_embed_dims", {})
        if level in dims:
            return dims[level]

    raise ValueError(f"Could not infer probe input dimension for '{input_key}'.")

def _build_hjepa_optimizer_factory(model, cfg):
    optimizer_cfg = OmegaConf.to_container(cfg.optimizer, resolve=True)
    base_optimizer_cfg = _optimizer_base_config(optimizer_cfg)
    default_lr = base_optimizer_cfg.get("lr", None)
    if default_lr is None:
        raise ValueError(
            "cfg.optimizer.lr must be set to use per-level JEPA learning rates."
        )
    def optimizer_factory(params):
        param_groups = []
        assigned_param_ids = set()

        for level in range(1, int(cfg.num_levels) + 1):
            level_cfg = cfg[f"level{level}"]
            level_lr = _get_level_lr(level_cfg, default_lr)
            jepa = model.get_level(level)
            level_params = [param for param in jepa.parameters() if param.requires_grad]

            if level_params:
                param_groups.append(
                    {
                        "params": level_params,
                        "lr": level_lr,
                        "name": f"level{level}",
                    }
                )
                assigned_param_ids.update(id(param) for param in level_params)

        global_params = [
            param
            for param in params
            if param.requires_grad and id(param) not in assigned_param_ids
        ]
        if global_params:
            param_groups.append(
                {
                    "params": global_params,
                    "lr": default_lr,
                    "name": "global",
                }
            )

        if not param_groups:
            raise ValueError("No trainable HJEPA parameters found for optimizer.")

        rows = [
            (group["name"], group["lr"], len(group["params"]))
            for group in param_groups
        ]
        logging.info(f"HJEPA optimizer parameter groups: {rows}")
        return spt.optim.create_optimizer(param_groups, base_optimizer_cfg)

    return optimizer_factory


def _configure_runtime_performance():
    torch.set_float32_matmul_precision("high")

    if not torch.cuda.is_available():
        return

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


def _can_skip_cpu_image_preprocess(train_dataset, val_dataset, col, img_size):
    train_shape = get_hdf5_image_shape(train_dataset, col)
    if not image_shape_matches_size(train_shape, img_size):
        return False

    if val_dataset is None:
        return True

    val_shape = get_hdf5_image_shape(val_dataset, col)
    return train_shape == val_shape and image_shape_matches_size(val_shape, img_size)


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


def _hjepa_training_forward(self, batch, stage, cfg):
    return hjepa_forward(
        self,
        batch,
        stage,
        cfg,
        normalize_batch=_normalize_uint8_images_on_device,
    )


@hydra.main(version_base=None, config_path="config/train", config_name=None)
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    _configure_runtime_performance()

    cache_dir = None
    if not hasattr(cfg, "local_cache_dir"):
        cache_dir = os.environ.get("SLURM_TMPDIR", None)

    cache_dir = os.environ.get("STABLEWM_HOME", cache_dir)

    dataset_cfg = {k: v for k, v in cfg.data.dataset.items() if k != "val_name"}
    val_total_transitions = dataset_cfg.pop("val_total_transitions", None)
    val_name = cfg.data.dataset.get("val_name", None)
    train_dataset = build_hdf5_dataset(dataset_cfg, transform=None, cache_dir=cache_dir)

    val_dataset = None
    if isinstance(val_name, str) and val_name.strip():
        val_dataset_cfg = {
            k: v for k, v in dataset_cfg.items()
            if k not in ("sources", "total_transitions")
        }
        val_dataset_cfg["name"] = val_name
        if val_total_transitions is not None:
            val_dataset_cfg["total_transitions"] = int(val_total_transitions)
        val_dataset = build_hdf5_dataset(
            val_dataset_cfg, transform=None, cache_dir=cache_dir
        )

    extra_transforms = []
    for col in cfg.data.dataset.keys_to_load:
        if col.startswith("pixels"):
            if _can_skip_cpu_image_preprocess(
                train_dataset,
                val_dataset,
                col,
                cfg.img_size,
            ):
                logging.info(
                    f"Skipping CPU image preprocessing for '{col}' because the "
                    f"stored image shape already matches img_size={cfg.img_size}; "
                    "uint8 images will be normalized on device."
                )
                continue
            processor = get_img_preprocessor(col, col, cfg.img_size)
            extra_transforms.append(processor)
            continue
        extra_transforms.append(get_column_normalizer(train_dataset, col, col))

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            for level in range(1, int(cfg.num_levels) + 1):
                level_wm = cfg[f"level{level}"].wm
                dim_key = f"{col}_dim"
                if level_wm.get(dim_key, None) is None:
                    setattr(level_wm, dim_key, train_dataset.get_dim(col))

    transform = spt.data.transforms.Compose(*extra_transforms)
    train_dataset.transform = transform
    if val_dataset is not None:
        val_dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    if val_dataset is None:
        if cfg.train_split is None:
            raise ValueError(
                "cfg.train_split must be set when cfg.data.dataset.val_name is null."
            )
        train_set, val_set = spt.data.random_split(
            train_dataset,
            lengths=[cfg.train_split, 1 - cfg.train_split],
            generator=rnd_gen,
        )
    else:
        train_set = train_dataset
        val_set = val_dataset


    train = DataLoader(
        train_set,
        **cfg.loader,
        generator=rnd_gen,
    )
    val_cfg = OmegaConf.to_container(cfg.loader, resolve=True)
    val_cfg["shuffle"] = True
    val_cfg["drop_last"] = False
    val = DataLoader(val_set, **val_cfg)

    ##############################
    ##       model / optim      ##
    ##############################

    normalizer_artifact = build_normalizer_artifact(cfg, train_dataset)
    world_model, losses, embed_dims = create_world_model(cfg)
    world_model.normalizer_artifact = normalizer_artifact
    models = {
        "model": world_model,
    }

    optimizers = {}
    hjepa_optimizer = _build_hjepa_optimizer_factory(world_model, cfg)
    for model_name in models.keys():
        optimizers[f"{model_name}_opt"] = {
            "modules": str(model_name),
            "optimizer": hjepa_optimizer,
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        **models,
        **losses,
        forward=partial(_hjepa_training_forward, cfg=cfg),
        optim=optimizers,
    )
    world_model._train_cfg = cfg
    world_model._output_model_name = cfg.output_model_name

    ##########################
    ##       training       ##
    ##########################

    # run_id = cfg.get("subdir") or ""

    # use rand_str to ensure unique run_dir for each run if subdir is not specified in cfg.
    rand_str = f"{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"
    run_id = cfg.get("subdir") or rand_str

    ckpt_root = Path(os.getenv("STABLEWM_HOME", str(swm.data.utils.get_cache_dir()))) / "ckpts"
    run_dir = ckpt_root / run_id
    logging.info(f"🫆🫆🫆 Run ID: {run_id} 🫆🫆🫆")

    logger = None
    if cfg.wandb.enabled and not cfg.get("quick_debug", False):
        from lightning.pytorch.loggers import WandbLogger
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))
    else:
        # Without an explicit logger, Lightning defaults the CSVLogger to the cwd,
        # so concurrent jobs sharing one snapshot dir all write the same
        # lightning_logs/version_0/metrics.csv and corrupt each other's header
        # rewrites. Root it at the per-run run_dir instead.
        from lightning.pytorch.loggers import CSVLogger
        logger = CSVLogger(save_dir=str(run_dir), name="lightning_logs")

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)
    normalizer_path = save_normalizer_artifact(normalizer_artifact, run_dir)
    logging.info(f"Saved training normalizer artifact to {normalizer_path}")

    lr_callback = pl.pytorch.callbacks.LearningRateMonitor(logging_interval="step")
    object_dump_callback = ModelObjectCallBack(
        dirpath=run_dir,
        filename=cfg.output_model_name,
        epoch_interval=int(cfg.save_every_n_epochs),
    )
    debug_cleanup_callback = DebugArtifactCleanupCallback(
        enabled=bool(cfg.get("quick_debug", False)),
        run_dir=run_dir,
    )
    _pe = cfg.planning_eval
    planning_eval_callback = PlanningEvalCallback(
        enabled=bool(_pe.enabled),
        every_n_epochs=_pe.every_n_epochs,
        eval_cfg=_pe,
        run_dir=run_dir,
        output_subdir=_pe.output_subdir,
        run_on_train_start=bool(_pe.get("run_on_train_start", False)),
        run_on_train_end=bool(_pe.run_on_train_end),
    )
    train_batch_limit_callback = TrainBatchLimitCallback(
        cfg.get("max_train_batches_total", None)
    )
    final_probing_decoding_eval_cfg = cfg.get(
        "final_probing_decoding_eval",
        OmegaConf.create({"enabled": False, "config_name": None, "config_path": None}),
    )
    final_probing_decoding_eval_callback = FinalProbingDecodingEvalCallback(
        eval_cfg=final_probing_decoding_eval_cfg,
        run_dir=run_dir,
        output_subdir=final_probing_decoding_eval_cfg.get(
            "output_subdir",
            "final_probing_decoding_eval",
        ),
    )

    optimizer_cfg = OmegaConf.to_container(cfg.optimizer, resolve=True)
    prober_optimizer_cfg = _optimizer_group_config(optimizer_cfg, "prober")
    probes = []
    for level in range(1, int(cfg.num_levels) + 1):
        level_cfg = cfg[f"level{level}"]
        if not _probes_enabled(level_cfg):
            logging.info(f"Skipping in-training probes for level{level}: probes.enabled=false.")
            continue
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels") or col in ["action"]:
                continue

            probe_input = _probe_input_key(level_cfg, col, level)
            if probe_input is None:
                continue

            probe_cfg = _probe_architecture(level_cfg, col)
            output_dim = train_dataset.get_dim(col)
            probe_input_dim = _probe_input_dim(
                world_model,
                embed_dims,
                level,
                probe_input,
            )
            probe = spt.callbacks.OnlineProbe(
                world_model,
                target=_probe_target_key(col, level),
                input=probe_input,
                name=f"level{level}_probe_{col}",
                probe=build_prober(
                    probe_cfg,
                    input_dim=probe_input_dim,
                    output_dim=output_dim,
                ),
                loss=nn.MSELoss(),
                optimizer=prober_optimizer_cfg,
                metrics=torchmetrics.regression.MeanSquaredError(),
            )
            probes.append(probe)

            pred_probe_input = _probe_input_key(level_cfg, col, level, pred=True)
            pred_probe_input_dim = _probe_input_dim(
                world_model,
                embed_dims,
                level,
                pred_probe_input,
            )
            pred_probe = spt.callbacks.OnlineProbe(
                world_model,
                target=_probe_target_key(col, level),
                input=pred_probe_input,
                name=f"level{level}_pred_probe_{col}",
                probe=build_prober(
                    probe_cfg,
                    input_dim=pred_probe_input_dim,
                    output_dim=output_dim,
                ),
                loss=nn.MSELoss(),
                optimizer=prober_optimizer_cfg,
                metrics=torchmetrics.regression.MeanSquaredError(),
            )
            probes.append(pred_probe)

            print(
                f"target: {col}_level{level}, "
                f"input: {probe_input} ({probe_input_dim}), "
                f"pred_input: {pred_probe_input} ({pred_probe_input_dim}), "
                f"output_dim: {output_dim}"
            )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[
        debug_cleanup_callback,
        train_batch_limit_callback,
        object_dump_callback,
        planning_eval_callback,
        final_probing_decoding_eval_callback,
        lr_callback,
        *probes,
        ],
        logger=logger,
        enable_checkpointing=False,
    )

    manager_ckpt_path = None
    if not cfg.get("quick_debug", False):
        manager_ckpt_path = run_dir / f"{cfg.output_model_name}_weights.ckpt"

    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        seed=cfg.seed,
        ckpt_path=manager_ckpt_path,
    )

    manager()
    return


if __name__ == "__main__":
    run()
