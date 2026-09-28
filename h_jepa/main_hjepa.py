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
from omegaconf import OmegaConf
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
    save_normalizer_artifact,
)

# uuid
import uuid
from datetime import datetime


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_hjepa_optimizer_factory(model, cfg):
    optimizer_cfg = OmegaConf.to_container(cfg.optimizer, resolve=True)

    def optimizer_factory(params):
        param_groups = []
        for level in range(1, int(cfg.num_levels) + 1):
            level_params = [
                param for param in model.get_level(level).parameters() if param.requires_grad
            ]
            param_groups.append(
                {
                    "params": level_params,
                    "lr": cfg[f"level{level}"].lr,
                    "name": f"level{level}",
                }
            )

        rows = [
            (group["name"], group["lr"], len(group["params"]))
            for group in param_groups
        ]
        logging.info(f"HJEPA optimizer parameter groups: {rows}")
        return spt.optim.create_optimizer(param_groups, optimizer_cfg)

    return optimizer_factory


def _configure_runtime_performance():
    torch.set_float32_matmul_precision("high")

    if not torch.cuda.is_available():
        return

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


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

    cache_dir = os.environ.get("STABLEWM_HOME", None)

    dataset_cfg = {k: v for k, v in cfg.data.dataset.items() if k != "val_name"}
    val_total_transitions = dataset_cfg.pop("val_total_transitions", None)
    val_name = cfg.data.dataset.get("val_name", None)
    train_dataset = build_hdf5_dataset(dataset_cfg, cache_dir=cache_dir)

    val_dataset_cfg = {
        k: v for k, v in dataset_cfg.items()
        if k not in ("sources", "total_transitions")
    }
    val_dataset_cfg["name"] = val_name
    if val_total_transitions is not None:
        val_dataset_cfg["total_transitions"] = int(val_total_transitions)
    val_dataset = build_hdf5_dataset(val_dataset_cfg, cache_dir=cache_dir)

    # Stored images already match img_size; uint8 pixels are normalized on device.
    extra_transforms = [
        get_column_normalizer(train_dataset, col, col)
        for col in cfg.data.dataset.keys_to_load
        if not col.startswith("pixels")
    ]
    transform = spt.data.transforms.Compose(*extra_transforms)
    train_dataset.transform = transform
    val_dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train = DataLoader(
        train_dataset,
        **cfg.loader,
        generator=rnd_gen,
    )
    val_cfg = OmegaConf.to_container(cfg.loader, resolve=True)
    val_cfg["shuffle"] = True
    val_cfg["drop_last"] = False
    val = DataLoader(val_dataset, **val_cfg)

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
        eval_cfg=_pe,
        run_dir=run_dir,
        seed=cfg.seed,
        output_subdir=_pe.output_subdir,
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

    probes = []
    for level in range(1, int(cfg.num_levels) + 1):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels") or col in ["action"]:
                continue

            output_dim = train_dataset.get_dim(col)
            for name, probe_input in (
                (f"level{level}_probe_{col}", f"embed_{level}"),
                (f"level{level}_pred_probe_{col}", f"pred_embed_{level}"),
            ):
                probes.append(
                    spt.callbacks.OnlineProbe(
                        world_model,
                        target=_probe_target_key(col, level),
                        input=probe_input,
                        name=name,
                        probe=build_prober(
                            {},
                            input_dim=embed_dims[level],
                            output_dim=output_dim,
                        ),
                        loss=nn.MSELoss(),
                        metrics=torchmetrics.regression.MeanSquaredError(),
                    )
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
