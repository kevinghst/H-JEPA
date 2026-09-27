import importlib
import logging
import math
import sys
import threading
import torch
from pathlib import Path
from lightning.pytorch.callbacks import Callback
from omegaconf import OmegaConf


def register_legacy_checkpoint_module_aliases() -> None:
    """Map legacy module paths to the reorganized model package for torch.load."""
    alias_pairs = {
        "jepa": "models.jepa",
        "hjepa": "models.hjepa",
        "module": "models.module",
        "seq_encoder": "models.encoders.seq_encoder",
        "spt_backbone_utils": "models.encoders.vit",
        "unit_tests.spt_backbone_utils": "models.encoders.vit",
        "factories.build_encoder": "models.encoders.build_encoder",
    }

    for legacy_name, current_name in alias_pairs.items():
        if legacy_name in sys.modules:
            continue
        sys.modules[legacy_name] = importlib.import_module(current_name)


def resolve_model_checkpoint_path(run_name: str, cache_dir: str | None):
    """Resolve a run name/path to an on-disk *_object.ckpt path."""
    run_path = Path(run_name).expanduser()
    if run_path.suffix == '.ckpt':
        if run_path.exists():
            return run_path

        if not run_path.is_absolute():
            import stable_worldmodel as swm

            cached_path = Path(cache_dir or swm.data.utils.get_cache_dir(), run_name)
            if cached_path.exists():
                return cached_path

        raise FileNotFoundError(
            f'Checkpoint path does not exist: {run_path}. Launch pretraining first.'
        )

    if not run_path.exists():
        import stable_worldmodel as swm

        run_path = Path(cache_dir or swm.data.utils.get_cache_dir(), run_name)

    if run_path.is_dir():
        ckpt_files = list(run_path.glob('*_object.ckpt'))
        ckpt_files.sort(key=lambda x: x.stat().st_ctime, reverse=True)
        if not ckpt_files:
            raise FileNotFoundError(
                f'No *_object.ckpt found in directory: {run_path}'
            )
        return ckpt_files[0]

    path = Path(f'{run_path}_object.ckpt')
    if not path.exists():
        raise FileNotFoundError(
            f'Checkpoint path does not exist: {path}. Launch pretraining first.'
        )
    return path


def resolve_checkpoint_config_path(
    ckpt_path: str | Path,
    config_path: str | Path | None = None,
) -> Path:
    """Resolve the training config associated with a model checkpoint."""
    path = (
        Path(config_path).expanduser()
        if config_path is not None
        else Path(ckpt_path).expanduser().parent / "config.yaml"
    )
    if not path.exists():
        raise FileNotFoundError(f"Training config not found: {path}")
    return path


def load_training_config_for_checkpoint(
    ckpt_path: str | Path,
    config_path: str | Path | None = None,
):
    return OmegaConf.load(resolve_checkpoint_config_path(ckpt_path, config_path))


def print_parameter_counts(
    module: torch.nn.Module,
    name: str | None = None,
) -> tuple[int, int]:
    """Print and return trainable and total parameter counts for a module."""
    total_params = sum(param.numel() for param in module.parameters())
    trainable_params = sum(
        param.numel() for param in module.parameters() if param.requires_grad
    )

    module_name = name or module.__class__.__name__
    print(
        f"{module_name} parameters: "
        f"trainable={trainable_params:,}, total={total_params:,}"
    )
    return trainable_params, total_params


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


class DebugArtifactCleanupCallback(Callback):
    """Delete environment dump artifacts as soon as they appear."""

    def __init__(self, enabled, run_dir, poll_interval=0.1):
        super().__init__()
        self.enabled = enabled
        self.run_dir = Path(run_dir)
        self.poll_interval = poll_interval
        self._stop_event = threading.Event()
        self._thread = None

    def _cleanup_once(self):
        if not self.enabled:
            return

        candidate_dirs = {
            Path.cwd(),
            Path(__file__).resolve().parent,
            self.run_dir,
        }

        patterns = (
            "environment.json",
            "environment_v*.json",
            "requirements_frozen.txt",
            "requirements_frozen_v*.txt",
        )

        for directory in candidate_dirs:
            if not directory.exists():
                continue
            for pattern in patterns:
                for artifact_path in directory.glob(pattern):
                    if artifact_path.exists():
                        artifact_path.unlink()
                        logging.info(f"Removed debug artifact: {artifact_path}")

    def _watch_loop(self):
        while not self._stop_event.is_set():
            self._cleanup_once()
            self._stop_event.wait(self.poll_interval)

    def setup(self, trainer, pl_module, stage):
        if not self.enabled or stage != "fit":
            return
        self._cleanup_once()
        self._thread = threading.Thread(
            target=self._watch_loop,
            name="debug-artifact-cleaner",
            daemon=True,
        )
        self._thread.start()

    def teardown(self, trainer, pl_module, stage):
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=2)
        self._cleanup_once()

    def on_exception(self, trainer, pl_module, exception):
        self._cleanup_once()


class ModelObjectCallBack(Callback):
    """Save the model object periodically and when training finishes."""

    def __init__(self, dirpath, filename='model_object', epoch_interval=1):
        super().__init__()
        self.dirpath, self.filename, self.epoch_interval = (
            Path(dirpath),
            filename,
            epoch_interval,
        )
        self._saved_final = False

    def _save_final(self, trainer, pl_module):
        if self._saved_final or not trainer.is_global_zero:
            return
        path = self.dirpath / f'{self.filename}_object.ckpt'
        torch.save(pl_module.model, path)
        self._saved_final = True
        logging.info(f'Saved final world model to {path}')

    def on_train_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        epoch = trainer.current_epoch + 1
        if epoch % self.epoch_interval == 0:
            path = self.dirpath / f'{self.filename}_epoch_{epoch}_object.ckpt'
            torch.save(pl_module.model, path)
            logging.info(f'Saved world model to {path}')
        if epoch == trainer.max_epochs:
            self._save_final(trainer, pl_module)

    def on_train_end(self, trainer, pl_module):
        self._save_final(trainer, pl_module)


class TrainBatchLimitCallback(Callback):
    """Stop training after a fixed number of train dataloader batches."""

    def __init__(self, max_train_batches_total):
        super().__init__()
        if max_train_batches_total is None:
            self.max_train_batches_total = None
        else:
            self.max_train_batches_total = int(max_train_batches_total)
            if self.max_train_batches_total <= 0:
                raise ValueError("max_train_batches_total must be positive.")
        self._train_batches_seen = 0

    def on_train_start(self, trainer, pl_module):
        self._train_batches_seen = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.max_train_batches_total is None:
            return

        self._train_batches_seen += 1
        if self._train_batches_seen < self.max_train_batches_total:
            return

        trainer.should_stop = True
        logging.info(
            "Stopping training after "
            f"{self._train_batches_seen} train batches "
            f"(max_train_batches_total={self.max_train_batches_total})."
        )


class PlanningEvalCallback(Callback):
    def __init__(
        self,
        *,
        enabled,
        every_n_epochs,
        eval_cfg,
        run_dir,
        output_subdir="planning_eval",
        run_on_train_start=False,
        run_on_train_end=True,
    ):
        super().__init__()
        self.enabled = enabled
        self.every_n_epochs = int(every_n_epochs)
        self.eval_cfg = eval_cfg
        self.run_dir = Path(run_dir)
        self.output_subdir = output_subdir
        self.run_on_train_start = run_on_train_start
        self.run_on_train_end = run_on_train_end
        self._last_eval_epoch = None

    def _should_run(self, epoch):
        if not self.enabled or self.every_n_epochs <= 0:
            return False
        if epoch <= 0 or epoch % self.every_n_epochs != 0:
            return False
        return self._last_eval_epoch != epoch

    def _planning_eval_seeds(self):
        seeds = self.eval_cfg.get("seeds", [42])
        if seeds is None:
            return [42]
        if isinstance(seeds, (int, float)):
            return [int(seeds)]
        return [int(seed) for seed in seeds] or [42]

    @staticmethod
    def _standard_error(values):
        if len(values) <= 1:
            return 0.0
        mean = sum(values) / len(values)
        var = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        return math.sqrt(var) / math.sqrt(len(values))

    @staticmethod
    def _is_scalar_number(value):
        if isinstance(value, bool):
            return True
        if isinstance(value, (int, float)):
            return math.isfinite(float(value))
        if hasattr(value, "item") and not getattr(value, "shape", ()):
            try:
                return math.isfinite(float(value.item()))
            except (TypeError, ValueError):
                return False
        return False

    @staticmethod
    def _as_float(value):
        if isinstance(value, bool):
            return float(value)
        if hasattr(value, "item") and not getattr(value, "shape", ()):
            return float(value.item())
        return float(value)

    @staticmethod
    def _finite_flat_values(value):
        if value is None:
            return []
        if hasattr(value, "detach"):
            value = value.detach().cpu().tolist()
        elif hasattr(value, "tolist"):
            value = value.tolist()

        stack = [value]
        values = []
        while stack:
            item = stack.pop()
            if isinstance(item, (list, tuple)):
                stack.extend(item)
                continue
            try:
                scalar = float(item)
            except (TypeError, ValueError):
                continue
            if math.isfinite(scalar):
                values.append(scalar)
        return values

    @staticmethod
    def _set_planning_seed(cfg, seed):
        if cfg.get("solver", None) is None:
            return

        solver_cfg = cfg.solver
        if solver_cfg.get("solvers", None) is not None:
            for level_solver in solver_cfg.solvers.values():
                level_solver.seed = int(seed)
            return

        solver_cfg.seed = int(seed)

    def _apply_eval_overrides(self, cfg):
        overrides = self.eval_cfg.get("overrides", None)
        if overrides is None:
            return cfg
        return OmegaConf.merge(cfg, overrides)

    def _aggregate_seed_metrics(self, metrics_by_seed):
        aggregate = {
            "planning_eval_num_seeds": len(metrics_by_seed),
            "planning_eval_seeds": [int(seed) for seed, _ in metrics_by_seed],
        }

        keys = sorted({key for _, metrics in metrics_by_seed for key in metrics})
        for key in keys:
            if key == "steps_to_success_success_only":
                continue
            values = []
            for _, metrics in metrics_by_seed:
                value = metrics.get(key)
                if self._is_scalar_number(value):
                    values.append(self._as_float(value))

            if not values:
                continue

            mean = sum(values) / len(values)
            aggregate[key] = mean
            aggregate[f"{key}_se"] = self._standard_error(values)

        steps_to_success = []
        saw_steps_to_success = False
        for _, metrics in metrics_by_seed:
            if "steps_to_success" in metrics:
                saw_steps_to_success = True
                steps_to_success.extend(
                    self._finite_flat_values(metrics.get("steps_to_success"))
                )
            elif "steps_to_success_success_only" in metrics:
                saw_steps_to_success = True
                steps_to_success.extend(
                    self._finite_flat_values(
                        metrics.get("steps_to_success_success_only")
                    )
                )
        if steps_to_success:
            aggregate["steps_to_success_success_only"] = (
                sum(steps_to_success) / len(steps_to_success)
            )
            aggregate["steps_to_success_success_only_se"] = self._standard_error(
                steps_to_success
            )
        elif saw_steps_to_success:
            aggregate["steps_to_success_success_only"] = float("nan")
            aggregate["steps_to_success_success_only_se"] = float("nan")

        return aggregate

    @staticmethod
    def _write_metrics(path, metrics):
        path.write_text(OmegaConf.to_yaml(OmegaConf.create(metrics)))

    def _log_metrics(self, trainer, metrics, step, *, prefix_suffix=None):
        logger = trainer.logger
        if logger is None:
            return

        eval_name = self.eval_cfg.get("config_name", None)
        if eval_name is None and self.eval_cfg.get("config_path", None) is not None:
            eval_name = Path(self.eval_cfg.config_path).stem
        prefix = f"planning_eval/{eval_name or 'eval'}"
        if prefix_suffix is not None:
            prefix = f"{prefix}/{prefix_suffix}"
        scalar_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, bool):
                scalar_metrics[f"{prefix}/{key}"] = float(value)
            elif isinstance(value, (int, float)):
                scalar_metrics[f"{prefix}/{key}"] = value
            elif hasattr(value, "item") and not getattr(value, "shape", ()):
                scalar_metrics[f"{prefix}/{key}"] = value.item()

        if scalar_metrics:
            logger.log_metrics(scalar_metrics, step=step)

    def _run_eval(
        self,
        trainer,
        pl_module,
        epoch,
        *,
        planning_seed=None,
        results_dir=None,
        log_metrics=True,
    ):
        from planning_eval import load_eval_config, run_planning_eval

        cfg = load_eval_config(
            config_name=self.eval_cfg.get("config_name", None),
            config_path=self.eval_cfg.get("config_path", None),
        )
        cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
        cfg = self._apply_eval_overrides(cfg)
        if planning_seed is not None:
            self._set_planning_seed(cfg, planning_seed)
        if results_dir is None:
            results_dir = self.run_dir / self.output_subdir / f"epoch_{epoch:04d}"

        module_states = [
            (module, module.training) for module in pl_module.model.modules()
        ]
        try:
            pl_module.model.eval()
            metrics = run_planning_eval(
                cfg,
                model=pl_module.model,
                results_dir=results_dir,
            )
        finally:
            for module, was_training in module_states:
                module.train(was_training)

        if log_metrics:
            self._log_metrics(trainer, metrics, trainer.global_step)
        self._last_eval_epoch = epoch
        return metrics

    def _run_final_eval(self, trainer, pl_module, epoch):
        seeds = self._planning_eval_seeds()
        metrics_by_seed = []
        results_root = self.run_dir / self.output_subdir / f"epoch_{epoch:04d}"
        for seed in seeds:
            if len(seeds) == 1:
                seed_results_dir = results_root
            else:
                seed_results_dir = results_root / f"seed{seed}"
            metrics = self._run_eval(
                trainer,
                pl_module,
                epoch,
                planning_seed=seed,
                results_dir=seed_results_dir,
                log_metrics=False,
            )
            metrics_by_seed.append((seed, metrics))
            self._log_metrics(
                trainer,
                metrics,
                trainer.global_step,
                prefix_suffix=f"seed_{seed}",
            )

        aggregate = self._aggregate_seed_metrics(metrics_by_seed)
        results_root.mkdir(parents=True, exist_ok=True)
        self._write_metrics(results_root / "metrics.yaml", aggregate)
        self._log_metrics(trainer, aggregate, trainer.global_step)
        self._last_eval_epoch = epoch

    def _maybe_run(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        if not self._should_run(epoch):
            return

        trainer.strategy.barrier("planning_eval_start")
        if trainer.is_global_zero:
            self._run_eval(trainer, pl_module, epoch)
        trainer.strategy.barrier("planning_eval_end")

    def on_train_epoch_end(self, trainer, pl_module):
        self._maybe_run(trainer, pl_module)

    def on_train_start(self, trainer, pl_module):
        if not self.enabled or not self.run_on_train_start or self._last_eval_epoch == 0:
            return

        trainer.strategy.barrier("planning_eval_initial_start")
        if trainer.is_global_zero:
            self._run_eval(trainer, pl_module, epoch=0)
        trainer.strategy.barrier("planning_eval_initial_end")

    def on_train_end(self, trainer, pl_module):
        if not self.enabled or not self.run_on_train_end:
            return
        final_epoch = trainer.current_epoch + 1
        if trainer.global_step <= 0 or self._last_eval_epoch == final_epoch:
            return

        trainer.strategy.barrier("planning_eval_final_start")
        if trainer.is_global_zero:
            self._run_final_eval(trainer, pl_module, final_epoch)
        trainer.strategy.barrier("planning_eval_final_end")
