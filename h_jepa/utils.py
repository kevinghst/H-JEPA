import logging
import threading
import torch
from pathlib import Path
from lightning.pytorch.callbacks import Callback
from omegaconf import OmegaConf


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


def _batches_done(trainer) -> int:
    """Train batches processed so far in this run, restored from a resume checkpoint."""
    return int(trainer.fit_loop.epoch_loop.batch_progress.total.processed)


class ResumeCheckpoint(Callback):
    """Full training state (weights, optimizer, scheduler, loop counters) to `path`.

    Written atomically every `every_n` train batches and at every epoch end, so a preempted
    run resumes mid-epoch (`ResumableManager` in main_hjepa.py loads it). The dataloader is
    not fast-forwarded: the resumed epoch restarts its sample order.
    """

    def __init__(self, path: Path, every_n: int):
        super().__init__()
        self.path, self.every_n = Path(path), int(every_n)

    def _save(self, trainer) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        trainer.save_checkpoint(str(tmp))
        if trainer.is_global_zero:
            tmp.replace(self.path)
            logging.info(f"Resume checkpoint after train batch {_batches_done(trainer)} -> {self.path}")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if _batches_done(trainer) % self.every_n == 0:
            self._save(trainer)

    def on_train_epoch_end(self, trainer, pl_module):
        self._save(trainer)


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
        # Count from the restored step so the budget spans the whole run across resumes.
        self._train_batches_seen = _batches_done(trainer)
        logging.info(f"TrainBatchLimitCallback: {self._train_batches_seen} train batches already done")
        if self.max_train_batches_total is not None and self._train_batches_seen >= self.max_train_batches_total:
            trainer.should_stop = True

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
    """Run the planning eval at the end of training and, if `every_n_epochs` > 0, every
    `every_n_epochs` epochs; planner seed = model seed."""

    def __init__(
        self,
        *,
        enabled,
        eval_cfg,
        run_dir,
        seed,
        output_subdir="planning_eval",
        run_on_train_end=True,
        every_n_epochs=0,
    ):
        super().__init__()
        self.enabled = enabled
        self.eval_cfg = eval_cfg
        self.run_dir = Path(run_dir)
        self.seed = int(seed)
        self.output_subdir = output_subdir
        self.run_on_train_end = run_on_train_end
        self.every_n_epochs = int(every_n_epochs)
        self._last_eval_step = None

    @staticmethod
    def _set_planning_seed(cfg, seed):
        solver_cfg = cfg.solver
        if solver_cfg.get("solvers", None) is not None:
            for level_solver in solver_cfg.solvers.values():
                level_solver.seed = int(seed)
            return

        solver_cfg.seed = int(seed)

    def _log_metrics(self, trainer, metrics, step):
        logger = trainer.logger
        if logger is None:
            return

        prefix = f"planning_eval/{self.eval_cfg.config_name}"
        scalar_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, bool):
                scalar_metrics[f"{prefix}/{key}"] = float(value)
            elif isinstance(value, (int, float)):
                scalar_metrics[f"{prefix}/{key}"] = value

        if scalar_metrics:
            logger.log_metrics(scalar_metrics, step=step)

    def _run_eval(self, trainer, pl_module, epoch):
        from planning_eval import load_eval_config, run_planning_eval

        cfg = load_eval_config(config_name=self.eval_cfg.config_name)
        cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
        overrides = self.eval_cfg.get("overrides", None)
        if overrides is not None:
            cfg = OmegaConf.merge(cfg, overrides)
        self._set_planning_seed(cfg, self.seed)

        module_states = [
            (module, module.training) for module in pl_module.model.modules()
        ]
        try:
            pl_module.model.eval()
            metrics = run_planning_eval(
                cfg,
                model=pl_module.model,
                results_dir=self.run_dir / self.output_subdir / f"epoch_{epoch:04d}",
            )
        finally:
            for module, was_training in module_states:
                module.train(was_training)

        self._log_metrics(trainer, metrics, trainer.global_step)

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        if not self.enabled or not self.every_n_epochs or epoch % self.every_n_epochs:
            return

        trainer.strategy.barrier("planning_eval_epoch_start")
        if trainer.is_global_zero:
            self._run_eval(trainer, pl_module, epoch)
        trainer.strategy.barrier("planning_eval_epoch_end")
        self._last_eval_step = trainer.global_step

    def on_train_end(self, trainer, pl_module):
        if not self.enabled or not self.run_on_train_end or trainer.global_step <= 0:
            return
        if self._last_eval_step == trainer.global_step:
            return

        trainer.strategy.barrier("planning_eval_final_start")
        if trainer.is_global_zero:
            self._run_eval(trainer, pl_module, trainer.current_epoch + 1)
        trainer.strategy.barrier("planning_eval_final_end")
