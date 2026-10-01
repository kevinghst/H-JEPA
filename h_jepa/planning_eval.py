import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import stable_worldmodel as swm
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms

from data import (
    IMAGENET_STATS,
    NORMALIZER_ARTIFACT_FILENAME,
    load_normalizer_artifact,
)
from eval_config_utils import _build_policy_plan_config, _is_hierarchical_solver
from stable_worldmodel.solver.hierarchical_solver import build_hierarchical_solver
from utils import resolve_model_checkpoint_path


def img_transform(cfg):
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**IMAGENET_STATS),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )


def generation_provenance():
    repo = Path(__file__).resolve().parent
    git = lambda *a: subprocess.run(["git", *a], cwd=repo, capture_output=True, text=True).stdout.strip()
    return {
        "command": shlex.join(["python", *sys.argv]),
        "cwd": str(Path.cwd()),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty_files": git("status", "--porcelain").splitlines(),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    }


def load_eval_config(config_name: str | None = None, config_path: str | None = None):
    if config_path is not None:
        return OmegaConf.load(config_path)

    if config_name is None:
        raise ValueError("Either config_name or config_path must be provided.")

    eval_dir = Path(__file__).parent / "config" / "eval"
    return OmegaConf.load(eval_dir / f"{config_name}.yaml")


def _resolve_existing_path(path: str | Path) -> Path:
    raw_path = Path(path).expanduser()
    for candidate in (Path.cwd() / raw_path, Path(__file__).parent / raw_path):
        if candidate.exists():
            return candidate
    return raw_path




def get_episodes_length(dataset, episodes):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"

    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def get_dataset(cfg, dataset_name):
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    return swm.data.HDF5Dataset(
        dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        keys_to_merge=cfg.dataset.get("keys_to_merge", None),
        cache_dir=dataset_path,
        level1=cfg.dataset["level1"],
    )


def save_eval_config(results_dir: Path, cfg: DictConfig) -> Path:
    config_path = results_dir / "eval_config.yaml"
    config_path.write_text(OmegaConf.to_yaml(cfg, resolve=True))
    return config_path


def resolve_results_dir(
    cfg: DictConfig,
    results_dir: str | Path | None = None,
) -> Path:
    if results_dir is not None:
        return Path(results_dir)

    output_dir = cfg.output.get("dir", None)
    if cfg.policy in (None, "random"):
        return Path(__file__).parent if output_dir is None else Path(output_dir)
    return Path(swm.data.utils.get_cache_dir(), cfg.policy).parent / (output_dir or "")


def _required_process_columns(cfg: DictConfig) -> list[str]:
    columns = [*cfg.dataset.keys_to_cache, *(cfg.dataset.get("keys_to_merge", None) or {})]
    return [col for col in dict.fromkeys(columns) if col != "pixels"]


def _normalizer_artifact_to_process(artifact: dict, required_cols: list[str]) -> dict:
    stats = artifact["stats"]
    missing = [col for col in required_cols if col not in stats]
    if missing:
        available = ", ".join(sorted(stats.keys()))
        raise KeyError(
            "Training normalizer artifact is missing required columns "
            f"{missing}. Available columns: {available}"
        )

    process = {}
    for col in required_cols:
        col_stats = stats[col]
        mean = np.asarray(col_stats["mean"], dtype=np.float64).reshape(-1)
        scale = np.asarray(col_stats["std"], dtype=np.float64).reshape(-1)

        processor = preprocessing.StandardScaler()
        processor.mean_ = mean
        processor.scale_ = scale
        processor.var_ = np.square(scale)
        processor.n_features_in_ = int(mean.shape[0])
        processor.n_samples_seen_ = int(col_stats.get("count", 0))
        process[col] = processor

        if col != "action":
            process[f"goal_{col}"] = processor

    return process


def _load_policy_normalizer_artifact(cfg: DictConfig) -> dict:
    ckpt_path = resolve_model_checkpoint_path(cfg.policy, cfg.get("cache_dir"))
    normalizer_path = ckpt_path.parent / NORMALIZER_ARTIFACT_FILENAME
    if not normalizer_path.exists():
        raise FileNotFoundError(
            "Training normalizer artifact not found for policy checkpoint. "
            f"Expected: {normalizer_path}."
        )
    return load_normalizer_artifact(normalizer_path)


def _build_eval_process_from_policy_normalizer(
    cfg: DictConfig,
    model=None,
) -> dict:
    artifact = getattr(model, "normalizer_artifact", None) if model is not None else None
    if artifact is None:
        if cfg.get("policy", "random") == "random":
            return {}
        artifact = _load_policy_normalizer_artifact(cfg)

    return _normalizer_artifact_to_process(artifact, _required_process_columns(cfg))


def _drop_levels_above_model(cfg, num_levels):
    """Let one planning config (e.g. the lN_project configs) serve models of any depth."""
    for key in list(cfg.solver.solvers):
        if int(key[len("level"):]) > num_levels:
            del cfg.solver.solvers[key]
            del cfg.hierarchical_plan_config[key]


def load_model(cfg: DictConfig, model=None):
    """Load cfg.policy (or pick the cost module of an in-memory flat model) for planning on cuda."""
    if model is None:
        if _is_hierarchical_solver(cfg):
            ckpt_path = resolve_model_checkpoint_path(cfg.policy, cfg.get("cache_dir"))
            model = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        else:
            model = swm.policy.AutoCostModel(cfg.policy)

        model = model.to("cuda")
    elif not _is_hierarchical_solver(cfg):
        selected_model = next((m for m in model.modules() if hasattr(m, "get_cost")), None)
        if selected_model is None:
            raise RuntimeError(
                "No module with 'get_cost' found in the provided in-memory model."
            )
        model = selected_model

    model = model.eval()
    model.interpolate_pos_encoding = True
    return model


def build_solver(cfg: DictConfig, model):
    if _is_hierarchical_solver(cfg):
        return build_hierarchical_solver(cfg, model)
    return hydra.utils.instantiate(cfg.solver, model=model)


def _build_policy(
    cfg: DictConfig,
    process: dict,
    transform: dict,
    model=None,
):
    if model is None and cfg.get("policy", "random") == "random":
        return swm.policy.RandomPolicy()

    model = load_model(cfg, model)
    if _is_hierarchical_solver(cfg):
        _drop_levels_above_model(cfg, model.num_levels)
    policy_config = _build_policy_plan_config(cfg)
    solver = build_solver(cfg, model)

    return swm.policy.WorldModelPolicy(
        solver=solver,
        config=policy_config,
        process=process,
        transform=transform,
        eval_budget=int(cfg.eval.eval_budget),
    )


def _sample_stratified_eval_starts(cfg: DictConfig, dataset):
    """Balance eval starts across trajectories, then space them out within each.

    Trajectories get near-equal quotas (capped by how many valid starts each has),
    and within a trajectory the valid-start range is split into contiguous bins with
    one start drawn per bin. Removes the length bias of uniform step sampling and
    avoids near-duplicate segments from the same trajectory.
    """
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices = np.unique(dataset.get_col_data(col_name))
    episode_len = get_episodes_length(dataset, ep_indices)

    # valid starts per episode: step_idx in [0, length - offset - 1]
    allowed = [np.arange(max(int(n) - int(cfg.eval.goal_offset_steps), 0)) for n in episode_len]

    eligible = np.array([len(a) > 0 for a in allowed])
    ep_indices = ep_indices[eligible]
    allowed = [a for a, keep in zip(allowed, eligible) if keep]
    capacity = np.array([len(a) for a in allowed], dtype=np.int64)
    print(int(capacity.sum()), "valid starting points found for evaluation.")

    num_eval = int(cfg.eval.num_eval)
    if capacity.sum() < num_eval:
        raise ValueError(
            f"Not enough valid starts for evaluation: requested {num_eval}, "
            f"found {int(capacity.sum())}."
        )

    g = np.random.default_rng(cfg.seed)

    # capacity-constrained round-robin: hand out one slot at a time in a shuffled
    # episode order until num_eval slots are placed, skipping saturated episodes.
    order = g.permutation(len(ep_indices))
    quotas = np.zeros(len(ep_indices), dtype=np.int64)
    remaining = num_eval
    while remaining > 0:
        for i in order:
            if remaining == 0:
                break
            if quotas[i] < capacity[i]:
                quotas[i] += 1
                remaining -= 1

    eval_episodes = []
    eval_start_idx = []
    for i in range(len(ep_indices)):
        k = int(quotas[i])
        if k == 0:
            continue
        v = int(capacity[i])
        edges = np.linspace(0, v, k + 1).astype(np.int64)
        for j in range(k):
            low, high = edges[j], edges[j + 1]
            eval_episodes.append(int(ep_indices[i]))
            eval_start_idx.append(int(allowed[i][g.integers(low, high)]))

    return np.array(eval_episodes), np.array(eval_start_idx)


def _serialize_metrics(metrics: dict):
    metrics_to_save = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            metrics_to_save[key] = value.tolist()
        elif isinstance(value, np.generic):
            metrics_to_save[key] = value.item()
        elif torch.is_tensor(value):
            metrics_to_save[key] = value.detach().cpu().tolist()
        else:
            metrics_to_save[key] = value
    return metrics_to_save


def pop_solve_records(solver) -> list:
    solvers = getattr(solver, "level_solvers", {} if solver is None else {1: solver})
    records = [r for level in sorted(solvers) for r in solvers[level].solve_records]
    for s in solvers.values():
        s.solve_records = []
    return records


def _run_chunked_eval(
    cfg: DictConfig,
    chunk_size: int,
    world_cfg: DictConfig,
    image_shape: tuple,
    load_eval_trajs_path: str,
    process: dict,
    transform: dict,
    model,
    results_dir: Path,
) -> dict:
    """Evaluate the first eval.num_eval loaded tasks chunk_size at a time (one World of chunk_size
    envs per chunk), so that a preempted eval only redoes the chunk it was in.

    Each chunk writes results_dir/chunks/tasks_<start>-<end>.json (per-task results + GD solve
    records) and is skipped when that file exists. Every chunk reseeds the planner generators with
    base seed + its first task index: its result does not depend on which chunks ran before it in
    the process, and a single chunk (chunk_size >= num_eval) plans with the unchunked eval's seed.
    metrics.yaml aggregates the chunks with the keys of the unchunked eval.
    """
    import json

    payload = torch.load(load_eval_trajs_path, map_location="cpu", weights_only=False)
    n = int(cfg.eval.num_eval)
    if len(payload["data"]) < n:
        raise ValueError(f"{load_eval_trajs_path}: {len(payload['data'])} tasks < eval.num_eval={n}")
    ep_idx = payload.get("episodes_idx")
    ep_idx = list(ep_idx) if ep_idx is not None and len(ep_idx) else list(range(len(payload["data"])))
    callables = OmegaConf.to_container(cfg.eval.get("callables"), resolve=True)
    chunk_dir = results_dir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_paths = {
        s: chunk_dir / f"tasks_{s:03d}-{min(s + chunk_size, n):03d}.json" for s in range(0, n, chunk_size)
    }

    policy, gens = None, []
    for start, path in chunk_paths.items():
        if path.exists():
            continue
        end = min(start + chunk_size, n)
        if policy is None:
            policy = _build_policy(cfg, process, transform, model=model)
            solver = getattr(policy, "solver", None)
            gens = [
                (s.torch_gen, s.torch_gen.initial_seed())
                for s in getattr(solver, "level_solvers", {1: solver}).values()
                if hasattr(s, "torch_gen")
            ]
        for gen, seed in gens:
            gen.manual_seed(seed + start)
        # shallow per-episode copies: evaluate_from_dataset reassigns their columns in place
        chunk = {**payload, "data": [dict(ep) for ep in payload["data"][start:end]], "episodes_idx": ep_idx[start:end]}
        world = swm.World(**{**world_cfg, "num_envs": end - start}, image_shape=image_shape)
        try:
            world.set_policy(policy)
            t0 = time.time()
            with torch.enable_grad(), sdpa_kernel(SDPBackend.MATH):
                metrics = world.evaluate_from_dataset(
                    None,
                    episodes_idx=None,
                    start_steps=None,
                    goal_offset_steps=cfg.eval.goal_offset_steps,
                    eval_budget=cfg.eval.eval_budget,
                    callables=callables,
                    load_eval_trajs_path=chunk,
                    process=process,
                    start_state_mode=cfg.eval.get("start_state_mode", "dataset_full"),
                    goal_state_mode=cfg.eval.get("goal_state_mode", "dataset_full"),
                )
            metrics["evaluation_time"] = time.time() - t0
        finally:
            world.close()
        metrics["solve_records"] = pop_solve_records(getattr(policy, "solver", None))
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(_serialize_metrics(metrics)))
        os.replace(tmp, path)  # atomic: a preemption never leaves a partial chunk file
        print(f"tasks {start}-{end}: success_rate {metrics['success_rate']:.1f}")

    chunks = [json.loads(p.read_text()) for p in chunk_paths.values()]
    successes = np.concatenate([np.asarray(c["episode_successes"], dtype=bool) for c in chunks])
    steps = np.concatenate([np.asarray(c["steps_to_success"], dtype=np.float64) for c in chunks])
    seeds = None if chunks[0]["seeds"] is None else [s for c in chunks for s in c["seeds"]]
    if seeds is not None:
        assert np.unique(np.asarray(seeds)).shape[0] == n, "Some episode seeds are identical!"
    metrics_to_save = {
        "success_rate": float(successes.sum()) / n * 100.0,
        "episode_successes": successes.tolist(),
        "seeds": seeds,
        "wall_clock_to_success": [x for c in chunks for x in c["wall_clock_to_success"]],
        "steps_to_success": steps.tolist(),
        "loaded_eval_original_lengths": [x for c in chunks for x in c["loaded_eval_original_lengths"]],
        "steps_to_success_success_only": (
            float(np.mean(steps[np.isfinite(steps)])) if np.isfinite(steps).any() else float("nan")
        ),
        "evaluation_time": sum(c["evaluation_time"] for c in chunks),
        "chunk_size": chunk_size,
    }
    (results_dir / "metrics.yaml").write_text(OmegaConf.to_yaml(metrics_to_save))
    print(metrics_to_save)
    solve_records = [r for c in chunks for r in c["solve_records"]]
    if solve_records:
        (results_dir / "planning_compute.json").write_text(json.dumps(solve_records))
    return metrics_to_save


def run_planning_eval(
    cfg: DictConfig,
    model=None,
    results_dir: str | Path | None = None,
):
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    resolved_results_dir = resolve_results_dir(cfg, results_dir)
    resolved_results_dir.mkdir(parents=True, exist_ok=True)
    save_eval_config(resolved_results_dir, cfg)

    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world_cfg = OmegaConf.create(OmegaConf.to_container(cfg.world, resolve=True))

    transform = {
        "pixels": img_transform(cfg),
        "goal": img_transform(cfg),
    }

    load_eval_trajs_path = cfg.get("load_eval_trajs_path", None)
    if load_eval_trajs_path:
        load_eval_trajs_path = str(_resolve_existing_path(load_eval_trajs_path))
    dump_eval_trajs_path = cfg.get("dump_eval_trajs_path", None)
    dump_eval_only = dump_eval_trajs_path is not None

    image_shape = (int(cfg.eval.img_size),) * 2

    dataset = None
    eval_episodes = None
    eval_start_idx = None
    process = {}

    if dump_eval_only:
        dataset = get_dataset(cfg, cfg.eval.dataset_name)
        eval_episodes, eval_start_idx = _sample_stratified_eval_starts(cfg, dataset)
    else:
        process = _build_eval_process_from_policy_normalizer(cfg, model=model)

    chunk_size = cfg.eval.get("chunk_size", None)  # +eval.chunk_size=N: resumable per chunk of N tasks
    if chunk_size:
        if dump_eval_only or not load_eval_trajs_path:
            raise ValueError("eval.chunk_size needs load_eval_trajs_path and no dump_eval_trajs_path")
        return _run_chunked_eval(
            cfg, int(chunk_size), world_cfg, image_shape, load_eval_trajs_path,
            process, transform, model, resolved_results_dir,
        )

    world = swm.World(**world_cfg, image_shape=image_shape)

    try:
        policy = _build_policy(cfg, process, transform, model=model)
        world.set_policy(policy)

        start_time = time.time()
        with torch.enable_grad(), sdpa_kernel(SDPBackend.MATH):
            callables = OmegaConf.to_container(
                cfg.eval.get("callables"), resolve=True
            )
            start_steps = (
                eval_start_idx.tolist() if eval_start_idx is not None else None
            )
            episodes_idx = (
                eval_episodes.tolist() if eval_episodes is not None else None
            )
            metrics = world.evaluate_from_dataset(
                dataset,
                start_steps=start_steps,
                goal_offset_steps=cfg.eval.goal_offset_steps,
                eval_budget=cfg.eval.eval_budget,
                episodes_idx=episodes_idx,
                callables=callables,
                dump_eval_trajs_path=dump_eval_trajs_path,
                load_eval_trajs_path=load_eval_trajs_path,
                process=process,
                start_state_mode=cfg.eval.get(
                    "start_state_mode", "dataset_full"
                ),
                goal_state_mode=cfg.eval.get(
                    "goal_state_mode", "dataset_full"
                ),
            )
        end_time = time.time()
    finally:
        world.close()

    if dump_eval_only:
        payload = torch.load(dump_eval_trajs_path, weights_only=False)
        payload["task_info"] = {
            "dataset_name": str(cfg.eval.dataset_name),
            "sampling_mode": "stratified",
            "seed": int(cfg.seed),
            "num_episodes": int(cfg.eval.num_eval),
            "eval_config": OmegaConf.to_container(cfg, resolve=False),
            **generation_provenance(),
        }
        torch.save(payload, dump_eval_trajs_path)
        print(metrics)
        return metrics

    metrics_to_save = _serialize_metrics(metrics)
    metrics_to_save["evaluation_time"] = end_time - start_time

    metrics_path = resolved_results_dir / "metrics.yaml"
    metrics_path.write_text(OmegaConf.to_yaml(metrics_to_save))
    print(metrics)

    # Per-solve GD compute metadata (level, horizon, num_samples, per-env n_iters,
    # wall_clock) for the planner-FLOPs Pareto analysis.
    solve_records = pop_solve_records(getattr(policy, "solver", None))
    if solve_records:
        import json
        (resolved_results_dir / "planning_compute.json").write_text(
            json.dumps(solve_records)
        )

    return metrics_to_save
