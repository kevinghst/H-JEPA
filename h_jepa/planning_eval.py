import shlex
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from omegaconf import DictConfig, ListConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms

from data import (
    NORMALIZER_ARTIFACT_FILENAME,
    load_normalizer_artifact,
)
from eval_config_utils import _build_policy_plan_config, _is_hierarchical_solver
from stable_worldmodel.solver.hierarchical_solver import HierarchicalSolver, _build_hierarchical_solver
from utils import (
    register_legacy_checkpoint_module_aliases,
    resolve_model_checkpoint_path,
)


def img_transform(cfg):
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
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
    if raw_path.is_absolute() and raw_path.exists():
        return raw_path

    candidates = [
        Path.cwd() / raw_path,
        Path(__file__).parent / raw_path,
        Path(__file__).parent.parent / raw_path,
    ]
    for candidate in candidates:
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
    if cfg.policy not in (None, "random"):
        base_dir = Path(swm.data.utils.get_cache_dir(), cfg.policy).parent
        if output_dir is None:
            return base_dir
        output_path = Path(output_dir)
        return output_path if output_path.is_absolute() else base_dir / output_path

    if output_dir is not None:
        return Path(output_dir)

    return Path(__file__).parent


def _build_process(cfg: DictConfig, dataset):
    process = {}
    process_cols = list(cfg.dataset.keys_to_cache)
    keys_to_merge = cfg.dataset.get("keys_to_merge", None)
    if keys_to_merge is not None:
        for target_col in keys_to_merge.keys():
            if target_col not in process_cols:
                process_cols.append(target_col)

    for col in process_cols:
        if col == "pixels":
            continue
        processor = preprocessing.StandardScaler()
        col_data = dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor

        if col != "action":
            process[f"goal_{col}"] = process[col]

    return process


def _required_process_columns(cfg: DictConfig) -> list[str]:
    columns = []
    for col in cfg.dataset.keys_to_cache:
        if col == "pixels":
            continue
        if col not in columns:
            columns.append(col)

    keys_to_merge = cfg.dataset.get("keys_to_merge", None)
    if keys_to_merge is not None:
        for col in keys_to_merge.keys():
            if col == "pixels":
                continue
            if col not in columns:
                columns.append(col)

    return columns


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
    ckpt_path = resolve_model_checkpoint_path(cfg.policy, cfg.cache_dir)
    normalizer_path = ckpt_path.parent / NORMALIZER_ARTIFACT_FILENAME
    if not normalizer_path.exists():
        raise FileNotFoundError(
            "Training normalizer artifact not found for policy checkpoint. "
            f"Expected: {normalizer_path}. "
            "Generate it with scripts/regenerate_training_normalizer.py."
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


def _find_model_with_attribute(model, attribute_name: str):
    """Mirror AutoCostModel behavior for in-memory models."""
    if hasattr(model, attribute_name):
        if isinstance(model, torch.nn.Module):
            model = model.eval()
        return model

    for child in model.children():
        result = _find_model_with_attribute(child, attribute_name)
        if result is not None:
            return result

    return None


def _drop_levels_above_model(cfg, num_levels):
    """Let one planning config (e.g. the lN_project configs) serve models of any depth."""
    for key in list(cfg.solver.solvers):
        if int(key[len("level"):]) > num_levels:
            del cfg.solver.solvers[key]
            del cfg.hierarchical_plan_config[key]


def _build_policy(
    cfg: DictConfig,
    process: dict,
    transform: dict,
    model=None,
):
    if model is None and cfg.get("policy", "random") == "random":
        return swm.policy.RandomPolicy()

    if model is None:
        if _is_hierarchical_solver(cfg):
            ckpt_path = resolve_model_checkpoint_path(cfg.policy, cfg.cache_dir)
            register_legacy_checkpoint_module_aliases()
            model = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        else:
            model = swm.policy.AutoCostModel(cfg.policy)

        model = model.to("cuda")
    elif not _is_hierarchical_solver(cfg):
        selected_model = _find_model_with_attribute(model, "get_cost")
        if selected_model is None:
            raise RuntimeError(
                "No module with 'get_cost' found in the provided in-memory model."
            )
        model = selected_model

    model = model.eval()
    model.interpolate_pos_encoding = True

    if _is_hierarchical_solver(cfg):
        _drop_levels_above_model(cfg, len(HierarchicalSolver._extract_level_models(model)))
    policy_config = _build_policy_plan_config(cfg)

    if _is_hierarchical_solver(cfg):
        solver = _build_hierarchical_solver(cfg, model)
    else:
        solver = hydra.utils.instantiate(cfg.solver, model=model)

    return swm.policy.WorldModelPolicy(
        solver=solver,
        config=policy_config,
        process=process,
        transform=transform,
        eval_budget=int(cfg.eval.eval_budget),
    )


def _solver_requires_grad(cfg: DictConfig) -> bool:
    """Return True when the configured planner needs autograd during eval."""
    solver_cfg = cfg.get("solver")
    if solver_cfg is None:
        return False

    def _has_grad_solver(node) -> bool:
        if node is None:
            return False

        if isinstance(node, DictConfig):
            target = str(node.get("_target_", ""))
            if target.endswith("GradientSolver"):
                return True

            for key in node.keys():
                if OmegaConf.is_missing(node, key):
                    continue
                if _has_grad_solver(node[key]):
                    return True
            return False

        if isinstance(node, (list, tuple, ListConfig)):
            return any(_has_grad_solver(value) for value in node)

        return False

    return _has_grad_solver(solver_cfg)


def _sample_eval_starts(cfg: DictConfig, dataset):
    sampling_mode = str(cfg.eval.get("traj_sampling_mode", "random"))
    if sampling_mode == "cube_pickup_centered":
        return _sample_cube_pickup_eval_starts(cfg, dataset, center_on="grasp")
    if sampling_mode == "cube_pickup_centered_lift":
        return _sample_cube_pickup_eval_starts(cfg, dataset, center_on="lift")
    if sampling_mode == "stratified":
        return _sample_stratified_eval_starts(cfg, dataset)
    if sampling_mode != "random":
        raise ValueError(f"Unsupported eval trajectory sampling mode: {sampling_mode}")

    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(dataset.get_col_data(col_name), return_index=True)

    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {
        ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)
    }
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )

    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(valid_mask.sum(), "valid starting points found for evaluation.")

    g = np.random.default_rng(cfg.seed)
    random_episode_indices = g.choice(
        len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False
    )
    random_episode_indices = np.sort(valid_indices[random_episode_indices])

    eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
    eval_start_idx = dataset.get_row_data(random_episode_indices)["step_idx"]

    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError("Not enough episodes with sufficient length for evaluation.")

    return eval_episodes, eval_start_idx


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


def _sample_cube_pickup_eval_starts(cfg: DictConfig, dataset, center_on="grasp"):
    required_cols = ["privileged_block_0_pos"]
    if center_on == "grasp":
        required_cols.append("proprio_gripper_contact")
    for required in required_cols:
        if required not in dataset.column_names:
            raise ValueError(
                f"cube_pickup_centered sampling requires '{required}'."
            )

    goal_offset_steps = int(cfg.eval.goal_offset_steps)
    midpoint = goal_offset_steps // 2
    height_delta = float(cfg.eval.get("pickup_height_delta", 0.02))
    contact_threshold = float(cfg.eval.get("pickup_contact_threshold", 0.5))

    block_pos = dataset.get_col_data("privileged_block_0_pos")
    if block_pos.ndim != 2 or block_pos.shape[1] < 3:
        raise ValueError(
            "'privileged_block_0_pos' must have shape [num_steps, >=3]."
        )
    if center_on == "grasp":
        contact_col = dataset.get_col_data("proprio_gripper_contact")
        contact_col = contact_col[:, 0] if contact_col.ndim == 2 else contact_col

    candidate_episodes = []
    candidate_starts = []
    for episode_idx, (offset, length) in enumerate(zip(dataset.offsets, dataset.lengths)):
        if length < goal_offset_steps:
            continue

        z_pos = block_pos[offset:offset + length, 2]
        lifted_steps = np.flatnonzero(z_pos >= z_pos[0] + height_delta)
        if lifted_steps.size == 0:
            continue
        lift_step = int(lifted_steps[0])

        if center_on == "lift":
            center_step = lift_step
        else:
            # Center on the grasp, not the lift: walk back from the lift over the
            # contiguous gripper-contact run to the step contact began. The lift
            # only fires once the cube is airborne (~4-5 steps after the grasp).
            contact = contact_col[offset:offset + length] >= contact_threshold
            if contact[lift_step]:
                grasp_end = lift_step
            else:
                prior_contact = np.flatnonzero(contact[:lift_step + 1])
                if prior_contact.size == 0:
                    continue
                grasp_end = int(prior_contact[-1])
            grasp_step = grasp_end
            while grasp_step - 1 >= 0 and contact[grasp_step - 1]:
                grasp_step -= 1
            center_step = grasp_step

        start_step = center_step - midpoint
        end_step = start_step + goal_offset_steps
        if start_step < 0 or end_step > length:
            continue

        candidate_episodes.append(episode_idx)
        candidate_starts.append(start_step)

    num_candidates = len(candidate_episodes)
    print(
        num_candidates,
        "pickup-centered cube starting points found for evaluation.",
    )
    if num_candidates < cfg.eval.num_eval:
        raise ValueError(
            "Not enough pickup-centered cube segments for evaluation: "
            f"requested {cfg.eval.num_eval}, found {num_candidates}."
        )

    g = np.random.default_rng(cfg.seed)
    sampled_indices = np.sort(
        g.choice(num_candidates, size=cfg.eval.num_eval, replace=False)
    )

    eval_episodes = np.asarray(candidate_episodes, dtype=np.int64)[sampled_indices]
    eval_start_idx = np.asarray(candidate_starts, dtype=np.int64)[sampled_indices]
    return eval_episodes, eval_start_idx


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


def _collect_solve_records(solver) -> list:
    """Gather per-solve GD compute records from a (possibly hierarchical) solver.

    Each GradientSolver appends a record per solve() call; a HierarchicalSolver
    holds one GradientSolver per level, each recording tagged with its level.
    """
    if solver is None:
        return []
    if hasattr(solver, "level_solvers"):
        records = []
        for level in sorted(solver.level_solvers):
            records.extend(getattr(solver.level_solvers[level], "solve_records", []))
        return records
    return list(getattr(solver, "solve_records", []))


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
    if "image_shape" in world_cfg:
        del world_cfg["image_shape"]

    transform = {
        "pixels": img_transform(cfg),
        "goal": img_transform(cfg),
    }

    load_eval_trajs_path = cfg.get("load_eval_trajs_path", None)
    if load_eval_trajs_path:
        load_eval_trajs_path = str(_resolve_existing_path(load_eval_trajs_path))
    dump_eval_trajs_path = cfg.get("dump_eval_trajs_path", None)
    dump_eval_only = dump_eval_trajs_path is not None

    img_size = cfg.eval.img_size
    if isinstance(img_size, (list, tuple, ListConfig)):
        image_shape = tuple(int(size) for size in img_size)
    else:
        image_shape = (int(img_size), int(img_size))

    dataset = None
    eval_episodes = None
    eval_start_idx = None
    process = {}

    if dump_eval_only:
        dataset = get_dataset(cfg, cfg.eval.dataset_name)
        eval_episodes, eval_start_idx = _sample_eval_starts(cfg, dataset)
    elif load_eval_trajs_path:
        process = _build_eval_process_from_policy_normalizer(cfg, model=model)
    else:
        dataset = get_dataset(cfg, cfg.eval.dataset_name)
        process = _build_eval_process_from_policy_normalizer(cfg, model=model)
        eval_episodes, eval_start_idx = _sample_eval_starts(cfg, dataset)

    world = swm.World(**world_cfg, image_shape=image_shape)

    try:
        policy = _build_policy(cfg, process, transform, model=model)
        world.set_policy(policy)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        start_time = time.time()
        solver_requires_grad = _solver_requires_grad(cfg)
        grad_context = torch.enable_grad if solver_requires_grad else torch.no_grad
        sdp_context = nullcontext()
        if solver_requires_grad and torch.cuda.is_available():
            sdp_context = sdpa_kernel(SDPBackend.MATH)

        with grad_context(), sdp_context:
            eval_start_index = int(cfg.eval.get("start_index", 0))
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
                eval_start_index=eval_start_index,
                process=process,
                start_state_mode=cfg.eval.get(
                    "start_state_mode", "dataset_full"
                ),
                goal_state_mode=cfg.eval.get(
                    "goal_state_mode", "dataset_full"
                ),
                expert_action_distance_horizon=cfg.eval.get(
                    "expert_action_distance_horizon", None
                ),
                expert_action_distance_dims=cfg.eval.get(
                    "expert_action_distance_dims", None
                ),
            )
        end_time = time.time()
    finally:
        world.close()

    if dump_eval_only:
        payload = torch.load(dump_eval_trajs_path, weights_only=False)
        payload["task_info"] = {
            "dataset_name": str(cfg.eval.dataset_name),
            "sampling_mode": str(cfg.eval.get("traj_sampling_mode", "random")),
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
    if torch.cuda.is_available():
        metrics_to_save["peak_gpu_mem_bytes"] = int(torch.cuda.max_memory_allocated())

    metrics_path = resolved_results_dir / "metrics.yaml"
    metrics_path.write_text(OmegaConf.to_yaml(metrics_to_save))
    print(metrics)

    # Per-solve GD compute metadata (level, horizon, num_samples, per-env n_iters,
    # wall_clock) for the planner-FLOPs Pareto analysis.
    solve_records = _collect_solve_records(getattr(policy, "solver", None))
    if solve_records:
        import json
        (resolved_results_dir / "planning_compute.json").write_text(
            json.dumps(solve_records)
        )

    return metrics_to_save
