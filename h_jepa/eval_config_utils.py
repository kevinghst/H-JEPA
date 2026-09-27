from omegaconf import DictConfig

import stable_worldmodel as swm


def _is_hierarchical_solver(cfg: DictConfig) -> bool:
    """Return True when eval config selects HierarchicalSolver."""
    target = cfg.solver.get('_target_', '')
    return str(target).endswith('HierarchicalSolver')


def _build_horizon_value(horizon_cfg, *, resolve_horizon: bool) -> int | list[int]:
    """Convert horizon config into either an int or a list of ints."""
    try:
        return int(horizon_cfg)
    except (TypeError, ValueError):
        pass

    if resolve_horizon:
        raise ValueError(
            'Expected integer horizon when resolve_horizon is enabled, '
            f'got {type(horizon_cfg).__name__}.'
        )

    return [int(horizon) for horizon in horizon_cfg]


def _build_cost_last_n(cost_cfg) -> int | list[int]:
    """Convert cost_last_n config into either an int or a list of ints."""
    try:
        return int(cost_cfg)
    except (TypeError, ValueError):
        return [int(x) for x in cost_cfg]


def _build_optional_positive_int(plan_cfg, key: str) -> int | None:
    value = plan_cfg.get(key, None)
    if value is None:
        return None

    value = int(value)
    if value <= 0:
        raise ValueError(f'{key} must be positive when set, got {value}.')
    return value


def _build_hierarchical_plan_config(cfg: DictConfig) -> swm.HierarchicalPlanConfig:
    """Build HierarchicalPlanConfig from hierarchical_plan_config section."""
    if 'hierarchical_plan_config' not in cfg:
        raise ValueError(
            "Missing 'hierarchical_plan_config' for hierarchical evaluation config."
        )

    hp_cfg = cfg.hierarchical_plan_config
    shared_receding = int(hp_cfg.receding_horizon)
    shared_history = int(hp_cfg.get('history_len', 1))
    shared_resolve_horizon = bool(hp_cfg.get('resolve_horizon', False))

    def _build_level(level_key: str) -> swm.PlanConfig:
        if level_key not in hp_cfg or hp_cfg[level_key] is None:
            raise ValueError(f"hierarchical_plan_config must define '{level_key}'.")

        level_cfg = hp_cfg[level_key]
        if 'resolve_horizon' in level_cfg:
            raise ValueError(
                f"hierarchical_plan_config.{level_key}.resolve_horizon is no "
                "longer supported. Set hierarchical_plan_config.resolve_horizon "
                "instead."
            )
        action_dim = level_cfg.get('action_dim', None)
        return swm.PlanConfig(
            horizon=_build_horizon_value(
                level_cfg.horizon,
                resolve_horizon=shared_resolve_horizon,
            ),
            receding_horizon=int(level_cfg.get('receding_horizon', shared_receding)),
            history_len=int(level_cfg.get('history_len', shared_history)),
            action_block=int(level_cfg.get('action_block', 1)),
            action_dim=(None if action_dim is None else int(action_dim)),
            pre_decay_horizon=_build_optional_positive_int(
                level_cfg,
                'pre_decay_horizon',
            ),
            num_subgoals=int(level_cfg.get('num_subgoals', 1)),
            cost_last_n=_build_cost_last_n(level_cfg.get('cost_last_n', 1)),
            intermediate_cost_weight=float(level_cfg.get('intermediate_cost_weight', 1.0)),
            action_cost_weight=float(level_cfg.get('action_cost_weight', 0.0)),
            action_cost_space=str(level_cfg.get('action_cost_space', 'pooled')),
            horizon_one_goal_cost_space=str(
                level_cfg.get('horizon_one_goal_cost_space', 'lower')
            ),
        )

    num_levels = sum(1 for key in hp_cfg if str(key).startswith('level'))
    return swm.HierarchicalPlanConfig(
        receding_horizon=shared_receding,
        history_len=shared_history,
        resolve_horizon=shared_resolve_horizon,
        levels=tuple(_build_level(f'level{level}') for level in range(1, num_levels + 1)),
    )


def _build_policy_plan_config(
    cfg: DictConfig,
) -> swm.PlanConfig | swm.HierarchicalPlanConfig:
    """Build policy plan config from eval yaml structure."""
    if _is_hierarchical_solver(cfg) and 'hierarchical_plan_config' in cfg:
        return _build_hierarchical_plan_config(cfg)

    if 'plan_config' not in cfg:
        raise ValueError("Missing 'plan_config' for non-hierarchical evaluation config.")

    plan_cfg = cfg.plan_config
    resolve_horizon = bool(plan_cfg.get('resolve_horizon', False))
    return swm.PlanConfig(
        horizon=_build_horizon_value(
            plan_cfg.horizon,
            resolve_horizon=resolve_horizon,
        ),
        receding_horizon=int(plan_cfg.receding_horizon),
        history_len=int(plan_cfg.get('history_len', 1)),
        action_block=int(plan_cfg.get('action_block', 1)),
        action_dim=(
            None
            if plan_cfg.get('action_dim', None) is None
            else int(plan_cfg.action_dim)
        ),
        resolve_horizon=resolve_horizon,
        pre_decay_horizon=_build_optional_positive_int(
            plan_cfg,
            'pre_decay_horizon',
        ),
        cost_last_n=_build_cost_last_n(plan_cfg.get('cost_last_n', 1)),
    )


def _execution_plan_config(
    config: swm.PlanConfig | swm.HierarchicalPlanConfig,
) -> swm.PlanConfig:
    """Return the low-level PlanConfig used to execute environment actions."""
    if isinstance(config, swm.HierarchicalPlanConfig):
        if config.level1 is None:
            raise ValueError('HierarchicalPlanConfig.level1 must be provided.')
        return config.level1
    return config
