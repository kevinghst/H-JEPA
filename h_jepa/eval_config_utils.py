from omegaconf import DictConfig

import stable_worldmodel as swm


def _is_hierarchical_solver(cfg: DictConfig) -> bool:
    """Return True when eval config selects HierarchicalSolver."""
    target = cfg.solver.get('_target_', '')
    return str(target).endswith('HierarchicalSolver')


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
    receding_horizon = int(hp_cfg.receding_horizon)

    def _build_level(level_key: str) -> swm.PlanConfig:
        if level_key not in hp_cfg or hp_cfg[level_key] is None:
            raise ValueError(f"hierarchical_plan_config must define '{level_key}'.")

        level_cfg = hp_cfg[level_key]
        action_dim = level_cfg.get('action_dim', None)
        return swm.PlanConfig(
            horizon=int(level_cfg.horizon),
            receding_horizon=receding_horizon,
            action_block=int(level_cfg.get('action_block', 1)),
            action_dim=(None if action_dim is None else int(action_dim)),
            pre_decay_horizon=_build_optional_positive_int(
                level_cfg,
                'pre_decay_horizon',
            ),
            num_subgoals=int(level_cfg.get('num_subgoals', 1)),
            cost_last_n=int(level_cfg.get('cost_last_n', 1)),
            intermediate_cost_weight=float(level_cfg.get('intermediate_cost_weight', 1.0)),
            action_cost_weight=float(level_cfg.get('action_cost_weight', 0.0)),
            horizon_one_goal_cost_space=str(
                level_cfg.get('horizon_one_goal_cost_space', 'lower')
            ),
        )

    num_levels = sum(1 for key in hp_cfg if str(key).startswith('level'))
    return swm.HierarchicalPlanConfig(
        receding_horizon=receding_horizon,
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
    return swm.PlanConfig(
        horizon=int(plan_cfg.horizon),
        receding_horizon=int(plan_cfg.receding_horizon),
        action_block=int(plan_cfg.get('action_block', 1)),
        action_dim=(
            None
            if plan_cfg.get('action_dim', None) is None
            else int(plan_cfg.action_dim)
        ),
        pre_decay_horizon=_build_optional_positive_int(
            plan_cfg,
            'pre_decay_horizon',
        ),
        cost_last_n=int(plan_cfg.get('cost_last_n', 1)),
    )
