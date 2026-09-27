"""Monotonicity metrics for a goal-cost curve along an embedded trajectory.

Each function scores one 1-D cost array for how cleanly it descends to the goal
(a greedy planner wants a curve that only ever goes down). All are scale-invariant
or computed on the max-normalized curve, so they match the plotted curves. Higher
= smoother for monotonicity; lower = smoother for backtracking.
"""

import numpy as np


def _spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else 0.0


def monotonicity_score(cost):
    """-Spearman(cost, t): +1 = perfectly decreasing to the goal, 0 = no trend."""
    return -_spearman(cost, np.arange(len(cost)))


def total_backtracking(cost):
    """Sum of positive steps on the max-normalized curve: total uphill a greedy
    descent must climb. 0 = monotone."""
    c = cost / (cost.max() + 1e-8)
    return float(np.clip(np.diff(c), 0, None).sum())


# Emitted into planning-eval metrics as `gt_cost_{name}_level{i}`.
GOAL_COST_METRICS = {
    "gt_cost_monotonicity": monotonicity_score,
    "gt_cost_backtracking": total_backtracking,
}
