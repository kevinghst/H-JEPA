"""CPU regression tests for best-iterate retention in gradient planning."""

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from stable_worldmodel.solver.gd import GradientSolver


class QuadraticCostModel(torch.nn.Module):
    level = 1

    def __init__(self):
        super().__init__()
        self.evaluations = []
        self.objectives = []

    def get_cost(self, info, actions):
        window = min(info['cost_last_n'], actions.shape[-2])
        weight = info['intermediate_cost_weight']
        self.objectives.append((window, weight))
        errors = (actions - info['target'].unsqueeze(-1)).square()
        weights = actions.new_ones(window)
        weights[:-1] = weight
        costs = (
            errors[..., -window:, :] * weights[None, None, :, None]
        ).sum(dim=(-2, -1))
        self.evaluations.append((
            info['target'][:, 0, 0].detach().clone(),
            actions.detach().clone(),
            costs.detach().clone(),
        ))
        return costs

    def rollout(self, info, actions):
        return {'predicted_embed_0': actions}


def make_solver(*, n_envs=1, n_steps=1, num_samples=1, batch_size=None,
                lr=1.5, **kwargs):
    model = QuadraticCostModel()
    solver = GradientSolver(
        model, n_steps=n_steps, num_samples=num_samples, batch_size=batch_size,
        var_scale=0.3, early_stop_patience=0, optimizer_kwargs={'lr': lr},
        seed=42, **kwargs,
    )
    solver.configure(
        action_space=gym.spaces.Box(
            low=-10, high=10, shape=(n_envs, 1), dtype=np.float32
        ),
        n_envs=n_envs,
        config=SimpleNamespace(
            horizon=3, action_block=1, receding_horizon=1,
            cost_last_n=1, intermediate_cost_weight=0.0,
        ),
    )
    return solver, model


def solve(solver, targets, *, horizon=1, init_action=None):
    return solver.solve(
        info_dict={
            'target': torch.tensor(targets, dtype=torch.float32)[:, None],
            'embed_0': torch.zeros(len(targets), 1, 1),
            'goal_embed_0': torch.zeros(len(targets), 1, 1),
        },
        planning_horizon=horizon, init_action=init_action,
    )


@pytest.mark.parametrize('retention', [None, False, True])
def test_default_and_opt_in_selection_after_overshoot(retention):
    kwargs = {} if retention is None else {'retain_best': retention}
    solver, _ = make_solver(**kwargs)
    result = solve(solver, [1.0])
    # One SGD update with lr=1.5 raises cost from 1 (action 0) to 4 (action 3).
    expected = (
        torch.zeros(1, 1, 1) if retention else torch.full((1, 1, 1), 3.0)
    )
    torch.testing.assert_close(result['actions'], expected)
    torch.testing.assert_close(
        result['predictions']['predicted_embed_0'], expected
    )
    assert solver.solve_records[-1]['retain_best'] is bool(retention)


def test_final_updated_actions_are_also_eligible():
    solver, _ = make_solver(retain_best=True, n_steps=2, lr=0.75)
    result = solve(solver, [1.0])
    torch.testing.assert_close(result['actions'], torch.tensor([[[0.75]]]))


def test_out_of_bounds_initial_optimum_is_not_returned():
    solver, _ = make_solver(retain_best=True, action_clip_sigma=0.1)
    result = solve(solver, [0.3], init_action=torch.tensor([[[0.3]]]))
    torch.testing.assert_close(result['actions'], torch.tensor([[[0.1]]]))


def test_early_stopping_and_cache_reset():
    solver, _ = make_solver(retain_best=True, n_steps=10)
    solver.min_its = 1
    solver.early_stop_patience = 1
    result = solve(solver, [1.0])
    torch.testing.assert_close(result['actions'], torch.zeros(1, 1, 1))
    assert solver.solve_records[-1]['n_iters'] == [2]
    result = solve(solver, [2.0], init_action=torch.tensor([[[1.5]]]))
    torch.testing.assert_close(result['actions'], torch.tensor([[[1.5]]]))


@pytest.mark.parametrize('batch_size', [None, 1, 2])
@pytest.mark.parametrize('horizon', [1, 3])
def test_best_sample_and_iteration_are_selected_per_environment(
    batch_size, horizon
):
    targets = [1.0, -2.0, 0.5]
    solver, model = make_solver(
        n_envs=3, n_steps=3, retain_best=True, num_samples=4,
        batch_size=batch_size,
    )
    solver._config.cost_last_n = horizon
    solver._config.intermediate_cost_weight = 0.25
    result = solve(
        solver, targets, horizon=horizon,
        init_action=torch.tensor([0.8, -1.8, 0.2])[:, None, None],
    )
    assert model.objectives == [(horizon, 0.25)] * len(model.evaluations)
    for env, target in enumerate(targets):
        costs_seen, actions_seen = [], []
        for recorded_targets, actions, costs in model.evaluations:
            for row in torch.where(recorded_targets == target)[0]:
                costs_seen.append(costs[row].flatten())
                actions_seen.append(actions[row])
        all_costs = torch.cat(costs_seen)
        all_actions = torch.cat(actions_seen)
        torch.testing.assert_close(
            result['actions'][env], all_actions[all_costs.argmin()]
        )


def test_retention_preserves_evaluation_budget_optimizer_and_random_draws():
    baseline, baseline_model = make_solver(
        n_steps=3, num_samples=4, action_noise=0.1
    )
    retained, retained_model = make_solver(
        n_steps=3, num_samples=4, action_noise=0.1, retain_best=True
    )
    solve(baseline, [1.0], horizon=3)
    solve(retained, [1.0], horizon=3)
    assert len(baseline_model.evaluations) == 4
    assert len(retained_model.evaluations) == 4
    for original, candidate in zip(
        baseline_model.evaluations, retained_model.evaluations, strict=True
    ):
        for original_tensor, candidate_tensor in zip(
            original, candidate, strict=True
        ):
            torch.testing.assert_close(
                original_tensor, candidate_tensor, rtol=0, atol=0
            )
    torch.testing.assert_close(baseline.init, retained.init, rtol=0, atol=0)
    assert baseline.solve_records[-1]['n_iters'] == [3]
    assert retained.solve_records[-1]['n_iters'] == [3]


@pytest.mark.parametrize(
    'bad_cost', [float('nan'), float('inf'), -float('inf')]
)
def test_nonfinite_sample_cannot_hide_finite_candidate(bad_cost):
    actions = torch.tensor([[[[2.0]], [[1.0]], [[float('nan')]]]])
    best_actions, best_costs = GradientSolver._retain_best_candidate(
        actions, torch.tensor([[bad_cost, 1.0, 0.0]]), None, None
    )
    actions.add_(10)
    torch.testing.assert_close(best_actions, torch.ones(1, 1, 1))
    torch.testing.assert_close(best_costs, torch.ones(1))


def test_earlier_finite_candidate_survives_invalid_final_cost():
    class InvalidFinalModel(QuadraticCostModel):
        def get_cost(self, info, actions):
            costs = super().get_cost(info, actions)
            if len(self.evaluations) == 1:
                return costs
            return costs * 0 + float('inf')

    solver, _ = make_solver(retain_best=True)
    solver.model = InvalidFinalModel()
    torch.testing.assert_close(
        solve(solver, [1.0])['actions'], torch.zeros(1, 1, 1)
    )


@pytest.mark.parametrize(
    'bad_cost', [float('nan'), float('inf'), -float('inf')]
)
def test_no_eligible_candidate_reports_global_environment_index(bad_cost):
    class InvalidCostModel(QuadraticCostModel):
        def get_cost(self, info, actions):
            costs = super().get_cost(info, actions)
            return torch.where(
                info['target'][..., 0] < 0, costs * 0 + bad_cost, costs
            )

    solver, _ = make_solver(n_envs=3, retain_best=True, batch_size=2)
    solver.model = InvalidCostModel()
    with pytest.raises(RuntimeError, match=r'environment indices \[2\]'):
        solve(solver, [1.0, 2.0, -1.0])
    assert solver.solve_records == []


def test_invalid_final_cost_cannot_resurrect_infeasible_initial_optimum():
    class InvalidInsideBoundsModel(QuadraticCostModel):
        def get_cost(self, info, actions):
            costs = super().get_cost(info, actions)
            inside = (actions.abs() <= 0.1).all(dim=-1).all(dim=-1)
            return torch.where(inside, costs * 0 + float('inf'), costs)

    solver, _ = make_solver(retain_best=True, action_clip_sigma=0.1)
    solver.model = InvalidInsideBoundsModel()
    with pytest.raises(
        RuntimeError, match='no finite, feasible action candidate'
    ):
        solve(solver, [0.3], init_action=torch.tensor([[[0.3]]]))
