"""Gradient-based solver for model-based planning."""

import time
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from .solver import Costable, get_planning_horizon


class GradientSolver(torch.nn.Module):
    """Gradient-based solver using backpropagation through the world model.

    Args:
        model: World model implementing the Costable protocol.
        n_steps: Number of gradient descent iterations.
        min_its: Minimum iterations to run before early stopping can trigger.
        batch_size: Number of environments to process in parallel.
        var_scale: Initial variance scale for action perturbations.
        num_samples: Number of action samples to optimize in parallel.
        action_noise: Noise added to actions during optimization.
        early_stop_patience: Consecutive non-improving steps before stopping.
        early_stop_rel_delta: Minimum relative improvement required to reset patience.
        action_clip_sigma: Clip actions to prior mean ± sigma·std each step; None uses queue quantiles.
        device: Device for tensor computations.
        seed: Random seed for reproducibility.
        optimizer_cls: PyTorch optimizer class to use.
        optimizer_kwargs: Keyword arguments for the optimizer.
    """

    def __init__(
        self,
        model: Costable,
        n_steps: int,
        min_its: int = 30,
        batch_size: int | None = None,
        var_scale: float = 1,
        num_samples: int = 1,
        action_noise: float = 0.0,
        early_stop_patience: int = 5,
        early_stop_rel_delta: float = 1e-2,
        action_clip_sigma: float | None = None,
        device: str | torch.device = 'cpu',
        seed: int = 1234,
        optimizer_cls: type[torch.optim.Optimizer] = torch.optim.SGD,
        optimizer_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.n_steps = n_steps
        self.min_its = max(0, int(min_its))
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.var_scale = var_scale
        self.action_noise = action_noise
        self.early_stop_patience = max(0, int(early_stop_patience))
        self.early_stop_rel_delta = max(0.0, float(early_stop_rel_delta))
        self.action_clip_sigma = action_clip_sigma
        self.device = device
        self.torch_gen = torch.Generator(device=device).manual_seed(seed)

        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = (
            optimizer_kwargs if optimizer_kwargs is not None else {'lr': 1.0}
        )

        self._n_envs = None
        self._action_dim = None
        self._config = None
        # Per-solve compute metadata, appended on every solve() call; collected by
        # the eval to compute planner FLOPs (see planning_eval._collect_solve_records).
        self.solve_records = []

    def configure(
        self,
        *,
        action_space: gym.Space,
        n_envs: int,
        config: Any,
        level: int = 1,
    ) -> None:
        """Configure the solver with environment specifications."""
        self._action_space = action_space
        self._n_envs = n_envs
        self._config = config
        self.level = int(level)
        if level == 1:
            self._action_dim = int(np.prod(action_space.shape[1:]))
        else:
            self._action_dim = int(config.action_dim)

    def _expand_info_dict_for_samples(
        self,
        info_dict: dict,
        start_idx: int,
        end_idx: int,
        num_samples: int,
    ) -> dict:
        """Slice env batch and add sample dimension expected by world models."""
        current_bs = end_idx - start_idx
        expanded_infos = {}

        for k, v in info_dict.items():
            if torch.is_tensor(v):
                v_batch = v[start_idx:end_idx].to(self.device)
                v_batch = v_batch.unsqueeze(1).expand(
                    current_bs, num_samples, *v_batch.shape[1:]
                )
            elif isinstance(v, np.ndarray):
                v_batch = v[start_idx:end_idx]
                v_batch = np.repeat(v_batch[:, None, ...], num_samples, axis=1)
            else:
                v_batch = v

            expanded_infos[k] = v_batch

        return expanded_infos

    @staticmethod
    def _has_significant_improvement(
        score: float,
        best_score: float,
        rel_delta: float,
    ) -> bool:
        """Return True when `score` improves enough over `best_score`."""
        if not np.isfinite(best_score):
            return True

        improvement = best_score - score
        threshold = rel_delta * max(abs(best_score), 1e-12)
        return improvement > threshold

    @torch.inference_mode()
    def _rollout_final_actions(self, info_dict: dict, actions: torch.Tensor) -> dict:
        """Roll out the model once with optimized actions and return predictions."""
        action_candidates = actions.to(self.device).unsqueeze(1)
        info_dict = self._expand_info_dict_for_samples(
            info_dict, start_idx=0, end_idx=self.n_envs, num_samples=1
        )

        rollout_info = self.model.rollout(
            info_dict,
            action_candidates,
        )

        predictions: dict[str, torch.Tensor] = {}

        for k, v in rollout_info.items():
            if not (k.startswith('predicted') and torch.is_tensor(v)):
                continue

            value = v.detach().cpu()
            if value.ndim > 1 and value.shape[1] == 1:
                value = value[:, 0]
            predictions[k] = value

        return predictions

    def _get_action_prior_stats(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Estimate mean/std prior from queued latent actions, if available."""
        queue = getattr(self.model, 'get_latent_action_queue', lambda: None)()
        if not (torch.is_tensor(queue) and queue.numel() > 0):
            return None

        assert queue.ndim == 2

        queue = queue.detach().to(self.device)
        mean = queue.mean(dim=0)
        std = queue.std(dim=0, unbiased=False).clamp_min(1e-4)

        return mean, std

    def _get_action_clip_bounds(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Estimate per-dimension clipping bounds from queued latent actions."""
        if self.action_clip_sigma is not None:
            prior_stats = self._get_action_prior_stats()
            if prior_stats is None:
                mean = torch.zeros(self.action_dim, device=self.device)
                std = torch.ones(self.action_dim, device=self.device)
            else:
                mean, std = prior_stats
            return mean - self.action_clip_sigma * std, mean + self.action_clip_sigma * std

        queue = getattr(self.model, 'get_latent_action_queue', lambda: None)()
        if not (torch.is_tensor(queue) and queue.numel() > 0):
            return None

        assert queue.ndim == 2

        queue = queue.detach().to(self.device)
        lower = torch.quantile(queue, 0.02, dim=0)
        upper = torch.quantile(queue, 0.98, dim=0)

        return lower, upper

    @property
    def n_envs(self) -> int:
        """Number of parallel environments."""
        return self._n_envs

    @property
    def action_dim(self) -> int:
        """Flattened action dimension including action_block grouping."""
        return self._action_dim * self._config.action_block

    @property
    def horizon(self) -> int:
        """Planning horizon in timesteps."""
        return self._config.horizon

    def __call__(self, *args: Any, **kwargs: Any) -> dict:
        """Make solver callable, forwarding to solve()."""
        return self.solve(*args, **kwargs)

    def init_action(
        self,
        actions: torch.Tensor | None = None,
        planning_horizon: int | None = None,
    ) -> None:
        """Initialize the action tensor for optimization."""
        if planning_horizon is None:
            planning_horizon = self.horizon

        print(f"level {self.level} planning horizon: {planning_horizon }")

        prior_stats = self._get_action_prior_stats()
        if self.model.level > 1:
            assert prior_stats is not None, (
                'Higher-level GD requires latent action priors from the model.'
            )

        if prior_stats is None:
            base_mean = torch.zeros(self.action_dim, device=self.device)
            base_std = torch.ones(self.action_dim, device=self.device)
        else:
            base_mean, base_std = prior_stats
        base_std = base_std * self.var_scale

        if actions is None:
            actions = base_mean.view(1, 1, -1).expand(
                self._n_envs, planning_horizon, -1
            ).clone()
        else:
            actions = actions.to(self.device)
            if actions.shape[1] < planning_horizon:
                remaining = planning_horizon - actions.shape[1]
                new_actions = base_mean.view(1, 1, -1).expand(
                    self._n_envs, remaining, -1
                ).clone()
                actions = torch.cat([actions, new_actions], dim=1)
            else:
                actions = actions[:, :planning_horizon]

        actions = actions.unsqueeze(1).repeat_interleave(
            self.num_samples, dim=1
        )
        actions[:, 1:] += (
            torch.randn(
                actions[:, 1:].shape,
                generator=self.torch_gen,
                device=self.device,
            )
            * base_std.view(1, 1, 1, -1)
        )

        if hasattr(self, 'init') and self.init.shape == actions.shape:
            self.init.copy_(actions)
        else:
            self.init = torch.nn.Parameter(actions)

    def solve(
        self,
        info_dict: dict,
        init_action: torch.Tensor | None = None,
        planning_horizon: int | None = None,
        steps_taken: int | None = None,
        eval_budget: int | None = None,
    ) -> dict:
        """Solve the planning problem using gradient descent."""
        start_time = time.time()
        cost_last_n = max(1, int(self._config.cost_last_n))
        intermediate_cost_weight = float(self._config.intermediate_cost_weight)
        if planning_horizon is None:
            planning_horizon = get_planning_horizon(
                configured_horizon=self.horizon,
                replanning_interval=int(self._config.receding_horizon) * int(self._config.action_block),
                steps_taken=steps_taken,
                eval_budget=eval_budget,
            )
        outputs = {}

        with torch.no_grad():
            self.init_action(init_action, planning_horizon=planning_horizon)
        clip_bounds = self._get_action_clip_bounds()

        # Determine batch size (default to all envs if not specified which can cause memory issues)
        batch_size = (
            self.batch_size if self.batch_size is not None else self.n_envs
        )
        total_envs = self.n_envs

        # Lists to hold results from each batch to be concatenated later
        batch_top_actions_list = []

        # --- Outer Loop: Iterate over batches ---
        per_env_iters = []
        per_env_time = []
        for start_idx in range(0, total_envs, batch_size):
            end_idx = min(start_idx + batch_size, total_envs)
            current_bs = end_idx - start_idx
            batch_t0 = time.time()

            batch_init = self.init[start_idx:end_idx].clone().detach()
            batch_init.requires_grad = True

            # We initialize the optimizer class passed in __init__ with the kwargs
            optim = self.optimizer_cls([batch_init], **self.optimizer_kwargs)

            # Prepare Batch Infos
            # Slice the input info_dict and then expand dimensions
            expanded_infos = self._expand_info_dict_for_samples(
                info_dict, start_idx, end_idx, num_samples=self.num_samples
            )

            # Perform Gradient Descent for this batch
            best_batch_score = float('inf')
            stalled_steps = 0

            for step in range(self.n_steps):
                current_info = expanded_infos.copy()
                current_info['cost_last_n'] = cost_last_n
                current_info['intermediate_cost_weight'] = intermediate_cost_weight

                # Calculate cost using the batch parameter
                costs = self.model.get_cost(current_info, batch_init)
                for k in ('embed_0', 'goal_embed_0'):  # encode the observation and goal once per solve
                    expanded_infos.setdefault(k, current_info[k])

                assert isinstance(costs, torch.Tensor), (
                    f'Got {type(costs)} cost, expect torch.Tensor'
                )
                assert (
                    costs.ndim == 2
                    and costs.shape[0] == current_bs
                    and costs.shape[1] == self.num_samples
                ), (
                    f'Cost should be of shape ({current_bs}, {self.num_samples}), got {costs.shape}'
                )
                assert costs.requires_grad, (
                    'Cost must requires_grad for GD solver.'
                )

                cost = costs.sum()  # Sum cost for this batch
                cost.backward()
                optim.step()
                optim.zero_grad(set_to_none=True)

                # Add noise
                if self.action_noise > 0:
                    batch_init.data += (
                        torch.randn(
                            batch_init.shape,
                            generator=self.torch_gen,
                            device=batch_init.device,
                        )
                        * self.action_noise
                    )

                if clip_bounds is not None:
                    lower, upper = clip_bounds
                    batch_init.data.clamp_(
                        min=lower.view(1, 1, 1, -1),
                        max=upper.view(1, 1, 1, -1),
                    )

                if self.early_stop_patience > 0:
                    total_best_costs = costs.min(dim=1).values.detach().cpu().tolist()
                    batch_score = float(np.mean(total_best_costs))
                    if self._has_significant_improvement(
                        batch_score,
                        best_batch_score,
                        self.early_stop_rel_delta,
                    ):
                        best_batch_score = batch_score
                        stalled_steps = 0
                    else:
                        stalled_steps += 1
                        if (
                            step + 1 >= self.min_its
                            and stalled_steps >= self.early_stop_patience
                        ):
                            break

            iters_run = step + 1  # actual GD iterations for this batch (early-stop aware)

            # Update the global self.init with the optimized batch values
            with torch.no_grad():
                self.init[start_idx:end_idx] = batch_init

            with torch.no_grad():
                final_info = expanded_infos.copy()
                final_info['cost_last_n'] = cost_last_n
                final_info['intermediate_cost_weight'] = intermediate_cost_weight
                final_costs = self.model.get_cost(final_info, batch_init)

            top_idx = torch.argsort(final_costs, dim=1)[:, 0]
            batch_indices = torch.arange(current_bs, device=batch_init.device)

            top_actions_batch = batch_init[batch_indices, top_idx]
            batch_top_actions_list.append(top_actions_batch.detach().cpu())

            per_env_iters.extend([int(iters_run)] * current_bs)
            per_env_time.extend([time.time() - batch_t0] * current_bs)

        # Concatenate all batch results
        outputs['actions'] = torch.cat(batch_top_actions_list, dim=0)
        outputs['predictions'] = self._rollout_final_actions(
            info_dict, outputs['actions']
        )
        end_time = time.time()
        print(
            f'GradientSolver.solve completed in {end_time - start_time:.4f} seconds.'
        )

        self.solve_records.append({
            'level': int(getattr(self, 'level', 1)),
            'horizon': int(planning_horizon),
            'num_samples': int(self.num_samples),
            'n_iters': per_env_iters,          # one entry per env (batch_size=1 -> per episode)
            'wall_clock': end_time - start_time,
            'wall_clock_per_env': per_env_time,
        })

        return outputs
