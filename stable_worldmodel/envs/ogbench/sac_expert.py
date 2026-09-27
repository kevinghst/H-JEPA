"""Minimal OGBench SAC expert code needed for AntMaze data collection.

This module vendors the small subset of OGBench's ``impls`` code required to
restore the Ant directional SAC expert and run ``sample_actions``. It is not a
training implementation.
"""

from __future__ import annotations

import copy
import functools
import glob
import pickle
from collections.abc import Mapping, Sequence
from typing import Any

import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax


nonpytree_field = functools.partial(flax.struct.field, pytree_node=False)


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """Split a JAX PRNG key before each policy call."""

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped


class ModuleDict(nn.Module):
    """Dictionary wrapper for named Flax modules."""

    modules: dict[str, nn.Module]

    @nn.compact
    def __call__(self, *args, name=None, **kwargs):
        if name is None:
            if kwargs.keys() != self.modules.keys():
                raise ValueError(
                    'kwargs must contain arguments for every module when '
                    '`name` is not specified'
                )
            out = {}
            for key, value in kwargs.items():
                if isinstance(value, Mapping):
                    out[key] = self.modules[key](**value)
                elif isinstance(value, Sequence):
                    out[key] = self.modules[key](*value)
                else:
                    out[key] = self.modules[key](value)
            return out

        return self.modules[name](*args, **kwargs)


class TrainState(flax.struct.PyTreeNode):
    """Small TrainState compatible with OGBench expert checkpoints."""

    step: int
    apply_fn: Any = nonpytree_field()
    model_def: Any = nonpytree_field()
    params: Any
    tx: Any = nonpytree_field()
    opt_state: Any

    @classmethod
    def create(cls, model_def, params, tx=None, **kwargs):
        opt_state = tx.init(params) if tx is not None else None
        return cls(
            step=1,
            apply_fn=model_def.apply,
            model_def=model_def,
            params=params,
            tx=tx,
            opt_state=opt_state,
            **kwargs,
        )

    def __call__(self, *args, params=None, method=None, **kwargs):
        if params is None:
            params = self.params
        variables = {'params': params}
        method_name = getattr(self.model_def, method) if method is not None else None
        return self.apply_fn(variables, *args, method=method_name, **kwargs)

    def select(self, name):
        return functools.partial(self, name=name)


def default_init(scale=1.0):
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


def ensemblize(cls, num_qs, out_axes=0, **kwargs):
    return nn.vmap(
        cls,
        variable_axes={'params': 0},
        split_rngs={'params': True},
        in_axes=None,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs,
    )


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Any = nn.gelu
    activate_final: bool = False
    kernel_init: Any = default_init()
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
        return x


class LogParam(nn.Module):
    init_value: float = 1.0

    @nn.compact
    def __call__(self):
        log_value = self.param(
            'log_value', init_fn=lambda key: jnp.full((), jnp.log(self.init_value))
        )
        return jnp.exp(log_value)


class TransformedWithMode(distrax.Transformed):
    def mode(self):
        return self.bijector.forward(self.distribution.mode())


class GCActor(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    log_std_min: float = -5
    log_std_max: float = 2
    tanh_squash: bool = False
    state_dependent_std: bool = False
    const_std: bool = True
    final_fc_init_scale: float = 1e-2
    gc_encoder: nn.Module | None = None

    def setup(self):
        self.actor_net = MLP(self.hidden_dims, activate_final=True)
        self.mean_net = nn.Dense(
            self.action_dim, kernel_init=default_init(self.final_fc_init_scale)
        )
        if self.state_dependent_std:
            self.log_std_net = nn.Dense(
                self.action_dim, kernel_init=default_init(self.final_fc_init_scale)
            )
        elif not self.const_std:
            self.log_stds = self.param(
                'log_stds', nn.initializers.zeros, (self.action_dim,)
            )

    def __call__(self, observations, goals=None, goal_encoded=False, temperature=1.0):
        if self.gc_encoder is not None:
            inputs = self.gc_encoder(
                observations, goals, goal_encoded=goal_encoded
            )
        else:
            inputs = [observations]
            if goals is not None:
                inputs.append(goals)
            inputs = jnp.concatenate(inputs, axis=-1)

        outputs = self.actor_net(inputs)
        means = self.mean_net(outputs)
        if self.state_dependent_std:
            log_stds = self.log_std_net(outputs)
        elif self.const_std:
            log_stds = jnp.zeros_like(means)
        else:
            log_stds = self.log_stds

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)
        distribution = distrax.MultivariateNormalDiag(
            loc=means,
            scale_diag=jnp.exp(log_stds) * temperature,
        )
        if self.tanh_squash:
            distribution = TransformedWithMode(
                distribution, distrax.Block(distrax.Tanh(), ndims=1)
            )
        return distribution


class GCValue(nn.Module):
    hidden_dims: Sequence[int]
    layer_norm: bool = True
    ensemble: bool = True
    gc_encoder: nn.Module | None = None

    def setup(self):
        mlp_module = ensemblize(MLP, 2) if self.ensemble else MLP
        self.value_net = mlp_module(
            (*self.hidden_dims, 1),
            activate_final=False,
            layer_norm=self.layer_norm,
        )

    def __call__(self, observations, goals=None, actions=None):
        if self.gc_encoder is not None:
            inputs = [self.gc_encoder(observations, goals)]
        else:
            inputs = [observations]
            if goals is not None:
                inputs.append(goals)
        if actions is not None:
            inputs.append(actions)
        inputs = jnp.concatenate(inputs, axis=-1)
        return self.value_net(inputs).squeeze(-1)


class SACAgent(flax.struct.PyTreeNode):
    """SAC agent subset needed for restoring and sampling actions."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @jax.jit
    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        dist = self.network.select('actor')(
            observations,
            goals,
            temperature=temperature,
        )
        actions = dist.sample(seed=seed)
        return jnp.clip(actions, -1, 1)

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)
        action_dim = ex_actions.shape[-1]

        if config['target_entropy'] is None:
            config['target_entropy'] = (
                -config['target_entropy_multiplier'] * action_dim
            )

        critic_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=True,
        )
        actor_def = GCActor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            log_std_min=-5,
            tanh_squash=config['tanh_squash'],
            state_dependent_std=config['state_dependent_std'],
            const_std=False,
            final_fc_init_scale=config['actor_fc_scale'],
        )
        alpha_def = LogParam()

        network_info = {
            'critic': (critic_def, (ex_observations, None, ex_actions)),
            'target_critic': (
                copy.deepcopy(critic_def),
                (ex_observations, None, ex_actions),
            ),
            'actor': (actor_def, (ex_observations, None)),
            'alpha': (alpha_def, ()),
        }
        network_def = ModuleDict({k: v[0] for k, v in network_info.items()})
        network_args = {k: v[1] for k, v in network_info.items()}
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)
        network.params['modules_target_critic'] = network.params['modules_critic']

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def restore_agent(agent, restore_path, restore_epoch):
    candidates = glob.glob(restore_path)
    if len(candidates) != 1:
        raise FileNotFoundError(
            f'Expected exactly one restore path, found {candidates}'
        )

    params_path = f'{candidates[0]}/params_{restore_epoch}.pkl'
    with open(params_path, 'rb') as f:
        load_dict = pickle.load(f)

    agent = flax.serialization.from_state_dict(agent, load_dict['agent'])
    print(f'Restored from {params_path}')
    return agent
