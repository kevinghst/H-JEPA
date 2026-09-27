from __future__ import annotations

import torch
from torch import nn

from models.module import MLP


PROBER_CONV_LAYERS_CONFIG = {
    "c": [
        (-1, 32, 3, 1, 1),
        ("max_pool", 2, 2, 0),
        (32, 32, 3, 1, 1),
        ("max_pool", 2, 2, 0),
        (32, 32, 3, 1, 1),
        (32, 32, 3, 1, 1),
        ("fc", -1, 2),
    ],
}


class ConvProber(nn.Module):
    """Conv probe for spatial embeddings."""

    def __init__(
        self,
        input_dim: tuple[int, int, int],
        output_dim: int,
        *,
        subtype: str = "c",
        group_factor: int = 4,
    ):
        super().__init__()
        if subtype not in PROBER_CONV_LAYERS_CONFIG:
            raise ValueError(f"Unsupported conv prober subtype {subtype!r}")

        self.input_dim = tuple(int(dim) for dim in input_dim)
        self.output_dim = int(output_dim)
        self.net = _build_conv_prober(
            PROBER_CONV_LAYERS_CONFIG[subtype],
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            group_factor=group_factor,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 4:
            raise ValueError(
                "ConvProber expects spatial input with shape (..., C, H, W), "
                f"got {tuple(x.shape)}"
            )

        leading_shape = x.shape[:-3]
        x = x.reshape(-1, *x.shape[-3:]).float()
        x = self.net(x)
        return x.reshape(*leading_shape, self.output_dim)


def _build_conv_prober(
    layers_config,
    *,
    input_dim: tuple[int, int, int],
    output_dim: int,
    group_factor: int = 4,
) -> nn.Sequential:
    input_channels = input_dim[0]
    layers = []

    for i, layer_config in enumerate(layers_config[:-1]):
        if isinstance(layer_config[0], str) and "pool" in layer_config[0]:
            pool_type, kernel_size, stride, padding = layer_config
            if pool_type == "max_pool":
                layers.append(nn.MaxPool2d(kernel_size, stride, padding))
            elif pool_type == "avg_pool":
                layers.append(nn.AvgPool2d(kernel_size, stride, padding))
            else:
                raise ValueError(f"Unsupported pool layer {pool_type!r}")
            continue

        in_channels, out_channels, kernel_size, stride, padding = layer_config
        if i == 0:
            in_channels = input_channels
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
        layers.append(nn.GroupNorm(out_channels // group_factor, out_channels))
        layers.append(nn.ReLU())

    last_layer_config = layers_config[-1]
    if last_layer_config[0] != "fc":
        raise ValueError("Conv prober configs must end with an fc layer")

    _, fc_in_dim, _ = last_layer_config
    if fc_in_dim == -1:
        with torch.no_grad():
            sample = torch.zeros(1, *input_dim)
            sample_output = nn.Sequential(*layers)(sample)
            fc_in_dim = int(torch.prod(torch.tensor(sample_output.shape)))

    layers.append(nn.Flatten(1, -1))
    layers.append(nn.Linear(fc_in_dim, output_dim))
    return nn.Sequential(*layers)


def build_prober(probe_cfg, *, input_dim, output_dim: int) -> nn.Module:
    prober_type = str(probe_cfg.get("type", "mlp")).lower()

    if prober_type == "mlp":
        return MLP(
            input_dim=int(input_dim),
            hidden_dim=int(probe_cfg.get("hidden_dim", 512)),
            output_dim=int(output_dim),
        )

    if prober_type == "linear":
        # A genuine linear readout: no hidden layer, no nonlinearity. `mlp` stays
        # the default, so nothing that omits `type` changes behaviour.
        return nn.Linear(int(input_dim), int(output_dim))

    if prober_type == "conv":
        return ConvProber(
            input_dim=tuple(input_dim),
            output_dim=int(output_dim),
            subtype=str(probe_cfg.get("subtype", "c")),
        )

    raise ValueError(f"Unsupported prober type {probe_cfg.get('type')!r}")
