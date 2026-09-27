from __future__ import annotations

from collections.abc import Mapping

from torch import nn

from ..module import MLP, ResidualLatentMLP


def to_plain_encoder_dict(cfg) -> dict:
    if cfg is None:
        return {}
    if isinstance(cfg, Mapping):
        return {k: v for k, v in cfg.items()}
    raise TypeError(f"Expected encoder config mapping, got {type(cfg)}")


def build_mlp_encoder(encoder_kwargs: dict) -> tuple[nn.Module, int]:
    hidden_dim = encoder_kwargs.get("output_dim", encoder_kwargs.get("input_dim"))
    if hidden_dim is None:
        raise KeyError(
            "MLP encoder requires 'output_dim' or 'input_dim' in levelN.encoder config."
        )

    encoder = MLP(**encoder_kwargs)
    return encoder, int(hidden_dim)


def build_identity_encoder(encoder_kwargs: dict) -> tuple[nn.Module, int]:
    input_dim = encoder_kwargs.get("input_dim", None)
    output_dim = encoder_kwargs.get("output_dim", None)

    if input_dim is None and output_dim is None:
        raise KeyError(
            "Identity encoder requires 'input_dim' or 'output_dim' in "
            "levelN.encoder config."
        )

    if input_dim is None:
        return nn.Identity(), int(output_dim)
    if output_dim is None:
        return nn.Identity(), int(input_dim)

    input_dim = int(input_dim)
    output_dim = int(output_dim)
    if input_dim != output_dim:
        raise ValueError(
            "Identity encoder cannot change feature size: "
            f"got input_dim={input_dim}, output_dim={output_dim}."
        )

    return nn.Identity(), input_dim


def build_latent_mlp_encoder(encoder_kwargs: dict) -> tuple[nn.Module, int]:
    input_dim = encoder_kwargs.get("input_dim", None)
    output_dim = encoder_kwargs.get("output_dim", input_dim)

    if input_dim is None and output_dim is None:
        raise KeyError(
            "latent_mlp encoder requires 'input_dim' or 'output_dim' in "
            "levelN.encoder config."
        )

    encoder = ResidualLatentMLP(**encoder_kwargs)
    return encoder, int(output_dim)
