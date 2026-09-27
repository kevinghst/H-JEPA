from collections.abc import Mapping

# import stable_pretraining as spt
from torch import nn
from transformers import ViTConfig, ViTModel

from .hf_vit import create_hf_vit


def vit_hf(
    size: str = "tiny",
    patch_size: int = 16,
    image_size: int = 224,
    pretrained: bool = False,
    use_mask_token: bool = True,
    **kwargs,
) -> nn.Module:
    size_config = kwargs.pop("size_config", None)
    custom_size_configs = kwargs.pop("custom_size_configs", None)

    if size_config is None and custom_size_configs is not None:
        if not isinstance(custom_size_configs, Mapping):
            raise TypeError(
                "custom_size_configs must be a dict mapping size names to config dicts."
            )
        custom_size_configs = dict(custom_size_configs)
        if size in custom_size_configs:
            size_config = custom_size_configs[size]

    if size_config is None:
        backbone = create_hf_vit(
            size=size,
            patch_size=patch_size,
            image_size=image_size,
            pretrained=pretrained,
            use_mask_token=use_mask_token,
            **kwargs,
        )
    else:
        if pretrained:
            raise ValueError(
                "pretrained=True is not supported when using a custom ViT size_config."
            )
        if not isinstance(size_config, Mapping):
            raise TypeError("size_config must be a dict of ViTConfig parameters.")

        config_params = dict(size_config)
        required_keys = {"hidden_size", "num_hidden_layers", "num_attention_heads"}
        missing_keys = sorted(required_keys - set(config_params))
        if missing_keys:
            raise KeyError(
                "size_config is missing required keys: "
                + ", ".join(missing_keys)
            )

        config_params.setdefault(
            "intermediate_size",
            int(config_params["hidden_size"]) * 4,
        )
        config_params["image_size"] = image_size
        config_params["patch_size"] = patch_size
        config_params.update(kwargs)

        config = ViTConfig(**config_params)
        backbone = ViTModel(
            config,
            add_pooling_layer=False,
            use_mask_token=use_mask_token,
        )
        backbone.config.interpolate_pos_encoding = True

    return backbone


__all__ = ["vit_hf"]
