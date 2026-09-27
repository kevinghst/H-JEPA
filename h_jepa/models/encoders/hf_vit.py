from torch import nn
from transformers import ViTConfig, ViTModel


VIT_SIZE_CONFIGS = {
    "tiny": {"hidden_size": 192, "num_hidden_layers": 12, "num_attention_heads": 3},
    "small": {"hidden_size": 384, "num_hidden_layers": 12, "num_attention_heads": 6},
    "base": {"hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12},
    "large": {"hidden_size": 1024, "num_hidden_layers": 24, "num_attention_heads": 16},
    "huge": {"hidden_size": 1280, "num_hidden_layers": 32, "num_attention_heads": 16},
}


def create_hf_vit(
    size: str = "tiny",
    patch_size: int = 16,
    image_size: int = 224,
    pretrained: bool = False,
    use_mask_token: bool = True,
    **kwargs,
) -> nn.Module:
    if size not in VIT_SIZE_CONFIGS:
        raise ValueError(
            f"Invalid size '{size}'. Choose from {list(VIT_SIZE_CONFIGS.keys())}"
        )

    if pretrained:
        model_name = f"google/vit-{size}-patch{patch_size}-{image_size}"
        model = ViTModel.from_pretrained(
            model_name,
            add_pooling_layer=False,
            use_mask_token=use_mask_token,
        )
    else:
        config_params = dict(VIT_SIZE_CONFIGS[size])
        config_params["intermediate_size"] = config_params["hidden_size"] * 4
        config_params["image_size"] = image_size
        config_params["patch_size"] = patch_size
        config_params.update(kwargs)

        config = ViTConfig(**config_params)
        model = ViTModel(
            config,
            add_pooling_layer=False,
            use_mask_token=use_mask_token,
        )

    model.config.interpolate_pos_encoding = True
    return model
