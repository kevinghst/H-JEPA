from __future__ import annotations

from .encoders import (
    build_identity_encoder,
    build_latent_mlp_encoder,
    build_mlp_encoder,
    to_plain_encoder_dict,
)
from .hf_vit import create_hf_vit
from .seq_encoder import FlattenedSequenceEncoder


def build_encoder(
    encoder_cfg,
    *,
    default_patch_size: int | None = None,
    default_image_size: int | None = None,
):
    """Build an encoder from levelN.encoder config.

    Supports:
    - type: vit -> models.encoders.hf_vit.create_hf_vit
    - type: mlp -> models.module.MLP
    - type: latent_mlp -> models.module.ResidualLatentMLP
    - type: seq_encoder -> models.encoders.seq_encoder.FlattenedSequenceEncoder
    - type: identity -> torch.nn.Identity

    Returns:
        tuple[module, int]: (encoder, hidden_dim)
    """
    cfg = to_plain_encoder_dict(encoder_cfg)
    if "type" not in cfg:
        raise KeyError("Encoder config must include a 'type' field")

    encoder_type = str(cfg["type"]).lower()
    encoder_kwargs = {k: v for k, v in cfg.items() if k != "type"}

    if encoder_type == "vit":
        # Support config alias: levelN.encoder.scale -> create_hf_vit(size=...)
        if "size" not in encoder_kwargs and "scale" in encoder_kwargs:
            encoder_kwargs["size"] = encoder_kwargs.pop("scale")

        if "size" not in encoder_kwargs:
            raise KeyError("ViT encoder requires 'size' or 'scale' in levelN.encoder")

        if "patch_size" not in encoder_kwargs and default_patch_size is not None:
            encoder_kwargs["patch_size"] = default_patch_size

        if "image_size" not in encoder_kwargs and default_image_size is not None:
            encoder_kwargs["image_size"] = default_image_size

        encoder = create_hf_vit(**encoder_kwargs)
        hidden_dim = int(encoder.config.hidden_size)
        return encoder, hidden_dim

    if encoder_type == "mlp":
        return build_mlp_encoder(encoder_kwargs)

    if encoder_type == "latent_mlp":
        return build_latent_mlp_encoder(encoder_kwargs)

    if encoder_type == "seq_encoder":
        encoder = FlattenedSequenceEncoder(**encoder_kwargs)
        return encoder, int(encoder.output_dim)

    if encoder_type == "identity":
        return build_identity_encoder(encoder_kwargs)

    raise ValueError(f"Unsupported encoder type '{cfg['type']}'")
