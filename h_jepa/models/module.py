"""Module Implementations for Transformer-based Models"""

from einops import rearrange
import torch
from torch.nn import functional as F
import torch.nn as nn


def build_projector(projector_cfg, *, input_dim: int, output_dim: int) -> nn.Module:
    projector_type = str(projector_cfg.get("type", "mlp")).lower()

    if projector_type == "identity":
        if int(input_dim) != int(output_dim):
            raise ValueError(
                "Identity projector cannot change feature size: "
                f"got input_dim={input_dim}, output_dim={output_dim}."
            )
        return nn.Identity()

    if projector_type == "mlp":
        return MLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=2048,
            norm_fn=nn.BatchNorm1d,
        )

    raise ValueError(f"Unsupported projector type '{projector_cfg.get('type')}'")


def modulate(x, shift, scale):
    """AdaLN-zero modulation"""
    return x * (1 + scale) + shift


class FeedForward(nn.Module):
    """FeedForward network used in Transformers"""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Scaled dot-product attention with causal masking"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, causal=True):
        """
        x : (B, T, D)
        """
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # q, k, v: (B, heads, T, dim_head)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Block(nn.Module):
    """Standard Transformer block"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    """Standard Transformer with support for AdaLN-zero blocks"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        block_class=Block,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])

        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )

        for _ in range(depth):
            self.layers.append(
                block_class(hidden_dim, heads, dim_head, mlp_dim, dropout)
            )

    def forward(self, x, c=None):
        x = self.input_proj(x)

        if c is not None:
            c = self.cond_proj(c)

        for block in self.layers:
            x = block(x) if isinstance(block, Block) else block(x, c)
        x = self.norm(x)
        x = self.output_proj(x)
        return x


class Embedder(nn.Module):
    def __init__(
        self,
        input_dim=10,
        smoothed_dim=10,
        emb_dim=10,
        mlp_scale=4,
        patch_embed=True,
    ):
        super().__init__()
        self.patch_embed = (
            nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
            if patch_embed
            else None
        )
        feature_dim = smoothed_dim if patch_embed else input_dim
        self.embed = nn.Sequential(
            nn.Linear(feature_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x):
        """
        x: (B, T, D) or (B, T, C, D)
        """
        x = x.float()
        if self.patch_embed is not None:
            if x.dim() == 3:
                x = x.permute(0, 2, 1)
                x = self.patch_embed(x)
                x = x.permute(0, 2, 1)
            elif x.dim() == 4:
                b, t, c, d = x.shape
                x = x.reshape(b, t * c, d)
                x = x.permute(0, 2, 1)
                x = self.patch_embed(x)
                x = x.permute(0, 2, 1)
                x = x.reshape(b, t, c, -1)
            else:
                raise ValueError(f"Expected 3D or 4D input, got shape {tuple(x.shape)}")
        x = self.embed(x)
        return x


class MLP(nn.Module):
    """Simple MLP with optional normalization and activation"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
        final_ln=False,
    ):
        super().__init__()
        out_dim = output_dim or input_dim
        norm_fn = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_fn,
            act_fn(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.final_norm = nn.LayerNorm(out_dim) if final_ln else nn.Identity()
    
    def forward(self, x):
        """
        x: (B*T, D)
        """
        return self.final_norm(self.net(x))


class ResidualLatentMLP(nn.Module):
    """MLP for refining latent vectors."""

    def __init__(
        self,
        input_dim,
        hidden_dim=384,
        output_dim=None,
        act_fn=nn.GELU,
        final_ln=True,
    ):
        super().__init__()
        out_dim = output_dim or input_dim
        self.input_norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = act_fn()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.final_norm = nn.LayerNorm(out_dim) if final_ln else nn.Identity()

    def forward(self, x):
        x = self.input_norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return self.final_norm(x)


class CLSDecoder(nn.Module):
    def __init__(
        self,
        cls_dim=384,
        img_size=224,
        patch_size=16,
        dim=256,
        heads=8,
        depth=3,
    ):
        super().__init__()

        self.num_patches = (img_size // patch_size) ** 2
        patch_dim = patch_size * patch_size * 3

        # Learnable queries with better initialization
        self.queries = nn.Parameter(torch.zeros(1, self.num_patches, dim))
        nn.init.normal_(self.queries, std=0.02)

        # Project CLS token
        self.cls_proj = nn.Sequential(
            nn.Linear(cls_dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
        )

        # Multiple cross-attention layers
        self.layers = nn.ModuleList()

        for _ in range(depth):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "cross_attn": nn.MultiheadAttention(
                            dim, heads, batch_first=True
                        ),
                        "norm1": nn.LayerNorm(dim),
                        "mlp": nn.Sequential(
                            nn.Linear(dim, dim * 4),
                            nn.GELU(),
                            nn.Linear(dim * 4, dim),
                        ),
                        "norm2": nn.LayerNorm(dim),
                    }
                )
            )

        # Final projection to pixels
        self.to_pixels = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, patch_dim),
        )

        self.patch_size = patch_size

    def forward(self, x):
        """
        x: (B*T, D) with T optional
        """
        B = x.size(0)
        P = self.num_patches

        # Project and expand CLS token
        kv = self.cls_proj(x).unsqueeze(1)  # (B, 1, h)
        q = self.queries.expand(B, -1, -1)  # (B, P, h)

        # Cross-attention layers with residuals
        for layer in self.layers:
            attn_out = layer["cross_attn"](q, kv, kv)[0]
            q = layer["norm1"](q + attn_out)
            mlp_out = layer["mlp"](q)
            q = layer["norm2"](q + mlp_out)

        # Project to pixels
        patches = self.to_pixels(q)  # (B, P, patch_dim)
        patches = patches.reshape(B, P, self.patch_size, self.patch_size, 3)

        # Reshape to image
        H = W = int(self.num_patches**0.5)
        patches = patches.reshape(B, H, W, self.patch_size, self.patch_size, 3)
        img = patches.permute(0, 5, 1, 3, 2, 4)
        img = img.reshape(B, 3, H * self.patch_size, W * self.patch_size)

        return img
