import torch
import torch.nn.functional as F
from torch import nn


class _CTFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class _CTAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.dropout = dropout
        self.to_qkv = nn.Linear(dim, dim_head * heads * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(dim_head * heads, dim), nn.Dropout(dropout))

    def forward(self, x):
        q, k, v = (
            t.view(t.shape[0], t.shape[1], self.heads, -1).transpose(1, 2)
            for t in self.to_qkv(x).chunk(3, dim=-1)
        )  # each [B, heads, T, dim_head]
        drop = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=True)
        out = out.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1)  # [B, T, heads * dim_head]
        return self.to_out(out)


class _CTConditionalBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_ratio, dropout, adaln_init_scale):
        super().__init__()
        self.attn = _CTAttention(dim, heads, dim_head, dropout)
        self.mlp = _CTFeedForward(dim, int(mlp_ratio * dim), dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.normal_(self.adaLN_modulation[-1].weight, std=adaln_init_scale)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(self.norm1(x) * (1 + scale_msa) + shift_msa)
        x = x + gate_mlp * self.mlp(self.norm2(x) * (1 + scale_mlp) + shift_mlp)
        return x


class _CTTransformer(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, output_dim, depth, heads, dim_head, mlp_ratio, dropout, adaln_init_scale
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.cond_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.output_proj = nn.Linear(hidden_dim, output_dim) if hidden_dim != output_dim else nn.Identity()
        self.layers = nn.ModuleList(
            [
                _CTConditionalBlock(hidden_dim, heads, dim_head, mlp_ratio, dropout, adaln_init_scale)
                for _ in range(depth)
            ]
        )

    def forward(self, x, c):
        x = self.input_proj(x)
        c = self.cond_proj(c)
        for block in self.layers:
            x = block(x, c)
        return self.output_proj(self.norm(x))


class CausalTransformerPredictor(nn.Module):
    def __init__(
        self,
        input_dim,
        action_dim,
        depth,
        heads,
        dim_head,
        mlp_ratio=4.0,
        max_seq_len=16,
        dropout=0.1,
        emb_dropout=0.0,
        predictor_dim=None,
        adaln_init_scale=0.02,
    ):
        super().__init__()
        self.action_embedder = nn.Sequential(
            nn.Linear(action_dim, input_dim), nn.SiLU(), nn.Linear(input_dim, input_dim)
        )
        self.pos_embedding = nn.Parameter(0.02 * torch.randn(1, max_seq_len, input_dim))
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.transformer = _CTTransformer(
            input_dim=input_dim,
            hidden_dim=predictor_dim or input_dim,
            output_dim=input_dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            adaln_init_scale=adaln_init_scale,
        )

    def forward(self, x, c):
        x = self.emb_dropout(x + self.pos_embedding[:, : x.size(1)])
        return self.transformer(x, self.action_embedder(c))
