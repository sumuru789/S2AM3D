import torch
import torch.nn as nn
import math
from .ContinuousScaleEmbedding import ContinuousScaleEmbedding


def pos2posemb3d(pos, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_z = pos[..., 2, None] / dim_t

    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_z = torch.stack((pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_y, pos_x, pos_z), dim=-1)
    return posemb


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, 
                 attn_drop=0., proj_drop=0., drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, bias=qkv_bias)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(proj_drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(proj_drop)
        )
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        # Self-attention
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = residual + self.drop_path(x)

        # MLP
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + self.drop_path(x)
        return x


class PointFeatureEnhancer(nn.Module):
    def __init__(self, feature_dim=448, pos_embed_dim=384, feature_proj_dim=384,
                 pos_num_feats=128, temperature=10000, dropout=0.1,
                 transformer_num_layers=4, transformer_num_heads=8,
                 transformer_hidden_dim=384, transformer_mlp_ratio=4,
                 use_continuous_scale=True, scale_freq_bands=64, scale_dropout_rate=0.0):
        super(PointFeatureEnhancer, self).__init__()
        
        self.feature_proj = nn.Linear(feature_dim, feature_proj_dim)
        
        # Transformer Blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                dim=transformer_hidden_dim,
                num_heads=transformer_num_heads,
                mlp_ratio=transformer_mlp_ratio,
                qkv_bias=True,
                attn_drop=dropout,
                proj_drop=dropout,
                drop_path=dropout
            ) for _ in range(transformer_num_layers)
        ])
        
        self.pos_num_feats = pos_num_feats
        self.temperature = temperature
        
        self.use_continuous_scale = use_continuous_scale
        self.scale_dropout_rate = scale_dropout_rate
        if use_continuous_scale:
            self.scale_embedding = ContinuousScaleEmbedding(
                feature_dim=transformer_hidden_dim,
                scale_freq_bands=scale_freq_bands
            )
            self.scale_to_film = nn.Sequential(
                nn.LayerNorm(transformer_hidden_dim),
                nn.Linear(transformer_hidden_dim, transformer_hidden_dim * 2)
            )
            nn.init.zeros_(self.scale_to_film[1].weight)
            nn.init.zeros_(self.scale_to_film[1].bias)
            self.scale_gain = nn.Parameter(torch.tensor(0.1))
        else:
            self.scale_embedding = None
            self.scale_to_film = None
            self.register_parameter('scale_gain', None)
        
        self.output_layer = nn.Linear(transformer_hidden_dim, transformer_hidden_dim)

    def forward(self, point_feat, point_coords, point_color=None, continuous_scales=None):
        batch_size, num_points, _ = point_feat.shape
        
        point_feat = self.feature_proj(point_feat)  # (B, N, 384)
        
        pos_embed = pos2posemb3d(point_coords, self.pos_num_feats, self.temperature)  # (B, N, 384)
        
        x = point_feat + pos_embed  # (B, N, 384)
        
        if self.use_continuous_scale and self.scale_embedding is not None:
            if continuous_scales is not None:
                if self.training and torch.rand(1) < self.scale_dropout_rate:
                    zero_scales = torch.zeros_like(continuous_scales)
                    scale_emb = self.scale_embedding(zero_scales)
                    scale_emb = torch.where(torch.ones_like(scale_emb).bool(), torch.zeros_like(scale_emb), scale_emb)
                else:
                    scale_emb = self.scale_embedding(continuous_scales)
            else:
                zero_scales = torch.zeros(batch_size, device=x.device, dtype=x.dtype)
                scale_emb = self.scale_embedding(zero_scales)
                scale_emb = torch.where(torch.ones_like(scale_emb).bool(), torch.zeros_like(scale_emb), scale_emb)
            
            gamma_beta = self.scale_to_film(scale_emb)  # (B, 2D)
            gamma, beta = gamma_beta.chunk(2, dim=-1)   # (B, D), (B, D)
            x = x * (1.0 + self.scale_gain * gamma).unsqueeze(1) + (self.scale_gain * beta).unsqueeze(1)
        
        for block in self.transformer_blocks:
            if self.use_continuous_scale and self.scale_embedding is not None:
                gamma_beta = self.scale_to_film(scale_emb)  # (B, 2D)
                gamma, beta = gamma_beta.chunk(2, dim=-1)   # (B, D), (B, D)
                x = x * (1.0 + self.scale_gain * gamma).unsqueeze(1) + (self.scale_gain * beta).unsqueeze(1)
            x = block(x)
        
        enhanced_feats = self.output_layer(x)
        
        return enhanced_feats


def make(cfg):
    return PointFeatureEnhancer(
        feature_dim=cfg.enhancer.feature_dim,
        pos_embed_dim=cfg.enhancer.pos_embed_dim,
        feature_proj_dim=cfg.enhancer.feature_proj_dim,
        pos_num_feats=cfg.enhancer.pos_num_feats,
        temperature=cfg.enhancer.temperature,
        dropout=cfg.enhancer.dropout,
        transformer_num_layers=cfg.enhancer.transformer.num_layers,
        transformer_num_heads=cfg.enhancer.transformer.num_heads,
        transformer_hidden_dim=cfg.enhancer.transformer.hidden_dim,
        transformer_mlp_ratio=cfg.enhancer.transformer.mlp_ratio,
        use_continuous_scale=cfg.get('use_continuous_scale', True),
        scale_freq_bands=cfg.enhancer.get('scale_freq_bands', 64),
        scale_dropout_rate=cfg.get('scale_dropout_rate', 0.0)
    )
