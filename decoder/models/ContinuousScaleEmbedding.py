import torch
import torch.nn as nn
import math


class ContinuousScaleEmbedding(nn.Module):
    def __init__(self, feature_dim=384, scale_freq_bands=64):
        super(ContinuousScaleEmbedding, self).__init__()
        self.feature_dim = feature_dim
        self.scale_freq_bands = scale_freq_bands
        
        self.scale_freqs = nn.Parameter(torch.randn(scale_freq_bands) * 0.1)
        self.scale_phases = nn.Parameter(torch.randn(scale_freq_bands) * 0.1)
        
        self.scale_proj = nn.Linear(scale_freq_bands * 2, feature_dim)
        
        self._init_parameters()
    
    def _init_parameters(self):
        nn.init.normal_(self.scale_freqs, mean=0.0, std=0.1)
        nn.init.normal_(self.scale_phases, mean=0.0, std=0.1)
        
        nn.init.xavier_uniform_(self.scale_proj.weight)
        nn.init.zeros_(self.scale_proj.bias)
    
    def forward(self, scale_ratio):
        scale_normalized = scale_ratio * 2 * math.pi
        
        freqs = self.scale_freqs.unsqueeze(0) * scale_normalized.unsqueeze(1)  # (B, freq_bands)
        phases = self.scale_phases.unsqueeze(0)  # (B, freq_bands)
        
        sin_encodings = torch.sin(freqs + phases)  # (B, freq_bands)
        cos_encodings = torch.cos(freqs + phases)  # (B, freq_bands)
        
        scale_encoding = torch.cat([sin_encodings, cos_encodings], dim=-1)  # (B, freq_bands*2)
        
        scale_embedding = self.scale_proj(scale_encoding)  # (B, feature_dim)
        
        return scale_embedding


def make_continuous_scale_embedding(cfg):
    return ContinuousScaleEmbedding(
        feature_dim=cfg.enhancer.transformer.hidden_dim,
        scale_freq_bands=cfg.enhancer.get('scale_freq_bands', 64)
    )
