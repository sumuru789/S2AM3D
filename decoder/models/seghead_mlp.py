import torch
import torch.nn as nn


class SegHead(nn.Module):
    def __init__(self, feature_dim=384, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, interacted_features):
        # interacted_features: (batch, 10000, 384)
        logits = self.mlp(interacted_features)
        probs = self.sigmoid(logits)

        return probs.squeeze(-1)

def make(cfg):

    return SegHead(cfg.seg_head.feature_dim, cfg.seg_head.dropout)
