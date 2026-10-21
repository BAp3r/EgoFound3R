from __future__ import annotations


import math
import torch
from torch import nn
from torch.nn import functional as F


class HandInViewHead(nn.Module):
    def __init__(self, *, dino_dim: int, hidden_dim: int) -> None:
        super().__init__()
        if dino_dim <= 0 or hidden_dim <= 0:
            raise ValueError("dino_dim and hidden_dim must be positive")
        self.dino_dim = int(dino_dim)
        self.hidden_dim = int(hidden_dim)
        groups = max(1, min(8, hidden_dim))
        while hidden_dim % groups != 0:
            groups -= 1
        self.feature_proj = nn.Sequential(
            nn.Conv2d(dino_dim, hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(groups, hidden_dim),
            nn.GELU(),
        )
        self.side_queries = nn.Parameter(torch.randn(2, hidden_dim) * 0.02)
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, dino_feature_map: torch.Tensor) -> torch.Tensor:
        if dino_feature_map.ndim != 5:
            raise ValueError("dino_feature_map must have shape [B, T, C, H, W]")
        batch_size, num_frames, channels, height, width = dino_feature_map.shape
        if channels != self.dino_dim:
            raise ValueError(f"expected DINO dim {self.dino_dim}, got {channels}")
        features = self.feature_proj(
            dino_feature_map.reshape(batch_size * num_frames, channels, height, width)
        )
        tokens = features.flatten(2).transpose(1, 2)
        normalized_tokens = self.token_norm(tokens)
        queries = self.query_norm(self.side_queries).unsqueeze(0).expand(batch_size * num_frames, -1, -1)
        attention_logits = torch.einsum("bsh,bnh->bsn", queries, normalized_tokens) / math.sqrt(self.hidden_dim)
        attention = F.softmax(attention_logits, dim=-1)
        summary = torch.einsum("bsn,bnh->bsh", attention, tokens)
        logits = self.classifier(torch.cat([summary, queries], dim=-1)).squeeze(-1)
        return logits.reshape(batch_size, num_frames, 2)
