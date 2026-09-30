from __future__ import annotations


import torch
from torch import nn


def _stage_multiplier(value: float, *, name: str) -> float:
    resolved = float(value)
    if resolved not in (0.0, 1.0):
        raise ValueError(f"{name} must be exactly 0.0 or 1.0")
    return resolved


class HandViewFusionMLP(nn.Module):
    def __init__(self, *, prompt_dim: int, hand_feature_dim: int) -> None:
        super().__init__()
        if prompt_dim <= 0 or hand_feature_dim <= 0:
            raise ValueError("prompt_dim and hand_feature_dim must be positive")
        self.prompt_dim = int(prompt_dim)
        self.hand_feature_dim = int(hand_feature_dim)
        self.in_view_embedding = nn.Parameter(torch.randn(2, hand_feature_dim) * 0.02)
        self.keypoint_context_proj = nn.Sequential(
            nn.LayerNorm(hand_feature_dim),
            nn.Linear(hand_feature_dim, prompt_dim),
        )
        self.view_embedding_proj = nn.Sequential(
            nn.LayerNorm(hand_feature_dim),
            nn.Linear(hand_feature_dim, prompt_dim),
        )
        self.delta = nn.Sequential(
            nn.LayerNorm(prompt_dim * 3),
            nn.Linear(prompt_dim * 3, prompt_dim),
            nn.GELU(),
            nn.Linear(prompt_dim, prompt_dim),
        )
        self.gate_logit = nn.Parameter(torch.tensor(-2.0))

    def forward(
        self,
        *,
        base_hand_prompt: torch.Tensor,
        keypoint_context: torch.Tensor,
        in_view_probability: torch.Tensor,
        stage_multiplier: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if base_hand_prompt.ndim != 5 or base_hand_prompt.shape[-2:] != (3, self.prompt_dim):
            raise ValueError("base_hand_prompt must have shape [B, T, 2, 3, C]")
        if keypoint_context.shape != (*base_hand_prompt.shape[:4], self.hand_feature_dim):
            raise ValueError("keypoint_context must have shape [B, T, 2, 3, C_hand]")
        if in_view_probability.shape != base_hand_prompt.shape[:3]:
            raise ValueError("in_view_probability must have shape [B, T, 2]")
        multiplier = _stage_multiplier(stage_multiplier, name="stage_multiplier")
        probability = in_view_probability.detach().to(
            device=base_hand_prompt.device,
            dtype=base_hand_prompt.dtype,
        )
        probability = torch.nan_to_num(
            probability,
            nan=0.5,
            posinf=1.0,
            neginf=0.0,
        ).clamp_(0.0, 1.0)
        out_embedding, in_embedding = self.in_view_embedding.unbind(dim=0)
        view_embedding = out_embedding + probability.unsqueeze(-1) * (
            in_embedding - out_embedding
        )
        keypoint_prompt = self.keypoint_context_proj(
            keypoint_context.to(dtype=base_hand_prompt.dtype)
        )
        view_prompt = self.view_embedding_proj(view_embedding)
        repeated_view = view_prompt.unsqueeze(-2).expand_as(base_hand_prompt)
        delta = self.delta(
            torch.cat([base_hand_prompt, keypoint_prompt, repeated_view], dim=-1)
        )
        if multiplier == 0.0:
            return base_hand_prompt, view_embedding
        return (
            base_hand_prompt
            + multiplier * torch.sigmoid(self.gate_logit) * delta,
            view_embedding,
        )


class ViewMemoryProjector(nn.Module):
    def __init__(self, *, view_feature_dim: int, hand_feature_dim: int) -> None:
        super().__init__()
        self.view_feature_dim = int(view_feature_dim)
        self.hand_feature_dim = int(hand_feature_dim)
        self.in_view_embedding = nn.Parameter(torch.randn(2, view_feature_dim) * 0.02)
        self.project = nn.Sequential(
            nn.LayerNorm(view_feature_dim),
            nn.Linear(view_feature_dim, hand_feature_dim * 3),
        )
        self.gate_logit = nn.Parameter(torch.tensor(-2.0))

    def forward(
        self,
        in_view_probability: torch.Tensor,
        *,
        stage_multiplier: float = 1.0,
    ) -> torch.Tensor:
        if in_view_probability.ndim != 3 or in_view_probability.shape[-1] != 2:
            raise ValueError("in_view_probability must have shape [B, T, 2]")
        multiplier = _stage_multiplier(stage_multiplier, name="stage_multiplier")
        probability = in_view_probability.detach().to(dtype=self.in_view_embedding.dtype)
        probability = torch.nan_to_num(
            probability,
            nan=0.5,
            posinf=1.0,
            neginf=0.0,
        ).clamp_(0.0, 1.0)
        out_embedding, in_embedding = self.in_view_embedding.unbind(dim=0)
        embedding = out_embedding + probability.unsqueeze(-1) * (
            in_embedding - out_embedding
        )
        memory = self.project(embedding).reshape(
            *embedding.shape[:-1],
            3,
            self.hand_feature_dim,
        )
        return memory * (multiplier * torch.sigmoid(self.gate_logit))
