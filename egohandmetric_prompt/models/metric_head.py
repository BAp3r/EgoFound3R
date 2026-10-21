from __future__ import annotations


import math
import torch
from torch import nn


class MetricHead(nn.Module):
    def __init__(
        self,
        prompt_dim: int,
        hand_feature_dim: int | None = None,
        hidden_dim: int | None = None,
        metric_scale_min_value: float = 0.01,
        metric_scale_max_value: float = 100.0,
    ) -> None:
        super().__init__()
        min_value = float(metric_scale_min_value)
        max_value = float(metric_scale_max_value)
        if not math.isfinite(min_value) or min_value <= 0.0:
            raise ValueError("metric_scale_min_value must be finite and > 0.")
        if not math.isfinite(max_value) or max_value <= min_value:
            raise ValueError("metric_scale_max_value must be finite and > metric_scale_min_value.")
        if min_value < float(torch.finfo(torch.float32).tiny):
            raise ValueError("metric_scale_min_value must be representable as a positive float32 value.")
        if math.log(max_value) >= math.log(float(torch.finfo(torch.float32).max)):
            raise ValueError("metric_scale_max_value must remain finite after float32 log-space decoding.")
        self.prompt_dim = int(prompt_dim)
        self.hand_feature_dim = self.prompt_dim if hand_feature_dim is None else int(hand_feature_dim)
        self.hidden_dim = self.prompt_dim if hidden_dim is None else int(hidden_dim)
        self.metric_scale_min_value = min_value
        self.metric_scale_max_value = max_value
        self._metric_log_scale_min = math.log(min_value)
        self._metric_log_scale_max = math.log(max_value)
        self.hand_metric_fusion = nn.Sequential(
            nn.LayerNorm(self.hand_feature_dim * 2 + 2),
            nn.Linear(self.hand_feature_dim * 2 + 2, self.hand_feature_dim),
            nn.GELU(),
            nn.Linear(self.hand_feature_dim, self.hand_feature_dim),
        )
        self.metric_feature_fusion = nn.Sequential(
            nn.LayerNorm(self.hand_feature_dim * 2),
            nn.Linear(self.hand_feature_dim * 2, self.hand_feature_dim),
            nn.GELU(),
            nn.Linear(self.hand_feature_dim, self.hand_feature_dim),
        )
        self.metric_feature_proj = (
            nn.Identity()
            if self.hand_feature_dim == self.prompt_dim
            else nn.Sequential(
                nn.LayerNorm(self.hand_feature_dim),
                nn.Linear(self.hand_feature_dim, self.prompt_dim),
            )
        )
        self.frame_fuse = nn.Sequential(
            nn.LayerNorm(self.prompt_dim * 2),
            nn.Linear(self.prompt_dim * 2, self.hidden_dim),
            nn.GELU(),
        )
        self.scale_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def fuse_metric_feature(
        self,
        *,
        scene_metric_feature: torch.Tensor,
        hand_feature_set: torch.Tensor,
        in_view_probability: torch.Tensor,
    ) -> torch.Tensor:
        if scene_metric_feature.ndim != 3 or scene_metric_feature.shape[-1] != self.hand_feature_dim:
            raise ValueError("scene_metric_feature must have shape [B, T, C_hand]")
        batch, frames, _ = scene_metric_feature.shape
        if hand_feature_set.shape != (batch, frames, 2, 3, self.hand_feature_dim):
            raise ValueError("hand_feature_set must have shape [B, T, 2, 3, C_hand]")
        if in_view_probability.shape != (batch, frames, 2):
            raise ValueError("in_view_probability must have shape [B, T, 2]")
        side_features = hand_feature_set.detach().mean(dim=3)
        view = in_view_probability.detach().to(device=side_features.device, dtype=side_features.dtype)
        view = torch.nan_to_num(view, nan=0.5, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
        hand_metric_feature = self.hand_metric_fusion(
            torch.cat([side_features[..., 0, :], side_features[..., 1, :], view], dim=-1)
        )
        return self.metric_feature_fusion(torch.cat([scene_metric_feature, hand_metric_feature], dim=-1))

    def decode(
        self,
        final_metric_prompt: torch.Tensor,
        *,
        metric_feature: torch.Tensor,
        frame_present_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if final_metric_prompt.ndim == 4:
            if final_metric_prompt.shape[2] != 1:
                raise ValueError("final_metric_prompt with 4 dims must have one metric prompt")
            final_metric_prompt = final_metric_prompt.squeeze(2)
        if final_metric_prompt.ndim != 3:
            raise ValueError("final_metric_prompt must have shape [B, T, C] or [B, T, 1, C]")
        projected_feature = self.metric_feature_proj(metric_feature)
        if projected_feature.shape != final_metric_prompt.shape:
            raise ValueError("metric_feature must match final_metric_prompt after projection")
        if frame_present_mask.shape != final_metric_prompt.shape[:2]:
            raise ValueError("frame_present_mask must have shape [B, T]")
        fused = self.frame_fuse(torch.cat([final_metric_prompt, projected_feature], dim=-1))
        valid = frame_present_mask.to(device=fused.device, dtype=torch.bool)
        weights = valid.to(dtype=fused.dtype).unsqueeze(-1)
        pooled = (fused * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        raw_logit = self.scale_head(pooled).squeeze(-1)
        log_min = raw_logit.new_tensor(self._metric_log_scale_min, dtype=torch.float32)
        log_max = raw_logit.new_tensor(self._metric_log_scale_max, dtype=torch.float32)
        log_metric_value = log_min + (log_max - log_min) * torch.sigmoid(raw_logit.float())
        metric_min = raw_logit.new_tensor(self.metric_scale_min_value, dtype=torch.float32)
        metric_max = raw_logit.new_tensor(self.metric_scale_max_value, dtype=torch.float32)
        return {
            "log_metric_value": log_metric_value,
            "metric_value": torch.exp(log_metric_value).clamp(min=metric_min, max=metric_max),
        }

    def forward(
        self,
        final_metric_prompt: torch.Tensor,
        *,
        scene_metric_feature: torch.Tensor,
        hand_feature_set: torch.Tensor,
        in_view_probability: torch.Tensor,
        frame_present_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        metric_feature = self.fuse_metric_feature(
            scene_metric_feature=scene_metric_feature,
            hand_feature_set=hand_feature_set,
            in_view_probability=in_view_probability,
        )
        return self.decode(
            final_metric_prompt,
            metric_feature=metric_feature,
            frame_present_mask=frame_present_mask,
        )
