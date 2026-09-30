from __future__ import annotations


import math
import torch
from torch import nn
from torch.nn import functional as F


KEYPOINT_HEATMAP_UPSAMPLE_FACTOR = 2


KEYPOINT_LOCAL_WINDOW_RADIUS = 1


KEYPOINT_OFFSET_RANGE_CELLS = 1.5


class _EfficientSpatialResidual(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
        )
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.pointwise(F.gelu(self.depthwise(value)))


class _MaskedTemporalResidual(nn.Module):
    def __init__(self, channels: int, *, residual_scale_init: float = 0.1) -> None:
        super().__init__()
        self.depthwise_weight = nn.Parameter(torch.empty(channels, 3))
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))
        nn.init.kaiming_uniform_(self.depthwise_weight.unsqueeze(1), a=math.sqrt(5.0))
        nn.init.zeros_(self.pointwise.bias)

    def forward(
        self,
        value: torch.Tensor,
        frame_present_mask: torch.Tensor,
    ) -> torch.Tensor:
        if value.ndim != 5:
            raise ValueError("temporal feature must have shape [B, T, C, H, W]")
        if frame_present_mask.shape != value.shape[:2]:
            raise ValueError("frame_present_mask must align with temporal feature [B, T]")
        present = frame_present_mask.to(device=value.device, dtype=torch.bool)
        present_5d = present[:, :, None, None, None]
        temporal_input = value * present_5d.to(dtype=value.dtype)
        previous = F.pad(
            temporal_input[:, :-1],
            (0, 0, 0, 0, 0, 0, 1, 0),
        )
        following = F.pad(
            temporal_input[:, 1:],
            (0, 0, 0, 0, 0, 0, 0, 1),
        )
        weights = self.depthwise_weight.to(dtype=value.dtype)
        temporal_context = (
            previous * weights[None, None, :, 0, None, None]
            + temporal_input * weights[None, None, :, 1, None, None]
            + following * weights[None, None, :, 2, None, None]
        )
        batch, frames, channels, height, width = value.shape
        temporal_delta = self.pointwise(
            F.gelu(temporal_context.reshape(batch * frames, channels, height, width))
        ).reshape(batch, frames, channels, height, width)
        refined = value + self.residual_scale.to(dtype=value.dtype) * temporal_delta
        return torch.where(
            present_5d,
            refined,
            torch.zeros_like(refined),
        ).contiguous()


def decode_local_keypoint_cells(
    heatmap_logits: torch.Tensor,
    keypoint_offset: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if heatmap_logits.ndim != 6:
        raise ValueError("heatmap_logits must have shape [B, T, 2, J, H, W]")
    if keypoint_offset.shape != (*heatmap_logits.shape[:4], 2, *heatmap_logits.shape[-2:]):
        raise ValueError("keypoint_offset must have shape [B, T, 2, J, 2, H, W]")

    batch, frames, sides, joints, height, width = heatmap_logits.shape
    area = height * width
    safe_logits = torch.nan_to_num(
        heatmap_logits.float(),
        nan=-1.0e4,
        posinf=-1.0e4,
        neginf=-1.0e4,
    )
    flat_logits = safe_logits.reshape(batch, frames, sides, joints, area)
    peak_logits, cells = flat_logits.max(dim=-1)
    peak_confidence = peak_logits.sigmoid()
    peak_y = torch.div(cells, width, rounding_mode="floor")
    peak_x = cells.remainder(width)

    delta_y, delta_x = torch.meshgrid(
        torch.arange(
            -KEYPOINT_LOCAL_WINDOW_RADIUS,
            KEYPOINT_LOCAL_WINDOW_RADIUS + 1,
            device=cells.device,
        ),
        torch.arange(
            -KEYPOINT_LOCAL_WINDOW_RADIUS,
            KEYPOINT_LOCAL_WINDOW_RADIUS + 1,
            device=cells.device,
        ),
        indexing="ij",
    )
    delta_x = delta_x.reshape(-1)
    delta_y = delta_y.reshape(-1)
    neighbor_x = peak_x.unsqueeze(-1) + delta_x
    neighbor_y = peak_y.unsqueeze(-1) + delta_y
    in_bounds = (
        (neighbor_x >= 0)
        & (neighbor_x < width)
        & (neighbor_y >= 0)
        & (neighbor_y < height)
    )
    safe_x = neighbor_x.clamp(0, width - 1)
    safe_y = neighbor_y.clamp(0, height - 1)
    neighbor_indices = safe_y * width + safe_x

    flat_offsets = keypoint_offset.float().reshape(
        batch,
        frames,
        sides,
        joints,
        2,
        area,
    )
    offset_indices = neighbor_indices.unsqueeze(-2).expand(*neighbor_indices.shape[:-1], 2, -1)
    raw_offsets = flat_offsets.gather(-1, offset_indices)
    finite_offsets = torch.isfinite(raw_offsets).all(dim=-2)
    valid = in_bounds & finite_offsets
    has_valid = valid.any(dim=-1, keepdim=True)
    center_index = (delta_x == 0) & (delta_y == 0)
    fallback = center_index.view(*((1,) * (valid.ndim - 1)), -1).expand_as(valid)
    valid = torch.where(has_valid, valid, fallback)

    neighbor_logits = flat_logits.gather(-1, neighbor_indices)
    masked_logits = neighbor_logits.masked_fill(~valid, -1.0e4)
    weights = torch.softmax(masked_logits, dim=-1)
    weights = weights * valid.to(dtype=weights.dtype)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(weights.dtype).eps
    )

    offsets = KEYPOINT_OFFSET_RANGE_CELLS * torch.tanh(
        torch.nan_to_num(raw_offsets, nan=0.0, posinf=0.0, neginf=0.0)
    )
    offsets = offsets.movedim(-2, -1)
    candidate_xy = torch.stack((safe_x, safe_y), dim=-1).to(dtype=torch.float32) + offsets
    continuous_xy = (candidate_xy * weights.unsqueeze(-1)).sum(dim=-2)
    return continuous_xy, peak_logits, peak_confidence, cells


class HandKeypointLocalizationHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        joint_count: int = 21,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.joint_count = joint_count
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.heatmap_head = nn.Conv2d(hidden_channels, 2 * joint_count, kernel_size=1)
        self.refine_channels = max(hidden_channels // 4, 8)
        self.highres_reduce = nn.Conv2d(hidden_channels, self.refine_channels, kernel_size=1)
        self.temporal_refine = _MaskedTemporalResidual(
            self.refine_channels,
            residual_scale_init=0.1,
        )
        self.highres_refine = _EfficientSpatialResidual(self.refine_channels)
        self.side_refine = nn.ModuleList(
            [_EfficientSpatialResidual(self.refine_channels) for _ in range(2)]
        )
        self.side_heatmap_residual_heads = nn.ModuleList(
            [nn.Conv2d(self.refine_channels, joint_count, kernel_size=1) for _ in range(2)]
        )
        self.side_offset_heads = nn.ModuleList(
            [nn.Conv2d(self.refine_channels, 2 * joint_count, kernel_size=1) for _ in range(2)]
        )
        self.center_head = nn.Conv2d(self.refine_channels, 2, kernel_size=1)
        for head in (*self.side_heatmap_residual_heads, *self.side_offset_heads):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(
        self,
        feature_map: torch.Tensor,
        frame_present_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if feature_map.ndim != 5:
            raise ValueError("feature_map must have shape [B, T, C, H, W]")
        batch_size, num_frames, channels, height, width = feature_map.shape
        if frame_present_mask is None:
            frame_present = torch.ones(
                batch_size,
                num_frames,
                device=feature_map.device,
                dtype=torch.bool,
            )
        else:
            if frame_present_mask.shape != (batch_size, num_frames):
                raise ValueError("frame_present_mask must have shape [B, T]")
            frame_present = frame_present_mask.to(
                device=feature_map.device,
                dtype=torch.bool,
            )
        flattened = feature_map.reshape(batch_size * num_frames, channels, height, width)
        features = self.stem(flattened)
        coarse_logits_flat = self.heatmap_head(features)
        coarse_logits = coarse_logits_flat.reshape(
            batch_size,
            num_frames,
            2,
            self.joint_count,
            height,
            width,
        )
        coarse_logits_high = F.interpolate(
            coarse_logits_flat,
            scale_factor=KEYPOINT_HEATMAP_UPSAMPLE_FACTOR,
            mode="bilinear",
            align_corners=False,
        ).reshape(
            batch_size * num_frames,
            2,
            self.joint_count,
            height * KEYPOINT_HEATMAP_UPSAMPLE_FACTOR,
            width * KEYPOINT_HEATMAP_UPSAMPLE_FACTOR,
        )
        highres = F.gelu(self.highres_reduce(features)).reshape(
            batch_size,
            num_frames,
            self.refine_channels,
            height,
            width,
        )
        highres = self.temporal_refine(highres, frame_present).reshape(
            batch_size * num_frames,
            self.refine_channels,
            height,
            width,
        )
        highres = F.interpolate(
            highres,
            scale_factor=KEYPOINT_HEATMAP_UPSAMPLE_FACTOR,
            mode="bilinear",
            align_corners=False,
        )
        highres = self.highres_refine(highres)

        side_logits: list[torch.Tensor] = []
        side_offsets: list[torch.Tensor] = []
        for side_index in range(2):
            side_feature = self.side_refine[side_index](highres)
            residual = self.side_heatmap_residual_heads[side_index](side_feature)
            side_logits.append(coarse_logits_high[:, side_index] + residual)
            offset = self.side_offset_heads[side_index](side_feature).reshape(
                batch_size * num_frames,
                self.joint_count,
                2,
                highres.shape[-2],
                highres.shape[-1],
            )
            side_offsets.append(offset)

        logits = torch.stack(side_logits, dim=1).reshape(
            batch_size,
            num_frames,
            2,
            self.joint_count,
            highres.shape[-2],
            highres.shape[-1],
        )
        offsets = torch.stack(side_offsets, dim=1).reshape(
            batch_size,
            num_frames,
            2,
            self.joint_count,
            2,
            highres.shape[-2],
            highres.shape[-1],
        )
        center_logits = self.center_head(highres).reshape(
            batch_size,
            num_frames,
            2,
            highres.shape[-2],
            highres.shape[-1],
        )
        return {
            "keypoint_heatmap_logits": logits,
            "keypoint_offset": offsets,
            "keypoint_coarse_heatmap_logits": coarse_logits,
            "keypoint_center_logits": center_logits,
        }
