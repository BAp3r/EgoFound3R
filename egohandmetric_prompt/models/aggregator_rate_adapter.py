from __future__ import annotations


import math
import torch
from torch import nn
from torch.nn import functional as F
from egohandmetric_prompt.multirate_contracts import (
    MultiRateFrameMap,
    gather_time_axis,
    gather_time_mask,
)


_MAX_BLOCK_SIZE = 5


def _slots(shape: tuple[int, ...]) -> int:
    return math.prod(shape) if shape else 1


def _expand_time(mask: torch.Tensor, tail: tuple[int, ...]) -> torch.Tensor:
    return mask.reshape(*mask.shape, *([1] * len(tail))).expand(*mask.shape, *tail)


def _require_float(name: str, value: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise ValueError(f"{name} must be a floating-point torch.Tensor")


def _require_high(
    name: str,
    value: torch.Tensor,
    frame_map: MultiRateFrameMap,
    channels: int,
) -> None:
    _require_float(name, value)
    if value.ndim < 3 or value.shape[-1] != channels:
        raise ValueError(f"{name} must have shape [B, T_H, ..., {channels}]")
    if value.shape[:2] != frame_map.high_frame_present.shape:
        raise ValueError(f"{name} must align with frame_map.high_frame_present")
    if value.device != frame_map.high_frame_present.device:
        raise ValueError(f"{name} and frame_map must be on the same device")


def _require_valid(name: str, valid: torch.Tensor, value: torch.Tensor) -> None:
    if not isinstance(valid, torch.Tensor) or valid.dtype is not torch.bool:
        raise ValueError(f"{name} must be a bool tensor")
    if valid.shape != value.shape[:-1] or valid.device != value.device:
        raise ValueError(f"{name} must match {tuple(value.shape[:-1])} on {value.device}")


def _missing_like(anchor: torch.Tensor, missing: torch.Tensor | None) -> torch.Tensor:
    if missing is None:
        return torch.zeros_like(anchor)
    if not isinstance(missing, torch.Tensor):
        raise ValueError("missing value must be a torch.Tensor")
    try:
        return torch.broadcast_to(missing.to(device=anchor.device, dtype=anchor.dtype), anchor.shape)
    except RuntimeError as error:
        raise ValueError("missing value must broadcast to the pooled output") from error


def interpolate_circular_phase_embedding(
    embedding: nn.Embedding,
    phase_position_5: torch.Tensor,
) -> torch.Tensor:
    if embedding.num_embeddings != _MAX_BLOCK_SIZE:
        raise ValueError("phase embedding table size does not match the block size contract")
    if not isinstance(phase_position_5, torch.Tensor) or not phase_position_5.is_floating_point():
        raise ValueError("phase_position_5 must be a floating-point tensor")
    if phase_position_5.device != embedding.weight.device:
        raise ValueError("phase positions and phase embedding must share a device")
    if not torch.isfinite(phase_position_5).all():
        raise ValueError("phase positions must be finite")
    wrapped = torch.remainder(phase_position_5, float(_MAX_BLOCK_SIZE))
    lower_float = torch.floor(wrapped)
    lower_index = lower_float.to(dtype=torch.long)
    lower = embedding(lower_index)
    fraction = (wrapped - lower_float).to(dtype=lower.dtype).unsqueeze(-1)
    if not bool(fraction.ne(0).any().item()):
        return lower
    upper_index = torch.remainder(lower_index + 1, _MAX_BLOCK_SIZE)
    upper = embedding(upper_index)
    interpolated = lower + fraction * (upper - lower)
    return torch.where(fraction.eq(0), lower, interpolated)


def _center_block(
    value: torch.Tensor,
    frame_map: MultiRateFrameMap,
    valid_high: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, high_frames = value.shape[:2]
    global_frames = frame_map.global_frame_present.shape[1]
    leading = tuple(value.shape[2:-1])
    slots = torch.arange(_MAX_BLOCK_SIZE, device=value.device, dtype=torch.int64)
    anchors = frame_map.global_anchor_indices.to(device=value.device, dtype=torch.int64)
    global_present = frame_map.global_frame_present.to(device=value.device, dtype=torch.bool)
    stride = frame_map.global_stride.to(device=value.device, dtype=torch.int64)
    anchor_phase = frame_map.anchor_phase.to(device=value.device, dtype=torch.int64)
    block_start = anchors - anchor_phase.unsqueeze(1)
    raw = block_start.unsqueeze(-1) + slots.reshape(1, 1, _MAX_BLOCK_SIZE)
    in_bounds = raw.ge(0) & raw.lt(high_frames)
    slot_valid = slots.reshape(1, 1, _MAX_BLOCK_SIZE) < stride.reshape(batch, 1, 1)
    gather_valid = global_present.unsqueeze(-1) & slot_valid & in_bounds
    indices = raw.clamp(0, high_frames - 1)
    flat_indices = indices.reshape(batch, global_frames * _MAX_BLOCK_SIZE)
    flat_valid = gather_valid.reshape(batch, global_frames * _MAX_BLOCK_SIZE)
    block = gather_time_axis(value, flat_indices, flat_valid).reshape(
        batch, global_frames, _MAX_BLOCK_SIZE, *value.shape[2:]
    )
    frame_valid = gather_time_mask(
        frame_map.high_frame_present.to(device=value.device, dtype=torch.bool), flat_indices, flat_valid
    ).reshape(batch, global_frames, _MAX_BLOCK_SIZE)
    block_valid = _expand_time(gather_valid & frame_valid, leading)
    anchor = gather_time_axis(value, anchors, global_present)
    anchor_valid = _expand_time(global_present, leading)
    if valid_high is not None:
        selected_valid = gather_time_mask(valid_high, flat_indices, flat_valid).reshape(
            batch, global_frames, _MAX_BLOCK_SIZE, *leading
        )
        block_valid = block_valid & selected_valid
        anchor_valid = anchor_valid & gather_time_mask(valid_high, anchors, global_present)
    phase_position = gather_time_axis(
        frame_map.phase_position_5.to(device=value.device, dtype=torch.float32),
        flat_indices,
        flat_valid,
    ).reshape(batch, global_frames, _MAX_BLOCK_SIZE)
    return anchor, anchor_valid, block, block_valid, phase_position


class AnchorCenteredResidualPooler(nn.Module):
    def __init__(
        self,
        *,
        value_dim: int,
        quality_dim: int = 0,
        phase_dim: int = 8,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if value_dim <= 0 or quality_dim < 0 or phase_dim <= 0:
            raise ValueError("value_dim and phase_dim must be positive; quality_dim must be non-negative")
        self.value_dim = int(value_dim)
        self.quality_dim = int(quality_dim)
        self.phase_dim = int(phase_dim)
        self.hidden_dim = int(hidden_dim) if hidden_dim is not None else max(self.value_dim, 16)
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        score_dim = self.value_dim + self.quality_dim + self.phase_dim
        self.phase_embedding = nn.Embedding(5, self.phase_dim)
        self.score = nn.Sequential(
            nn.LayerNorm(score_dim),
            nn.Linear(score_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.delta_norm = nn.LayerNorm(self.value_dim)
        self.anchor_norm = nn.LayerNorm(self.value_dim)
        self.residual = nn.Sequential(
            nn.Linear(self.value_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.value_dim),
        )
        self.residual_gate = nn.Parameter(torch.zeros(()))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(
        self,
        value_high: torch.Tensor,
        frame_map: MultiRateFrameMap,
        *,
        value_valid_high: torch.Tensor | None = None,
        quality_high: torch.Tensor | None = None,
        missing_value: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _require_high("value_high", value_high, frame_map, self.value_dim)
        if value_valid_high is not None:
            _require_valid("value_valid_high", value_valid_high, value_high)
        if quality_high is None:
            if self.quality_dim:
                quality_high = value_high.new_zeros(*value_high.shape[:-1], self.quality_dim)
        else:
            _require_float("quality_high", quality_high)
            if self.quality_dim == 0 or quality_high.shape != (*value_high.shape[:-1], self.quality_dim):
                raise ValueError("quality_high must align with value_high and configured quality_dim")
            if quality_high.device != value_high.device:
                raise ValueError("quality_high must be on the same device as value_high")
            quality_high = quality_high.detach().to(dtype=value_high.dtype)

        anchor, anchor_valid, block, block_valid, phase = _center_block(value_high, frame_map, value_valid_high)
        batch, global_frames = anchor.shape[:2]
        leading = tuple(anchor.shape[2:-1])
        slots = _slots(leading)
        rows = batch * global_frames * slots
        block_flat = block.permute(
            0, 1, *range(3, 3 + len(leading)), 2, block.ndim - 1
        ).reshape(rows, _MAX_BLOCK_SIZE, self.value_dim)
        valid_flat = block_valid.permute(0, 1, *range(3, 3 + len(leading)), 2).reshape(
            rows, _MAX_BLOCK_SIZE
        )
        anchor_flat = anchor.reshape(rows, self.value_dim)
        anchor_valid_flat = anchor_valid.reshape(rows)
        phase_flat = interpolate_circular_phase_embedding(
            self.phase_embedding,
            phase,
        ).to(dtype=value_high.dtype).unsqueeze(2).expand(
            batch,
            global_frames,
            slots,
            _MAX_BLOCK_SIZE,
            self.phase_dim,
        ).reshape(rows, _MAX_BLOCK_SIZE, self.phase_dim)
        if quality_high is None:
            quality_flat = None
        else:
            _, _, quality_block, _, _ = _center_block(quality_high, frame_map, value_valid_high)
            quality_flat = quality_block.permute(
                0, 1, *range(3, 3 + len(leading)), 2, quality_block.ndim - 1
            ).reshape(rows, _MAX_BLOCK_SIZE, self.quality_dim)

        has_valid = valid_flat.any(dim=-1)
        selected_rows = torch.nonzero(has_valid, as_tuple=False).flatten()
        selected_values = block_flat[selected_rows]
        score_input = [selected_values, phase_flat[selected_rows]]
        if quality_flat is not None:
            score_input.append(quality_flat[selected_rows])
        score = self.score(torch.cat(score_input, dim=-1)).squeeze(-1)
        score = score.masked_fill(~valid_flat[selected_rows], torch.finfo(score.dtype).min)
        attention = F.softmax(score, dim=-1)
        pooled = (attention.unsqueeze(-1) * selected_values).sum(dim=1)

        base = torch.where(
            anchor_valid_flat.unsqueeze(-1), anchor_flat, _missing_like(anchor, missing_value).reshape(rows, self.value_dim)
        )
        stride_one = frame_map.global_stride.eq(1).reshape(batch, 1, 1).expand(
            batch,
            global_frames,
            slots,
        ).reshape(rows)
        residual = self.residual(
            torch.cat(
                (
                    self.delta_norm(pooled - anchor_flat[selected_rows]),
                    self.anchor_norm(anchor_flat[selected_rows]),
                ),
                dim=-1,
            )
        )
        residual_scale = 2.0 * torch.sigmoid(self.residual_gate).to(dtype=value_high.dtype)
        candidate = base[selected_rows] + residual_scale * residual
        candidate = torch.where(
            stride_one[selected_rows].unsqueeze(-1),
            base[selected_rows],
            candidate,
        )
        return base.index_copy(0, selected_rows, candidate).reshape_as(anchor)


class HandPromptBlockPooler(AnchorCenteredResidualPooler):
    def __init__(self, *, prompt_dim: int, phase_dim: int = 8, hidden_dim: int | None = None) -> None:
        super().__init__(value_dim=prompt_dim, phase_dim=phase_dim, hidden_dim=hidden_dim)

    def forward(
        self,
        hand_prompt_high: torch.Tensor,
        frame_map: MultiRateFrameMap,
        *,
        hand_prompt_valid_high: torch.Tensor | None = None,
        missing_hand_prompt: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hand_prompt_high.ndim != 5 or hand_prompt_high.shape[2:4] != (2, 3):
            raise ValueError("hand_prompt_high must have shape [B, T_H, 2, 3, C]")
        return super().forward(
            hand_prompt_high,
            frame_map,
            value_valid_high=hand_prompt_valid_high,
            missing_value=missing_hand_prompt,
        )


class MetricFeatureBlockPooler(AnchorCenteredResidualPooler):
    def __init__(self, *, metric_feature_dim: int, phase_dim: int = 8, hidden_dim: int | None = None) -> None:
        super().__init__(value_dim=metric_feature_dim, phase_dim=phase_dim, hidden_dim=hidden_dim)

    def forward(
        self,
        metric_feature_high: torch.Tensor,
        frame_map: MultiRateFrameMap,
        *,
        metric_feature_valid_high: torch.Tensor | None = None,
        missing_metric_feature: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if metric_feature_high.ndim != 3:
            raise ValueError("metric_feature_high must have shape [B, T_H, C]")
        return super().forward(
            metric_feature_high,
            frame_map,
            value_valid_high=metric_feature_valid_high,
            missing_value=missing_metric_feature,
        )


def _query_layout(
    query: torch.Tensor, frame_map: MultiRateFrameMap, channels: int
) -> tuple[int, int, tuple[int, ...], int]:
    _require_high("query_high", query, frame_map, channels)
    batch, high = query.shape[:2]
    leading = tuple(query.shape[2:-1])
    return batch, high, leading, _slots(leading)


def _query_valid(
    query_valid_high: torch.Tensor | None, query: torch.Tensor, frame_map: MultiRateFrameMap
) -> torch.Tensor:
    batch, high, leading, slots = _query_layout(query, frame_map, query.shape[-1])
    present = frame_map.high_frame_present.to(device=query.device, dtype=torch.bool)
    if query_valid_high is None:
        return present.unsqueeze(-1).expand(batch, high, slots)
    if not isinstance(query_valid_high, torch.Tensor) or query_valid_high.dtype is not torch.bool:
        raise ValueError("query_valid_high must be a bool tensor")
    if query_valid_high.device != query.device:
        raise ValueError("query_valid_high must be on the same device as query_high")
    if query_valid_high.shape == (batch, high):
        return (present & query_valid_high).unsqueeze(-1).expand(batch, high, slots)
    if query_valid_high.shape != query.shape[:-1]:
        raise ValueError("query_valid_high must be [B, T_H] or match query_high without channels")
    return (query_valid_high & _expand_time(present, leading)).reshape(batch, high, slots)


def _endpoints(
    global_value: torch.Tensor,
    frame_map: MultiRateFrameMap,
    global_available: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, high = frame_map.high_frame_present.shape
    global_frames = global_value.shape[1]
    if global_value.shape[0] != batch or global_frames != frame_map.global_frame_present.shape[1]:
        raise ValueError("global value must align with frame_map global axis")
    if global_value.device != frame_map.high_frame_present.device:
        raise ValueError("global value and frame_map must be on the same device")
    if global_available.shape != (batch, global_frames) or global_available.dtype is not torch.bool:
        raise ValueError("global_available must be [B, T_G] bool")
    if global_frames == 0:
        zeros = global_value.new_zeros(batch, high, *global_value.shape[2:])
        valid = torch.zeros(batch, high, device=global_value.device, dtype=torch.bool)
        return zeros, zeros, valid, valid
    present = frame_map.high_frame_present.to(device=global_value.device, dtype=torch.bool)
    left_raw = frame_map.left_anchor.to(device=global_value.device, dtype=torch.int64)
    right_raw = frame_map.right_anchor.to(device=global_value.device, dtype=torch.int64)
    left_in = left_raw.ge(0) & left_raw.lt(global_frames)
    right_in = right_raw.ge(0) & right_raw.lt(global_frames)
    left_index = left_raw.clamp(0, global_frames - 1)
    right_index = right_raw.clamp(0, global_frames - 1)
    left_valid = (
        frame_map.left_anchor_valid.to(device=global_value.device, dtype=torch.bool)
        & left_in
        & present
        & global_available.gather(1, left_index)
    )
    right_valid = (
        frame_map.right_anchor_valid.to(device=global_value.device, dtype=torch.bool)
        & right_in
        & present
        & global_available.gather(1, right_index)
    )
    left = gather_time_axis(global_value, left_index, left_valid)
    right = gather_time_axis(global_value, right_index, right_valid)
    return left, right, left_valid, right_valid


class _OnDemandAnchorBridge(nn.Module):
    def __init__(
        self,
        *,
        query_dim: int,
        memory_dim: int,
        output_dim: int | None = None,
        bridge_dim: int | None = None,
        num_heads: int = 1,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        if query_dim <= 0 or memory_dim <= 0:
            raise ValueError("query_dim and memory_dim must be positive")
        self.query_dim = int(query_dim)
        self.memory_dim = int(memory_dim)
        self.output_dim = int(output_dim) if output_dim is not None else self.query_dim
        self.bridge_dim = int(bridge_dim) if bridge_dim is not None else max(self.query_dim, self.memory_dim, 16)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        if self.output_dim <= 0 or self.bridge_dim <= 0:
            raise ValueError("output_dim and bridge_dim must be positive")
        if self.num_heads <= 0 or self.num_layers <= 0:
            raise ValueError("num_heads and num_layers must be positive")
        if self.bridge_dim % self.num_heads != 0:
            raise ValueError("bridge_dim must be divisible by num_heads")
        self.head_dim = self.bridge_dim // self.num_heads
        self.query_projection: nn.Module = (
            nn.Identity() if self.output_dim == self.query_dim else nn.Linear(self.query_dim, self.output_dim)
        )
        self.query_attention = nn.Linear(self.query_dim, self.bridge_dim, bias=False)
        self.key_projection = nn.Linear(self.memory_dim, self.bridge_dim, bias=False)
        self.value_projection = nn.Linear(self.memory_dim, self.bridge_dim, bias=False)
        self.phase_embedding = nn.Embedding(5, self.bridge_dim)
        self.endpoint_role_embedding = nn.Embedding(2, self.bridge_dim)
        self.refinement_query_projections = nn.ModuleList(
            nn.Linear(self.bridge_dim, self.bridge_dim, bias=False)
            for _ in range(self.num_layers - 1)
        )
        self.refinement_key_projections = nn.ModuleList(
            nn.Linear(self.bridge_dim, self.bridge_dim, bias=False)
            for _ in range(self.num_layers - 1)
        )
        self.refinement_value_projections = nn.ModuleList(
            nn.Linear(self.bridge_dim, self.bridge_dim, bias=False)
            for _ in range(self.num_layers - 1)
        )
        self.refinement_norms = nn.ModuleList(
            nn.LayerNorm(self.bridge_dim) for _ in range(self.num_layers - 1)
        )
        self.residual_projection = nn.Sequential(nn.LayerNorm(self.bridge_dim), nn.Linear(self.bridge_dim, self.output_dim))
        self.bridge_gate = nn.Parameter(torch.zeros(()))
        self.missing_global_token = nn.Parameter(torch.randn(1, 1, 1, 1, self.bridge_dim) * 0.02)
        nn.init.zeros_(self.residual_projection[-1].weight)
        nn.init.zeros_(self.residual_projection[-1].bias)

    def _attend(
        self,
        *,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        visible: torch.Tensor,
    ) -> torch.Tensor:
        if query.ndim != 4 or key.ndim != 5 or value.shape != key.shape:
            raise ValueError("bridge attention tensors have incompatible ranks")
        if query.shape[:3] != key.shape[:3] or query.shape[-1] != self.bridge_dim:
            raise ValueError("bridge attention query does not align with memory")
        if key.shape[-1] != self.bridge_dim or visible.shape != key.shape[:-1]:
            raise ValueError("bridge attention memory validity does not align")
        query_heads = query.reshape(*query.shape[:-1], self.num_heads, self.head_dim)
        key_heads = key.reshape(*key.shape[:-1], self.num_heads, self.head_dim)
        value_heads = value.reshape(*value.shape[:-1], self.num_heads, self.head_dim)
        scores = torch.einsum("btshd,btsmhd->btshm", query_heads, key_heads)
        scores = scores / math.sqrt(float(self.head_dim))
        attention = F.softmax(
            scores.masked_fill(~visible.unsqueeze(-2), torch.finfo(scores.dtype).min),
            dim=-1,
        )
        context = torch.einsum("btshm,btsmhd->btshd", attention, value_heads)
        return context.reshape(*query.shape)

    def _fuse(
        self,
        *,
        query_high: torch.Tensor,
        selected_memory: torch.Tensor,
        selected_valid: torch.Tensor,
        frame_map: MultiRateFrameMap,
        query_valid_high: torch.Tensor | None,
    ) -> torch.Tensor:
        batch, high, leading, slots = _query_layout(query_high, frame_map, self.query_dim)
        if selected_memory.ndim != 5 or selected_memory.shape[:3] != (batch, high, slots):
            raise ValueError("selected memory must be [B, T_H, slots, M, C]")
        if selected_memory.shape[-1] != self.memory_dim:
            raise ValueError("selected memory has an incompatible channel count")
        if selected_valid.shape != selected_memory.shape[:-1]:
            raise ValueError("selected memory metadata must align with selected memory")
        if selected_memory.device != query_high.device:
            raise ValueError("selected memory and query must share a device")
        if selected_valid.dtype is not torch.bool or selected_valid.device != query_high.device:
            raise ValueError("selected memory validity must be bool on the query device")
        query = query_high.reshape(batch, high, slots, self.query_dim)
        active = _query_valid(query_valid_high, query_high, frame_map)
        query_state = self.query_attention(query)
        phase_state = interpolate_circular_phase_embedding(
            self.phase_embedding,
            frame_map.phase_position_5.to(device=query_high.device, dtype=torch.float32),
        )
        query_state = query_state + phase_state.to(dtype=query_high.dtype).unsqueeze(2)
        memory_rows = selected_memory.shape[-2]
        if memory_rows % 2 != 0:
            raise ValueError("selected memory must contain equal left/right endpoint groups")
        roles = torch.arange(memory_rows, device=query_high.device).ge(memory_rows // 2).to(dtype=torch.long)
        role_state = self.endpoint_role_embedding(roles).to(dtype=query_high.dtype).reshape(
            1, 1, 1, memory_rows, self.bridge_dim
        )
        key = self.key_projection(selected_memory) + role_state
        value = self.value_projection(selected_memory) + role_state
        missing = self.missing_global_token.to(device=query_high.device, dtype=query_high.dtype).expand(
            batch, high, slots, 1, self.bridge_dim
        )
        no_real = ~selected_valid.any(dim=-1, keepdim=True)
        key = torch.cat((key, missing), dim=-2)
        value = torch.cat((value, missing), dim=-2)
        visible = torch.cat((selected_valid, no_real), dim=-1)
        context = self._attend(
            query=query_state,
            key=key,
            value=value,
            visible=visible,
        )
        for query_projection, key_projection, value_projection, refinement_norm in zip(
            self.refinement_query_projections,
            self.refinement_key_projections,
            self.refinement_value_projections,
            self.refinement_norms,
            strict=True,
        ):
            context = context + self._attend(
                query=query_projection(refinement_norm(context)),
                key=key_projection(key),
                value=value_projection(value),
                visible=visible,
            )
        residual_scale = 2.0 * torch.sigmoid(self.bridge_gate).to(dtype=query_high.dtype)
        output = self.query_projection(query) + residual_scale * self.residual_projection(context)
        output = torch.where(active.unsqueeze(-1), output, torch.zeros_like(output))
        return output.reshape(batch, high, *leading, self.output_dim)


class HandGlobalTokenBridge(_OnDemandAnchorBridge):
    def __init__(
        self,
        *,
        query_dim: int,
        global_token_dim: int,
        output_dim: int | None = None,
        bridge_dim: int | None = None,
        num_heads: int = 1,
        num_layers: int = 1,
    ) -> None:
        super().__init__(
            query_dim=query_dim,
            memory_dim=global_token_dim,
            output_dim=output_dim,
            bridge_dim=bridge_dim,
            num_heads=num_heads,
            num_layers=num_layers,
        )

    def forward(
        self,
        query_high: torch.Tensor,
        global_patch_tokens: torch.Tensor,
        frame_map: MultiRateFrameMap,
        *,
        global_token_valid: torch.Tensor | None = None,
        query_valid_high: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, high, _, slots = _query_layout(query_high, frame_map, self.query_dim)
        _require_float("global_patch_tokens", global_patch_tokens)
        global_frames = frame_map.global_frame_present.shape[1]
        if global_patch_tokens.ndim != 4 or global_patch_tokens.shape[:2] != (batch, global_frames):
            raise ValueError("global_patch_tokens must be [B, T_G, N_patch, C]")
        if global_patch_tokens.shape[-1] != self.memory_dim or global_patch_tokens.device != query_high.device:
            raise ValueError("global_patch_tokens has an incompatible channel count or device")
        if global_token_valid is None:
            available = frame_map.global_frame_present.to(device=query_high.device, dtype=torch.bool)
        else:
            if global_token_valid.shape != (batch, global_frames) or global_token_valid.dtype is not torch.bool:
                raise ValueError("global_token_valid must be [B, T_G] bool")
            if global_token_valid.device != query_high.device:
                raise ValueError("global_token_valid must be on the query device")
            available = global_token_valid & frame_map.global_frame_present.to(dtype=torch.bool)
        left, right, left_valid, right_valid = _endpoints(
            global_patch_tokens, frame_map, available
        )
        patches = global_patch_tokens.shape[2]
        if len(query_high.shape[2:-1]) == 1 and query_high.shape[2] == patches:
            memory = torch.stack((left, right), dim=-2)
            valid = torch.stack((left_valid, right_valid), dim=-1).unsqueeze(2).expand(
                batch, high, patches, 2
            )
        else:
            memory = torch.stack((left, right), dim=2).reshape(batch, high, 2 * patches, self.memory_dim)
            valid = torch.stack((left_valid, right_valid), dim=-1).unsqueeze(-1).expand(
                batch, high, 2, patches
            ).reshape(batch, high, 2 * patches)
            memory = memory.unsqueeze(2).expand(batch, high, slots, 2 * patches, self.memory_dim)
            valid = valid.unsqueeze(2).expand(batch, high, slots, 2 * patches)
        return self._fuse(
            query_high=query_high,
            selected_memory=memory,
            selected_valid=valid,
            frame_map=frame_map,
            query_valid_high=query_valid_high,
        )


class _FinalPromptBridge(_OnDemandAnchorBridge):
    def __init__(
        self,
        *,
        query_dim: int,
        global_prompt_dim: int | None = None,
        output_dim: int | None = None,
        bridge_dim: int | None = None,
        num_heads: int = 1,
        num_layers: int = 1,
    ) -> None:
        super().__init__(
            query_dim=query_dim,
            memory_dim=query_dim if global_prompt_dim is None else global_prompt_dim,
            output_dim=output_dim,
            bridge_dim=bridge_dim,
            num_heads=num_heads,
            num_layers=num_layers,
        )

    def forward(
        self,
        query_high: torch.Tensor,
        global_prompts: torch.Tensor,
        frame_map: MultiRateFrameMap,
        *,
        global_prompt_valid: torch.Tensor | None = None,
        query_valid_high: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, high, leading, slots = _query_layout(query_high, frame_map, self.query_dim)
        _require_float("global_prompts", global_prompts)
        global_frames = frame_map.global_frame_present.shape[1]
        if global_prompts.ndim != query_high.ndim or global_prompts.shape[:2] != (batch, global_frames):
            raise ValueError("global_prompts must replace query_high's T_H axis by T_G")
        if tuple(global_prompts.shape[2:-1]) != leading or global_prompts.shape[-1] != self.memory_dim:
            raise ValueError("global_prompts must use matching prompt slots and configured channels")
        if global_prompts.device != query_high.device:
            raise ValueError("global_prompts must be on the query device")
        if global_prompt_valid is None:
            prompt_valid = _expand_time(
                frame_map.global_frame_present.to(device=query_high.device, dtype=torch.bool), leading
            )
        else:
            if not isinstance(global_prompt_valid, torch.Tensor) or global_prompt_valid.dtype is not torch.bool:
                raise ValueError("global_prompt_valid must be bool")
            if global_prompt_valid.device != query_high.device:
                raise ValueError("global_prompt_valid must be on the query device")
            if global_prompt_valid.shape == (batch, global_frames):
                prompt_valid = _expand_time(global_prompt_valid, leading)
            elif global_prompt_valid.shape == (batch, global_frames, *leading):
                prompt_valid = global_prompt_valid
            else:
                raise ValueError("global_prompt_valid must be [B, T_G] or match global prompts without channels")
            prompt_valid = prompt_valid & _expand_time(
                frame_map.global_frame_present.to(device=query_high.device, dtype=torch.bool), leading
            )
        available = frame_map.global_frame_present.to(device=query_high.device, dtype=torch.bool)
        left, right, left_valid, right_valid = _endpoints(global_prompts, frame_map, available)
        if global_frames == 0:
            left_slots = torch.zeros(batch, high, *leading, dtype=torch.bool, device=query_high.device)
            right_slots = left_slots
        else:
            left_index = frame_map.left_anchor.to(device=query_high.device, dtype=torch.int64).clamp(0, global_frames - 1)
            right_index = frame_map.right_anchor.to(device=query_high.device, dtype=torch.int64).clamp(0, global_frames - 1)
            left_slots = gather_time_mask(prompt_valid, left_index, left_valid)
            right_slots = gather_time_mask(prompt_valid, right_index, right_valid)
        memory = torch.stack(
            (
                left.reshape(batch, high, slots, self.memory_dim),
                right.reshape(batch, high, slots, self.memory_dim),
            ),
            dim=-2,
        )
        valid = torch.stack(
            (left_slots.reshape(batch, high, slots), right_slots.reshape(batch, high, slots)), dim=-1
        )
        return self._fuse(
            query_high=query_high,
            selected_memory=memory,
            selected_valid=valid,
            frame_map=frame_map,
            query_valid_high=query_valid_high,
        )


class FinalHandPromptBridge(_FinalPromptBridge):
    pass



class FinalMetricPromptBridge(_FinalPromptBridge):
    def forward(
        self,
        query_high: torch.Tensor,
        global_prompts: torch.Tensor,
        frame_map: MultiRateFrameMap,
        *,
        global_prompt_valid: torch.Tensor | None = None,
        query_valid_high: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if query_high.ndim == 3 and global_prompts.ndim == 4 and global_prompts.shape[2] == 1:
            return super().forward(
                query_high.unsqueeze(2),
                global_prompts,
                frame_map,
                global_prompt_valid=global_prompt_valid,
                query_valid_high=query_valid_high,
            ).squeeze(2)
        return super().forward(
            query_high,
            global_prompts,
            frame_map,
            global_prompt_valid=global_prompt_valid,
            query_valid_high=query_valid_high,
        )
