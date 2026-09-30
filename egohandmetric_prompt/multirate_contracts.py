from __future__ import annotations


from dataclasses import dataclass
import torch
from torch import Tensor


PRODUCTION_GLOBAL_STRIDE = 5


PRODUCTION_GLOBAL_ANCHOR_PHASE = 2


MIN_GLOBAL_STRIDE = 1


MAX_GLOBAL_STRIDE = 5


@dataclass(frozen=True, slots=True)
class MultiRateFrameMap:
    high_frame_present: Tensor
    high_temporal_pair_valid: Tensor
    global_anchor_indices: Tensor
    global_frame_present: Tensor
    global_temporal_pair_valid: Tensor
    left_anchor: Tensor
    right_anchor: Tensor
    left_anchor_valid: Tensor
    right_anchor_valid: Tensor
    alpha_right: Tensor
    phase_in_block: Tensor
    global_stride: Tensor
    anchor_phase: Tensor

    @property
    def boundary_override(self) -> Tensor:
        high_frames = self.high_frame_present.shape[1]
        axis = torch.arange(
            high_frames,
            device=self.high_frame_present.device,
            dtype=torch.int64,
        ).reshape(1, high_frames)
        normal = self.high_frame_present & axis.remainder(
            self.global_stride.unsqueeze(1)
        ).eq(self.anchor_phase.unsqueeze(1))
        return self.high_frame_present.any(dim=1) & normal.sum(dim=1).eq(0)

    @property
    def canonical_fixed_topology(self) -> Tensor:
        return self.global_stride.eq(PRODUCTION_GLOBAL_STRIDE) & self.anchor_phase.eq(
            PRODUCTION_GLOBAL_ANCHOR_PHASE
        )

    @property
    def phase_position_5(self) -> Tensor:
        relative = torch.remainder(
            self.phase_in_block - self.anchor_phase.unsqueeze(1),
            self.global_stride.unsqueeze(1),
        ).to(dtype=torch.float32)
        position = torch.remainder(
            2.0 + 5.0 * relative / self.global_stride.unsqueeze(1).to(dtype=torch.float32),
            5.0,
        )
        return position.detach()

    def validate(self) -> None:
        validate_multirate_frame_map(self)


def _require_tensor(name: str, value: Tensor) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")


def _require_device(reference: Tensor, name: str, value: Tensor) -> None:
    if value.device != reference.device:
        raise ValueError(f"{name} must be on {reference.device}, got {value.device}")


def _require_shape(name: str, value: Tensor, shape: tuple[int, ...]) -> None:
    if tuple(value.shape) != shape:
        rendered = ", ".join(str(size) for size in shape)
        raise ValueError(f"{name} must have shape [{rendered}]")


def _require_bool(name: str, value: Tensor) -> None:
    if value.dtype is not torch.bool:
        raise ValueError(f"{name} must have dtype torch.bool")


def _require_int64(name: str, value: Tensor) -> None:
    if value.dtype is not torch.int64:
        raise ValueError(f"{name} must have dtype torch.int64")


def _require_float(name: str, value: Tensor) -> None:
    if value.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"{name} must have dtype torch.float32 or torch.float64")


def _any(value: Tensor) -> bool:
    return bool(value.any().item())


def _validate_detached(name: str, value: Tensor) -> None:
    if value.requires_grad:
        raise ValueError(f"{name} must be detached")


def _validate_high_inputs(
    *,
    high_frame_present: Tensor,
    high_temporal_pair_valid: Tensor,
) -> None:
    _require_tensor("high_frame_present", high_frame_present)
    _require_tensor("high_temporal_pair_valid", high_temporal_pair_valid)
    if high_frame_present.ndim != 2:
        raise ValueError("high_frame_present must have shape [B, T_H]")
    batch_size, high_frames = high_frame_present.shape
    if batch_size < 1 or high_frames < 1:
        raise ValueError("high timeline requires B >= 1 and T_H >= 1")
    _require_shape(
        "high_temporal_pair_valid",
        high_temporal_pair_valid,
        (batch_size, max(high_frames - 1, 0)),
    )
    _require_bool("high_frame_present", high_frame_present)
    _require_bool("high_temporal_pair_valid", high_temporal_pair_valid)
    _require_device(high_frame_present, "high_temporal_pair_valid", high_temporal_pair_valid)


def _effective_high_pair_valid(
    *,
    high_frame_present: Tensor,
    high_temporal_pair_valid: Tensor,
) -> Tensor:
    if high_frame_present.shape[1] <= 1:
        return high_temporal_pair_valid.detach()
    present_pair = high_frame_present[:, :-1] & high_frame_present[:, 1:]
    return (high_temporal_pair_valid & present_pair).detach()


def _anchor_layout(
    high_frame_present: Tensor,
    global_stride: Tensor,
    anchor_phase: Tensor,
) -> tuple[Tensor, Tensor]:
    batch_size, high_frames = high_frame_present.shape
    device = high_frame_present.device
    axis = torch.arange(high_frames, device=device, dtype=torch.int64).expand(batch_size, -1)
    normal = high_frame_present & axis.remainder(global_stride.unsqueeze(1)).eq(
        anchor_phase.unsqueeze(1)
    )
    normal_count = normal.sum(dim=1, dtype=torch.int64)
    boundary_override = high_frame_present.any(dim=1) & normal_count.eq(0)
    anchor_count = normal_count + boundary_override.to(dtype=torch.int64)
    global_frames = int(anchor_count.max().item())
    if global_frames == 0:
        return (
            torch.empty(batch_size, 0, device=device, dtype=torch.int64),
            torch.empty(batch_size, 0, device=device, dtype=torch.bool),
        )

    sentinel = torch.full((batch_size, high_frames), high_frames, device=device, dtype=torch.int64)
    normal_indices = torch.where(normal, axis, sentinel).sort(dim=1).values[:, :global_frames]
    global_slot = torch.arange(global_frames, device=device, dtype=torch.int64).unsqueeze(0)
    normal_present = global_slot < normal_count.unsqueeze(1)
    global_present = normal_present | (boundary_override.unsqueeze(1) & global_slot.eq(0))
    last_present = torch.where(high_frame_present, axis, torch.full_like(axis, -1)).amax(dim=1)
    anchor_indices = torch.where(global_present, normal_indices, torch.zeros_like(normal_indices))
    anchor_indices = torch.where(
        boundary_override.unsqueeze(1) & global_slot.eq(0),
        last_present.unsqueeze(1),
        anchor_indices,
    )
    return anchor_indices.detach(), global_present.detach()


def gather_time_axis(value: Tensor, indices: Tensor, valid: Tensor) -> Tensor:
    _require_tensor("value", value)
    _require_tensor("indices", indices)
    _require_tensor("valid", valid)
    if value.ndim < 2 or value.shape[1] < 1:
        raise ValueError("value must have a non-empty [B, T, ...] layout")
    if indices.ndim != 2:
        raise ValueError("indices must have shape [B, G]")
    _require_int64("indices", indices)
    _require_bool("valid", valid)
    if valid.shape != indices.shape:
        raise ValueError("valid must have the same shape as indices")
    if indices.shape[0] != value.shape[0]:
        raise ValueError("value and indices must have the same batch size")
    _require_device(value, "indices", indices)
    _require_device(value, "valid", valid)
    batch_size, global_frames = indices.shape
    safe = indices.clamp(min=0, max=value.shape[1] - 1)
    tail_shape = value.shape[2:]
    index_view = safe.reshape(batch_size, global_frames, *([1] * len(tail_shape)))
    gathered = value.gather(1, index_view.expand(batch_size, global_frames, *tail_shape))
    valid_view = valid.reshape(batch_size, global_frames, *([1] * len(tail_shape)))
    return torch.where(valid_view, gathered, torch.zeros_like(gathered))


def gather_time_mask(mask: Tensor, indices: Tensor, valid: Tensor) -> Tensor:
    gathered = gather_time_axis(mask, indices, valid)
    tail_shape = mask.shape[2:]
    valid_view = valid.reshape(valid.shape[0], valid.shape[1], *([1] * len(tail_shape)))
    return gathered.to(dtype=torch.bool) & valid_view


def _invalid_edge_prefix(high_temporal_pair_valid: Tensor) -> Tensor:
    batch_size = high_temporal_pair_valid.shape[0]
    return torch.cat(
        (
            torch.zeros(batch_size, 1, device=high_temporal_pair_valid.device, dtype=torch.int64),
            (~high_temporal_pair_valid).to(dtype=torch.int64).cumsum(dim=1),
        ),
        dim=1,
    )


def _chain_valid(
    *,
    invalid_edge_prefix: Tensor,
    first: Tensor,
    second: Tensor,
) -> Tensor:
    lower = torch.minimum(first, second)
    upper = torch.maximum(first, second)
    return invalid_edge_prefix.gather(1, upper).sub(invalid_edge_prefix.gather(1, lower)).eq(0)


def _global_pair_valid(
    *,
    high_temporal_pair_valid: Tensor,
    global_anchor_indices: Tensor,
    global_frame_present: Tensor,
) -> Tensor:
    batch_size, global_frames = global_anchor_indices.shape
    if global_frames < 2:
        return torch.empty(batch_size, 0, device=global_anchor_indices.device, dtype=torch.bool)
    invalid_prefix = _invalid_edge_prefix(high_temporal_pair_valid)
    left = global_anchor_indices[:, :-1]
    right = global_anchor_indices[:, 1:]
    return (
        global_frame_present[:, :-1]
        & global_frame_present[:, 1:]
        & (right > left)
        & _chain_valid(invalid_edge_prefix=invalid_prefix, first=left, second=right)
    ).detach()


def _query_geometry(
    *,
    high_frame_present: Tensor,
    high_temporal_pair_valid: Tensor,
    global_anchor_indices: Tensor,
    global_frame_present: Tensor,
    global_stride: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    batch_size, high_frames = high_frame_present.shape
    global_frames = global_anchor_indices.shape[1]
    device = high_frame_present.device
    zero_index = torch.zeros(batch_size, high_frames, device=device, dtype=torch.int64)
    zero_bool = torch.zeros(batch_size, high_frames, device=device, dtype=torch.bool)
    zero_alpha = torch.zeros(batch_size, high_frames, device=device, dtype=torch.float32)
    phase = torch.arange(
        high_frames,
        device=device,
        dtype=torch.int64,
    ).reshape(1, high_frames).remainder(global_stride.unsqueeze(1)).detach()
    if global_frames == 0:
        return zero_index, zero_index, zero_bool, zero_bool, zero_alpha, phase

    high_axis = torch.arange(high_frames, device=device, dtype=torch.int64).view(1, high_frames, 1)
    anchor_axis = global_anchor_indices.unsqueeze(1)
    query_present = high_frame_present.unsqueeze(-1)
    anchor_present = global_frame_present.unsqueeze(1)
    invalid_prefix = _invalid_edge_prefix(high_temporal_pair_valid)
    lower = torch.minimum(high_axis, anchor_axis)
    upper = torch.maximum(high_axis, anchor_axis)
    prefix_by_query = invalid_prefix.unsqueeze(1).expand(-1, high_frames, -1)
    chain_valid = torch.take_along_dim(prefix_by_query, upper, dim=2).sub(
        torch.take_along_dim(prefix_by_query, lower, dim=2)
    ).eq(0)
    common = query_present & anchor_present & chain_valid
    left_candidates = common & (anchor_axis <= high_axis)
    right_candidates = common & (anchor_axis >= high_axis)
    global_slot = torch.arange(global_frames, device=device, dtype=torch.int64).view(1, 1, global_frames)
    left_choice = torch.where(left_candidates, global_slot, torch.full_like(global_slot, -1)).amax(dim=-1)
    right_choice = torch.where(right_candidates, global_slot, torch.full_like(global_slot, global_frames)).amin(dim=-1)
    left_valid = left_choice.ge(0)
    right_valid = right_choice.lt(global_frames)
    left_anchor = torch.where(left_valid, left_choice, torch.zeros_like(left_choice))
    right_anchor = torch.where(right_valid, right_choice, torch.zeros_like(right_choice))

    left_position = gather_time_axis(global_anchor_indices, left_anchor, left_valid)
    right_position = gather_time_axis(global_anchor_indices, right_anchor, right_valid)
    same_anchor = left_valid & right_valid & left_anchor.eq(right_anchor)
    interpolation_valid = left_valid & right_valid & ~same_anchor & (right_position > left_position)
    high_position = torch.arange(high_frames, device=device, dtype=torch.float32).expand(batch_size, -1)
    alpha = torch.where(
        interpolation_valid,
        (high_position - left_position.to(dtype=torch.float32))
        / (right_position - left_position).to(dtype=torch.float32).clamp_min(1.0),
        zero_alpha,
    ).clamp(0.0, 1.0)
    return (
        left_anchor.detach(),
        right_anchor.detach(),
        left_valid.detach(),
        right_valid.detach(),
        alpha.detach(),
        phase,
    )


def _derived_fields(frame_map: MultiRateFrameMap) -> dict[str, Tensor]:
    anchor_indices, global_present = _anchor_layout(
        frame_map.high_frame_present,
        frame_map.global_stride,
        frame_map.anchor_phase,
    )
    global_pair_valid = _global_pair_valid(
        high_temporal_pair_valid=frame_map.high_temporal_pair_valid,
        global_anchor_indices=anchor_indices,
        global_frame_present=global_present,
    )
    values = _query_geometry(
        high_frame_present=frame_map.high_frame_present,
        high_temporal_pair_valid=frame_map.high_temporal_pair_valid,
        global_anchor_indices=anchor_indices,
        global_frame_present=global_present,
        global_stride=frame_map.global_stride,
    )
    return {
        "global_anchor_indices": anchor_indices,
        "global_frame_present": global_present,
        "global_temporal_pair_valid": global_pair_valid,
        "left_anchor": values[0],
        "right_anchor": values[1],
        "left_anchor_valid": values[2],
        "right_anchor_valid": values[3],
        "alpha_right": values[4],
        "phase_in_block": values[5],
        "global_stride": frame_map.global_stride,
        "anchor_phase": frame_map.anchor_phase,
    }


def validate_multirate_frame_map(frame_map: MultiRateFrameMap) -> None:
    if not isinstance(frame_map, MultiRateFrameMap):
        raise TypeError("frame_map must be a MultiRateFrameMap")
    _validate_high_inputs(
        high_frame_present=frame_map.high_frame_present,
        high_temporal_pair_valid=frame_map.high_temporal_pair_valid,
    )
    for name in ("high_frame_present", "high_temporal_pair_valid"):
        _validate_detached(name, getattr(frame_map, name))
    expected_high_pair_valid = _effective_high_pair_valid(
        high_frame_present=frame_map.high_frame_present,
        high_temporal_pair_valid=frame_map.high_temporal_pair_valid,
    )
    if not torch.equal(frame_map.high_temporal_pair_valid, expected_high_pair_valid):
        raise ValueError("high_temporal_pair_valid does not match the H-axis pair contract")

    batch_size, high_frames = frame_map.high_frame_present.shape
    if frame_map.global_anchor_indices.ndim != 2:
        raise ValueError("global_anchor_indices must have shape [B, T_G]")
    global_frames = frame_map.global_anchor_indices.shape[1]
    expected_shapes = {
        "global_anchor_indices": (batch_size, global_frames),
        "global_frame_present": (batch_size, global_frames),
        "global_temporal_pair_valid": (batch_size, max(global_frames - 1, 0)),
        "left_anchor": (batch_size, high_frames),
        "right_anchor": (batch_size, high_frames),
        "left_anchor_valid": (batch_size, high_frames),
        "right_anchor_valid": (batch_size, high_frames),
        "alpha_right": (batch_size, high_frames),
        "phase_in_block": (batch_size, high_frames),
        "global_stride": (batch_size,),
        "anchor_phase": (batch_size,),
    }
    for name, shape in expected_shapes.items():
        value = getattr(frame_map, name)
        _require_tensor(name, value)
        _require_shape(name, value, shape)
        _require_device(frame_map.high_frame_present, name, value)
        _validate_detached(name, value)
    for name in (
        "global_anchor_indices",
        "left_anchor",
        "right_anchor",
        "phase_in_block",
        "global_stride",
        "anchor_phase",
    ):
        _require_int64(name, getattr(frame_map, name))
    for name in ("global_frame_present", "global_temporal_pair_valid", "left_anchor_valid", "right_anchor_valid"):
        _require_bool(name, getattr(frame_map, name))
    _require_float("alpha_right", frame_map.alpha_right)
    if _any(~torch.isfinite(frame_map.alpha_right)):
        raise ValueError("alpha_right must be finite")
    if _any((frame_map.alpha_right < 0.0) | (frame_map.alpha_right > 1.0)):
        raise ValueError("alpha_right must stay in [0, 1]")
    if _any(
        (frame_map.global_stride < MIN_GLOBAL_STRIDE)
        | (frame_map.global_stride > MAX_GLOBAL_STRIDE)
    ):
        raise ValueError(
            f"global_stride must stay in [{MIN_GLOBAL_STRIDE}, {MAX_GLOBAL_STRIDE}]"
        )
    if _any(
        (frame_map.anchor_phase < 0)
        | (frame_map.anchor_phase >= frame_map.global_stride)
    ):
        raise ValueError("anchor_phase must satisfy 0 <= phase < global_stride")
    if _any(
        (frame_map.phase_in_block < 0)
        | (frame_map.phase_in_block >= frame_map.global_stride.unsqueeze(1))
    ):
        raise ValueError("phase_in_block must stay in [0, global_stride)")
    if _any(~torch.isfinite(frame_map.phase_position_5)):
        raise ValueError("phase_position_5 must be finite")

    for name, expected in _derived_fields(frame_map).items():
        if not torch.equal(getattr(frame_map, name), expected):
            raise ValueError(f"{name} does not match the production index-domain contract")


def build_multirate_frame_map(
    high_frame_present: Tensor | None = None,
    high_temporal_pair_valid: Tensor | None = None,
    global_stride: int | Tensor = PRODUCTION_GLOBAL_STRIDE,
    global_anchor_phase: int | Tensor = PRODUCTION_GLOBAL_ANCHOR_PHASE,
) -> MultiRateFrameMap:
    if high_frame_present is None or high_temporal_pair_valid is None:
        raise TypeError("build_multirate_frame_map requires high_frame_present and high_temporal_pair_valid")

    _validate_high_inputs(
        high_frame_present=high_frame_present,
        high_temporal_pair_valid=high_temporal_pair_valid,
    )

    batch_size = high_frame_present.shape[0]
    device = high_frame_present.device

    def topology_tensor(name: str, value: int | Tensor) -> Tensor:
        if type(value) is int:
            return torch.full(
                (batch_size,),
                value,
                device=device,
                dtype=torch.int64,
            )
        if not isinstance(value, Tensor):
            raise TypeError(f"{name} must be an int or torch.Tensor")
        _require_shape(name, value, (batch_size,))
        _require_int64(name, value)
        _require_device(high_frame_present, name, value)
        return value.detach()

    global_stride_tensor = topology_tensor("global_stride", global_stride)
    anchor_phase_tensor = topology_tensor("global_anchor_phase", global_anchor_phase)
    if _any(
        (global_stride_tensor < MIN_GLOBAL_STRIDE)
        | (global_stride_tensor > MAX_GLOBAL_STRIDE)
    ):
        raise ValueError(
            f"global_stride must stay in [{MIN_GLOBAL_STRIDE}, {MAX_GLOBAL_STRIDE}]"
        )
    if _any(
        (anchor_phase_tensor < 0)
        | (anchor_phase_tensor >= global_stride_tensor)
    ):
        raise ValueError("global_anchor_phase must satisfy 0 <= phase < global_stride")

    high_frame_present = high_frame_present.detach()
    high_temporal_pair_valid = _effective_high_pair_valid(
        high_frame_present=high_frame_present,
        high_temporal_pair_valid=high_temporal_pair_valid.detach(),
    )
    anchor_indices, global_present = _anchor_layout(
        high_frame_present,
        global_stride_tensor,
        anchor_phase_tensor,
    )
    global_pair_valid = _global_pair_valid(
        high_temporal_pair_valid=high_temporal_pair_valid,
        global_anchor_indices=anchor_indices,
        global_frame_present=global_present,
    )
    values = _query_geometry(
        high_frame_present=high_frame_present,
        high_temporal_pair_valid=high_temporal_pair_valid,
        global_anchor_indices=anchor_indices,
        global_frame_present=global_present,
        global_stride=global_stride_tensor,
    )
    frame_map = MultiRateFrameMap(
        high_frame_present=high_frame_present,
        high_temporal_pair_valid=high_temporal_pair_valid,
        global_anchor_indices=anchor_indices,
        global_frame_present=global_present,
        global_temporal_pair_valid=global_pair_valid,
        left_anchor=values[0],
        right_anchor=values[1],
        left_anchor_valid=values[2],
        right_anchor_valid=values[3],
        alpha_right=values[4],
        phase_in_block=values[5],
        global_stride=global_stride_tensor,
        anchor_phase=anchor_phase_tensor,
    )
    frame_map.validate()
    return frame_map
