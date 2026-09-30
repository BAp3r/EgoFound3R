from __future__ import annotations

from typing import Any
from typing import Mapping
import torch
from egohandmetric_prompt.models import HandStageGatePlan
from egohandmetric_prompt.models import MarkerModel
from egohandmetric_prompt.multirate_contracts import MultiRateFrameMap
from egohandmetric_prompt.configs import ProjectConfig
from egohandmetric_prompt.models import TemporalInterpolationQueryMap
from egohandmetric_prompt.configs import active_marker_model_config
from egohandmetric_prompt.multirate_contracts import build_multirate_frame_map
from egohandmetric_prompt.models import build_temporal_interpolation_query_map


INFERENCE_HAND_STAGE_INDEX = 4
INFERENCE_HAND_STAGE_GATE_PLAN = HandStageGatePlan(4, 1.0, 1.0)




def _resolve_forward_temporal_masks(
    *,
    batch: Any,
    images: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if images.ndim != 5:
        raise ValueError("images must have shape [B, T, 3, H, W]")
    batch_size, num_frames = images.shape[:2]
    raw_frame_present = batch.get("frame_present_mask") if isinstance(batch, dict) else None
    if raw_frame_present is None:
        frame_present_mask = torch.ones(
            batch_size,
            num_frames,
            device=images.device,
            dtype=torch.bool,
        )
    elif isinstance(raw_frame_present, torch.Tensor):
        frame_present_mask = raw_frame_present.to(device=images.device, dtype=torch.bool)
        if tuple(frame_present_mask.shape) != (batch_size, num_frames):
            raise ValueError("batch['frame_present_mask'] must have shape [B, T]")
    else:
        raise TypeError("batch['frame_present_mask'] must be a tensor when provided")

    expected_pair_shape = (batch_size, max(num_frames - 1, 0))
    raw_pair_valid = batch.get("temporal_pair_valid") if isinstance(batch, dict) else None
    if raw_pair_valid is None:
        temporal_pair_valid = frame_present_mask[:, :-1] & frame_present_mask[:, 1:]
    elif isinstance(raw_pair_valid, torch.Tensor):
        temporal_pair_valid = raw_pair_valid.to(device=images.device, dtype=torch.bool)
        if tuple(temporal_pair_valid.shape) != expected_pair_shape:
            raise ValueError("batch['temporal_pair_valid'] must have shape [B, T - 1]")
        temporal_pair_valid = temporal_pair_valid & frame_present_mask[:, :-1] & frame_present_mask[:, 1:]
    else:
        raise TypeError("batch['temporal_pair_valid'] must be a tensor when provided")
    return frame_present_mask, temporal_pair_valid


def _resolve_multirate_topology(
    *,
    project_config: Any,
    global_stride_override: int | None,
    global_anchor_phase_override: int | torch.Tensor | None,
) -> tuple[int | torch.Tensor, int | torch.Tensor]:
    marker_config = active_marker_model_config(project_config)
    if global_stride_override is not None:
        stride = int(global_stride_override)
        if not 1 <= stride <= int(marker_config.global_stride):
            raise ValueError(
                "global stride must lie inside the supported range "
                f"[1, {int(marker_config.global_stride)}]"
            )
    else:
        stride = int(marker_config.global_stride)
    if global_anchor_phase_override is not None:
        phase: int | torch.Tensor = global_anchor_phase_override
    else:
        phase = min(int(marker_config.global_anchor_eval_phase), stride - 1)
    return stride, phase


def _build_multirate_frame_map_from_batch(
    *,
    batch: Any,
    frame_present_mask: torch.Tensor,
    temporal_pair_valid: torch.Tensor,
    images: torch.Tensor,
    project_config: Any | None = None,
    global_stride_override: int | None = None,
    global_anchor_phase_override: int | torch.Tensor | None = None,
) -> MultiRateFrameMap:
    if not isinstance(batch, Mapping):
        raise TypeError("multi-rate forward requires a mapping batch")
    batch_size, high_frames = images.shape[:2]
    high_pair_valid = batch.get("high_temporal_pair_valid", temporal_pair_valid)
    if not isinstance(high_pair_valid, torch.Tensor):
        raise ValueError("multi-rate batch high_temporal_pair_valid must be a tensor when provided")
    if tuple(high_pair_valid.shape) != (batch_size, max(high_frames - 1, 0)):
        raise ValueError("multi-rate high_temporal_pair_valid must have shape [B, T_H - 1]")
    if project_config is None:
        global_stride = 5 if global_stride_override is None else global_stride_override
        global_anchor_phase = (
            min(2, global_stride - 1)
            if global_anchor_phase_override is None
            else global_anchor_phase_override
        )
    else:
        global_stride, global_anchor_phase = _resolve_multirate_topology(
            project_config=project_config,
            global_stride_override=global_stride_override,
            global_anchor_phase_override=global_anchor_phase_override,
        )
    return build_multirate_frame_map(
        high_frame_present=frame_present_mask,
        high_temporal_pair_valid=high_pair_valid.to(device=images.device, dtype=torch.bool),
        global_stride=global_stride,
        global_anchor_phase=global_anchor_phase,
    )


def _build_refiner_interval_steps_from_batch(
    *,
    batch: Mapping[str, Any],
    images: torch.Tensor,
    force_identity: bool = False,
) -> torch.Tensor:
    batch_size, high_frames = images.shape[:2]
    shape = (batch_size, max(high_frames - 1, 0))
    if force_identity:
        return torch.ones(shape, device=images.device, dtype=torch.long)
    explicit = batch.get("refiner_interval_steps")
    if explicit is not None:
        if not isinstance(explicit, torch.Tensor) or tuple(explicit.shape) != shape:
            raise ValueError("batch refiner_interval_steps must have shape [B, T_H - 1]")
        return explicit.to(device=images.device, dtype=torch.long)
    explicit_factor = batch.get("refiner_interpolation_factor")
    if explicit_factor is not None:
        factors = torch.as_tensor(explicit_factor, device=images.device, dtype=torch.long)
        if factors.ndim == 0:
            factors = factors.expand(batch_size)
        if tuple(factors.shape) != (batch_size,):
            raise ValueError("refiner_interpolation_factor must be scalar or shape [B]")
    else:
        sources = batch.get("batch_sources")
        if isinstance(sources, list) and len(sources) == batch_size:
            factors = torch.tensor(
                [int(source.get("temporal_stride", 1)) for source in sources],
                device=images.device,
                dtype=torch.long,
            )
        else:
            factors = torch.ones(batch_size, device=images.device, dtype=torch.long)
    if bool((factors < 1).any().item()):
        raise ValueError("refiner interpolation factors must be positive integers")
    return factors.unsqueeze(1).expand(shape).clone()


def _build_forward_temporal_contract(
    *,
    project_config: Any,
    batch: Any,
    images: torch.Tensor,
    frame_present_mask: torch.Tensor,
    temporal_pair_valid: torch.Tensor,
    force_identity_refiner: bool = False,
    global_stride_override: int | None = None,
    global_anchor_phase_override: int | torch.Tensor | None = None,
) -> dict[str, Any]:
    frame_map = _build_multirate_frame_map_from_batch(
        batch=batch,
        frame_present_mask=frame_present_mask,
        temporal_pair_valid=temporal_pair_valid,
        images=images,
        project_config=project_config,
        global_stride_override=global_stride_override,
        global_anchor_phase_override=global_anchor_phase_override,
    )
    return {
        "multirate_frame_map": frame_map,
        "temporal_interpolation_query_map": build_temporal_interpolation_query_map(
            frame_map,
            interval_steps=_build_refiner_interval_steps_from_batch(
                batch=batch,
                images=images,
                force_identity=force_identity_refiner,
            ),
        ),
    }


def _hand_stage_gate_plan(stage_index: int) -> HandStageGatePlan:
    if int(stage_index) != INFERENCE_HAND_STAGE_INDEX:
        raise ValueError("hand stage index must be in [1, 4]")
    return INFERENCE_HAND_STAGE_GATE_PLAN


def _build_prediction_marker_forward_contract(
    *,
    project_config: ProjectConfig,
    batch: Any,
    images: torch.Tensor,
    hand_stage_gate_plan: HandStageGatePlan | None = None,
    global_stride: int | None = None,
    global_anchor_phase: int | torch.Tensor | None = None,
) -> dict[str, Any]:
    frame_present_mask, temporal_pair_valid = _resolve_forward_temporal_masks(batch=batch, images=images)
    stage_plan = hand_stage_gate_plan or _hand_stage_gate_plan(4)
    stage_plan.validate()
    return {
        **_build_forward_temporal_contract(
            project_config=project_config,
            batch=batch,
            images=images,
            frame_present_mask=frame_present_mask,
            temporal_pair_valid=temporal_pair_valid,
            global_stride_override=global_stride,
            global_anchor_phase_override=global_anchor_phase,
        ),
        "hand_stage_gate_plan": stage_plan,
    }


def _call_marker_model(
    model: MarkerModel,
    images: torch.Tensor,
    *,
    hand_stage_gate_plan: HandStageGatePlan,
    multirate_frame_map: MultiRateFrameMap,
    temporal_interpolation_query_map: TemporalInterpolationQueryMap | None = None,
    root_solver_intrinsics: torch.Tensor | None = None,
    root_solver_intrinsics_valid: torch.Tensor | None = None,
    root_uv_smoothing: bool = False,
) -> Any:
    model_kwargs = {
        "multirate_frame_map": multirate_frame_map,
        "temporal_interpolation_query_map": temporal_interpolation_query_map,
        "hand_stage_gate_plan": hand_stage_gate_plan,
        "root_solver_intrinsics": root_solver_intrinsics,
        "root_solver_intrinsics_valid": root_solver_intrinsics_valid,
    }
    if root_uv_smoothing:
        model_kwargs["root_uv_smoothing"] = True
    return model(
        images,
        **model_kwargs,
    )
