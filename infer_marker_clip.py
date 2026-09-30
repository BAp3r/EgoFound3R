from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import tee
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from egohandmetric_prompt.inference_hand_anchors import HandAnchorOptions, filter_hand_outputs
from egohandmetric_prompt.inference_hand_depth_scale import HandDepthScaleOptions, apply_hand_depth_scale
from egohandmetric_prompt.inference_intrinsics import RootIntrinsicsInput, root_solver_kwargs

from egohandmetric_prompt import marker_model_floating_dtype
from egohandmetric_prompt.inference_video_io import write_overlay_streams
from egohandmetric_prompt.inference_performance import ForwardMeasurement
from egohandmetric_prompt.models import (
    HAND_ROOT_SMOOTHING_MAX_STRENGTH,
    HAND_ROOT_SMOOTHING_MIN_STRENGTH,
    HAND_ROOT_SMOOTHING_MODE_CONFIDENCE_THREE_TAP,
    HAND_ROOT_SMOOTHING_MODE_NONE,
    HAND_ROOT_SMOOTHING_WINDOW_FRAMES,
    build_temporal_interpolation_query_map,
)
from infer_marker_video import (
    _as_public_multirate_outputs,
    _build_multirate_video_forward_contract,
    _call_marker_model,
    _marker_edges_from_config,
    _marker_faces_from_config,
    _project_points,
    _render_multimode_overlay_frames,
    _render_overlay_frames,
)


@dataclass(frozen=True, slots=True)
class ClipInferenceResult:
    outputs: dict[str, torch.Tensor]
    high_indices_local: torch.Tensor
    high_source_indices: torch.Tensor
    global_indices_local: torch.Tensor
    interval_steps: torch.Tensor
    full_frame_count: int
    forward_seconds: float
    peak_memory_bytes: int | None
    global_stride: int = 5
    global_anchor_phase: int = 2
    peak_reserved_memory_bytes: int | None = None
    hand_depth_scale: dict[str, Any] | None = None




def _nearest_global_intrinsics_full(
    global_intrinsics: torch.Tensor,
    *,
    global_high_indices: torch.Tensor,
    high_anchor_indices: torch.Tensor,
    output_present: torch.Tensor,
) -> torch.Tensor:
    batch_size, full_frames = output_present.shape
    global_frames = global_intrinsics.shape[1]
    if global_frames < 1 or tuple(global_intrinsics.shape[:2]) != (batch_size, global_frames):
        raise ValueError("global intrinsics must contain at least one valid G anchor")
    global_positions = high_anchor_indices.gather(1, global_high_indices)
    query_axis = torch.arange(full_frames, device=global_intrinsics.device).reshape(1, full_frames, 1)
    distance = (query_axis - global_positions.unsqueeze(1)).abs()
    nearest = distance.argmin(dim=2)
    gather_index = nearest[..., None, None].expand(batch_size, full_frames, 3, 3)
    result = global_intrinsics.gather(1, gather_index)
    return torch.where(output_present[..., None, None], result, torch.zeros_like(result))


def _cpu_temporal_outputs(outputs: dict[str, Any]) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for name, value in outputs.items():
        if isinstance(value, torch.Tensor) and value.ndim >= 1 and value.shape[0] == 1:
            result[name] = value[0].detach().cpu()
    return result


def run_clip_forward(
    *,
    project_config: Any,
    model: torch.nn.Module,
    frames: Sequence[torch.Tensor],
    high_indices_local: Sequence[int],
    source_frame_indices: Sequence[int] | None,
    fps: float,
    device: torch.device,
    hand_h_root_smoothing: bool = False,
    global_stride: int = 5,
    global_anchor_phase: int | None = None,
    root_intrinsics_input: RootIntrinsicsInput | None = None,
    hand_anchor_options: HandAnchorOptions | None = None,
    enable_hand_missing_fill: bool | None = None,
    hand_depth_scale_options: HandDepthScaleOptions | None = None,
) -> ClipInferenceResult:
    if enable_hand_missing_fill is not None:
        hand_anchor_options = (
            HandAnchorOptions(fill_missing=bool(enable_hand_missing_fill))
            if hand_anchor_options is None
            else replace(hand_anchor_options, fill_missing=bool(enable_hand_missing_fill))
        )
    if hand_anchor_options is not None:
        hand_anchor_options.validate()
        if hand_anchor_options.enabled and hand_h_root_smoothing:
            raise ValueError("Hand anchor filtering requires unsmoothed raw H inputs")
    full_frame_count = len(frames)
    if full_frame_count < 1:
        raise ValueError("clip must contain at least one R frame")
    high_indices = [int(index) for index in high_indices_local]
    if not high_indices or high_indices[0] != 0 or high_indices[-1] != full_frame_count - 1:
        raise ValueError("H indices must start at the first and end at the last R frame")
    if any(right <= left for left, right in zip(high_indices, high_indices[1:])):
        raise ValueError("H indices must be strictly increasing")
    if source_frame_indices is None:
        source_indices = list(range(full_frame_count))
    else:
        source_indices = [int(index) for index in source_frame_indices]
        if len(source_indices) != full_frame_count:
            raise ValueError("source_frame_indices must align with the R clip")
    high_source_indices = [source_indices[index] for index in high_indices]
    resolved_anchor_phase = (
        global_stride // 2
        if global_anchor_phase is None
        else int(global_anchor_phase)
    )
    images = torch.stack([frames[index] for index in high_indices], dim=0).unsqueeze(0).to(
        device=device,
        dtype=marker_model_floating_dtype(model),
    )
    forward_contract, frame_map = _build_multirate_video_forward_contract(
        project_config=project_config,
        images=images,
        window_indices=high_source_indices,
        fps=fps,
        global_stride=int(global_stride),
        global_anchor_phase=resolved_anchor_phase,
    )
    interval_steps = torch.diff(
        torch.as_tensor(high_indices, device=device, dtype=torch.long)
    ).reshape(1, -1)
    query_map = build_temporal_interpolation_query_map(
        frame_map,
        interpolation_factor=1,
        interval_steps=interval_steps,
    )
    if query_map.output_present.shape[1] != full_frame_count:
        raise RuntimeError("interpolation query map does not cover the complete R clip")
    forward_contract["temporal_interpolation_query_map"] = query_map
    forward_contract.update(root_solver_kwargs(root_intrinsics_input, high_source_indices, device))
    root_uv_smoothing = bool(
        hand_anchor_options is not None and hand_anchor_options.smooth_root_uv
    )
    forward_contract["root_uv_smoothing"] = root_uv_smoothing

    with torch.inference_mode():
        with ForwardMeasurement(device) as measurement:
            if hand_h_root_smoothing:
                raw_outputs = model(
                    images,
                    **forward_contract,
                    hand_h_root_smoothing=True,
                )
            else:
                raw_outputs = _call_marker_model(model, images, **forward_contract)
        outputs = _as_public_multirate_outputs(
            raw_outputs,
            global_frame_present=frame_map.global_frame_present,
        )
        outputs = filter_hand_outputs(outputs, frame_map, query_map, fps, hand_anchor_options)
        outputs, hand_depth_scale = apply_hand_depth_scale(outputs, frame_map, query_map, fps, hand_depth_scale_options)
    full_intrinsics = _nearest_global_intrinsics_full(
        outputs["intrinsics_global"],
        global_high_indices=frame_map.global_anchor_indices,
        high_anchor_indices=query_map.high_anchor_indices,
        output_present=query_map.output_present,
    )
    outputs = dict(outputs)
    outputs["intrinsics_full"] = full_intrinsics
    outputs["camera_pose_metric_full"] = outputs.get(
        "camera_pose_interpolated_full",
        outputs["camera_pose_refined_full"],
    )
    outputs["camera_pose_metric_valid_full"] = outputs.get(
        "camera_interpolation_valid_full",
        outputs["camera_refined_valid_full"],
    )
    outputs["depth_metric_global"] = outputs["depth_global"]
    outputs["depth_confidence_global"] = outputs["depth_conf_global"]
    outputs["intrinsics_K_global"] = outputs["intrinsics_global"]
    outputs["intrinsics_K_full"] = full_intrinsics
    outputs["hand_joint_xyz_metric_full"] = outputs.get(
        "dense_joint_xyz_interpolated_full",
        outputs["dense_joint_xyz_refined_full"],
    )
    outputs["hand_vertex_xyz_metric_full"] = outputs.get(
        "dense_vertex_xyz_interpolated_full",
        outputs["dense_vertex_xyz_refined_full"],
    )
    outputs["hand_metric_valid_full"] = outputs["hand_refined_valid_full"]
    global_present = frame_map.global_frame_present[0].to(dtype=torch.bool)
    global_high = frame_map.global_anchor_indices[0, global_present]
    global_local = query_map.high_anchor_indices[0].gather(0, global_high)
    return ClipInferenceResult(
        hand_depth_scale=hand_depth_scale,
        outputs=_cpu_temporal_outputs(outputs),
        high_indices_local=torch.as_tensor(high_indices, dtype=torch.long),
        high_source_indices=torch.as_tensor(high_source_indices, dtype=torch.long),
        global_indices_local=global_local.detach().cpu(),
        interval_steps=interval_steps[0].detach().cpu(),
        full_frame_count=full_frame_count,
        forward_seconds=measurement.seconds,
        peak_memory_bytes=measurement.peak_allocated_bytes,
        peak_reserved_memory_bytes=measurement.peak_reserved_bytes,
        global_stride=int(frame_map.global_stride[0].item()),
        global_anchor_phase=int(frame_map.anchor_phase[0].item()),
    )


def build_hand_h_root_smoothing_payload(
    result: ClipInferenceResult,
    *,
    enabled: bool,
) -> dict[str, Any]:
    outputs = result.outputs
    joint_xyz = outputs.get("dense_joint_xyz")
    valid = outputs.get(
        "interpolation_hand_anchor_valid_high",
        outputs.get("root_translation_valid"),
    )
    quality = outputs.get(
        "interpolation_hand_anchor_quality_high",
        outputs.get("root_translation_anchor_quality"),
    )
    if not isinstance(joint_xyz, torch.Tensor) or joint_xyz.ndim != 4:
        raise ValueError("H-rate dense_joint_xyz is required for Root smoothing diagnostics")
    expected = tuple(joint_xyz.shape[:2])
    if not isinstance(valid, torch.Tensor) or tuple(valid.shape) != expected:
        raise ValueError("root_translation_valid must align with H-rate [frame, side]")
    if not isinstance(quality, torch.Tensor) or tuple(quality.shape) != expected:
        raise ValueError("root_translation_anchor_quality must align with H-rate [frame, side]")
    root_raw = outputs.get("hand_root_xyz_raw_high", joint_xyz[..., 0, :])
    if not isinstance(root_raw, torch.Tensor) or tuple(root_raw.shape) != (*expected, 3):
        raise ValueError("raw H-rate Root diagnostics must have shape [T_H, 2, 3]")
    if enabled:
        root_smoothed = outputs.get("hand_root_xyz_smoothed_high")
        root_delta = outputs.get("hand_root_smoothing_delta_high")
        applied = outputs.get("hand_root_smoothing_applied_high")
        if (
            not isinstance(root_smoothed, torch.Tensor)
            or tuple(root_smoothed.shape) != tuple(root_raw.shape)
            or not isinstance(root_delta, torch.Tensor)
            or tuple(root_delta.shape) != tuple(root_raw.shape)
            or not isinstance(applied, torch.Tensor)
            or tuple(applied.shape) != expected
            or applied.dtype is not torch.bool
        ):
            raise ValueError("enabled H-rate Root smoothing diagnostics are incomplete")
    else:
        root_smoothed = root_raw
        root_delta = torch.zeros_like(root_raw)
        applied = torch.zeros(expected, dtype=torch.bool)
    payload = {
        "enabled": bool(enabled),
        "mode": (
            HAND_ROOT_SMOOTHING_MODE_CONFIDENCE_THREE_TAP
            if enabled
            else HAND_ROOT_SMOOTHING_MODE_NONE
        ),
        "window_frames_h": HAND_ROOT_SMOOTHING_WINDOW_FRAMES,
        "min_strength": HAND_ROOT_SMOOTHING_MIN_STRENGTH,
        "max_strength": HAND_ROOT_SMOOTHING_MAX_STRENGTH,
        "frame_indices": result.high_source_indices.clone(),
        "root_xyz_raw_metric": root_raw,
        "root_xyz_smoothed_metric": root_smoothed,
        "root_delta_metric": root_delta,
        "valid": valid.to(dtype=torch.bool),
        "anchor_quality": quality,
        "applied": applied,
    }
    for name in ("root_translation_reported_confidence", "root_translation_observation_reliability",
                 "root_depth_subset_spread_m", "root_depth_subset_relative_spread",
                 "root_depth_subset_valid_count", "root_depth_subset_valid"):
        value = outputs.get(name)
        if isinstance(value, torch.Tensor):
            if tuple(value.shape) != expected:
                raise ValueError(f"{name} must align with H observations")
            payload[name] = value
    return payload


def _write_optional_overlay(
    *,
    project_config: Any,
    payload: dict[str, Any],
    rgb_frames_uint8: Iterable[np.ndarray],
    output_dir: Path,
    fps: float,
    marker_count: int = 195,
    save_frame_images: bool = True,
    overlay_size_hw: tuple[int, int] | None = None,
) -> None:
    vertex = payload.get("vertex_xyz_display", payload.get("vertex_xyz"))
    joint = payload.get("joint_xyz_display", payload.get("joint_xyz"))
    intrinsics = payload.get("hand_projection_K_full", payload.get("intrinsics"))
    vertex_visibility = payload.get("vertex_visibility_logits")
    joint_visibility = payload.get("joint_visibility_logits")
    if not all(isinstance(value, torch.Tensor) for value in (vertex, joint, intrinsics, vertex_visibility, joint_visibility)):
        raise ValueError("full-rate hand/intrinsics outputs are required for overlay")
    if marker_count not in (195, 778):
        raise ValueError("overlay marker count must be 195 or 778")
    hand_778 = None
    if marker_count == 778:
        hand_778 = payload.get("hand_778")
        if not isinstance(hand_778, dict):
            raise ValueError("778 overlay requires the optional hand_778 output")
        vertex = hand_778["vertex_xyz_metric"]
        vertex_visibility = hand_778["vertex_visibility_logits"]
        marker_faces = hand_778["marker_faces"]
        marker_edges = hand_778["marker_edges"]
    else:
        marker_faces = _marker_faces_from_config(project_config, vertex_count=int(vertex.shape[2]))
        marker_edges = _marker_edges_from_config(project_config, vertex_count=int(vertex.shape[2]))
    marker_uv, marker_valid = _project_points(vertex, intrinsics)
    if hand_778 is not None:
        source_uv, source_valid = _project_points(payload["vertex_xyz"], intrinsics)
        source_ids = hand_778["source_vertex_ids"].to(device=marker_uv.device)
        marker_uv[..., source_ids, :] = source_uv.to(marker_uv)
        marker_valid[..., source_ids] = source_valid.to(marker_valid)
    joint_uv, joint_valid = _project_points(joint, intrinsics)
    if overlay_size_hw is not None:
        raster_hw = tuple(int(value) for value in payload["image_size_hw"])
        if tuple(overlay_size_hw) != raster_hw:
            scale = marker_uv.new_tensor(
                (
                    float(overlay_size_hw[1]) / float(raster_hw[1]),
                    float(overlay_size_hw[0]) / float(raster_hw[0]),
                )
            )
            marker_uv = marker_uv * scale
            joint_uv = joint_uv * scale
    calibration_valid = payload.get("hand_projection_K_valid_full")
    if isinstance(calibration_valid, torch.Tensor):
        marker_valid &= calibration_valid[:, None, None].to(device=marker_valid.device)
        joint_valid &= calibration_valid[:, None, None].to(device=joint_valid.device)
    hand_valid = payload.get(
        "hand_display_valid_full",
        payload.get("hand_metric_valid_full", payload.get("hand_refined_valid_full")),
    )
    if isinstance(hand_valid, torch.Tensor):
        expected_hand_valid_shape = tuple(vertex.shape[:2])
        if tuple(hand_valid.shape) != expected_hand_valid_shape:
            raise ValueError(
                "hand_refined_valid_full must align with full-rate [frame, side] Hand outputs"
            )
        hand_valid = hand_valid.to(device=marker_valid.device, dtype=torch.bool)
        marker_valid &= hand_valid.unsqueeze(-1)
        joint_valid &= hand_valid.unsqueeze(-1)
    rgb_iterator = iter(rgb_frames_uint8)
    rgb_streams = tee(rgb_iterator, 2)
    rendered = _render_overlay_frames(
        rgb_streams[0],
        marker_uv,
        marker_valid,
        vertex_visibility,
        joint_uv,
        joint_valid,
        joint_visibility,
        marker_edges,
        marker_faces,
        marker_depth=vertex[..., 2],
        marker_visibility_probability=(
            None if hand_778 is None else hand_778.get("vertex_visibility_probability")
        ),
        stream=True,
    )

    metric_predictions = payload.get("metric_predictions")
    hand_predictions = (
        metric_predictions.get("hand")
        if isinstance(metric_predictions, dict)
        and isinstance(metric_predictions.get("hand"), dict)
        else {}
    )

    def semantic_tensor(name: str) -> torch.Tensor | None:
        if hand_778 is not None and name.startswith("vertex_"):
            value = hand_778.get(name)
            return value if isinstance(value, torch.Tensor) else None
        value = hand_predictions.get(name)
        if not isinstance(value, torch.Tensor):
            value = payload.get(name)
        return value if isinstance(value, torch.Tensor) else None

    semantic_outputs = (
        semantic_tensor("vertex_contact_logits"),
        semantic_tensor("vertex_contact_distance_metric"),
        semantic_tensor("joint_contact_logits"),
        semantic_tensor("joint_contact_distance_metric"),
    )
    if not all(isinstance(value, torch.Tensor) for value in semantic_outputs):
        raise ValueError("contact class and metric distance outputs are required for multi-mode overlay")
    (
        vertex_contact,
        vertex_contact_distance,
        joint_contact,
        joint_contact_distance,
    ) = semantic_outputs
    multimode = _render_multimode_overlay_frames(
        rgb_streams[1],
        marker_uv,
        marker_valid,
        vertex_visibility,
        vertex_contact,
        vertex_contact_distance,
        joint_uv,
        joint_valid,
        joint_visibility,
        joint_contact,
        joint_contact_distance,
        marker_edges,
        marker_faces=marker_faces,
        marker_depth=vertex[..., 2],
        marker_visibility_probability=(
            None if hand_778 is None else hand_778.get("vertex_visibility_probability")
        ),
        marker_contact_probability=(
            None if hand_778 is None else hand_778.get("vertex_contact_probability")
        ),
        stream=True,
    )
    streams = [
        ({"overlay": image} for image in rendered),
        ({"overlay_multimode": image} for image in multimode),
    ]

    rows = ({key: value for group in row for key, value in group.items()}
            for row in zip(*streams, strict=True))
    try:
        write_overlay_streams(output_dir, rows, fps, save_frame_images=save_frame_images)
    finally:
        close = getattr(rgb_iterator, "close", None)
        if close is not None:
            close()
