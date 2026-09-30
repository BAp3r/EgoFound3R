from __future__ import annotations

from dataclasses import replace
import math
import time
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from egohandmetric_prompt.hand_upsampling import attach_mano778_output
from egohandmetric_prompt.inference_intrinsics import (
    RootIntrinsicsInput,
    attach_root_intrinsics,
    build_root_intrinsics_record,
    merge_root_intrinsics_records,
)
from egohandmetric_prompt.inference_root_z import RootZOptions, apply_root_z_postprocessing
from egohandmetric_prompt.inference_hand_anchors import (
    HandAnchorOptions,
    hand_anchor_filter_payload,
    _camera_space_smooth_fill,
    smooth_display_hand_local_shape,
    HAND_MISSING_FILL_MODE_CROSS_CHUNK,
)
from egohandmetric_prompt.inference_hand_depth_scale import HandDepthScaleOptions
from egohandmetric_prompt.inference_overlap_scale import (
    compact_depth_sample, estimate_overlap_depth_ratio, rebuild_chunk_scene, resolve_scale_graph,
)

from egohandmetric_prompt import marker_model_floating_dtype
from egohandmetric_prompt.inference_contracts import (
    attach_metric_prediction_contract,
    build_metric_prediction_contract,
    prepare_inference_output_dir,
    save_json_atomic,
)
from egohandmetric_prompt.metric_geometry import (
    unproject_depth_to_camera_points,
)
from egohandmetric_prompt.marker_runtime import save_marker_inference_output
from egohandmetric_prompt.inference_video_io import (
    WindowedVideoFrames,
    iter_video_rgb,
    resolve_full_frame_input_size,
)
from egohandmetric_prompt.inference_performance import build_inference_performance
from egohandmetric_prompt.models.camera_temporal_refiner import (
    _geometry_dtype,
    c2w_to_w2c,
    slerp_rotation_matrices,
    w2c_to_c2w,
)
from egohandmetric_prompt.temporal_rate_contract import (
    DEFAULT_CLIP_SECONDS,
    DEFAULT_OVERLAP_SECONDS,
    TemporalRateContract,
    build_overlapping_r_windows,
    build_video_global_high_axis,
    select_video_window_high_route,
)
from infer_marker_clip import (
    _write_optional_overlay,
    build_hand_h_root_smoothing_payload,
    run_clip_forward,
)

_CAMERA_POSE_FULL_KEYS = (
    "camera_pose_base_full",
    "camera_pose_interpolated_full",
    "camera_pose_refined_full",
)


OVERLAY_MAX_HEIGHT = 720


_HAND_SIDE_FULL_PREFIXES = ("dense_joint_", "dense_vertex_", "hand_")
_HAND_SIDE_FULL_NAMES = frozenset(
    {
        "root_translation_valid_full",
        "relative_hand_valid_full",
    }
)
_REBUILT_CANONICAL_FULL_KEYS = frozenset(
    {
        "camera_pose_metric_full",
        "camera_pose_metric_valid_full",
        "intrinsics_K_full",
        "hand_joint_xyz_metric_full",
        "hand_vertex_xyz_metric_full",
        "hand_metric_valid_full",
    }
)


def _scalar_tensor_value(
    outputs: dict[str, Any],
    *names: str,
    default: float = float("nan"),
) -> float:
    for name in names:
        value = outputs.get(name)
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            return float(value.item())
    return float(default)


def _is_per_side_hand_full_tensor(name: str, value: torch.Tensor) -> bool:
    return (
        value.ndim >= 2
        and value.shape[1] == 2
        and (
            name.startswith(_HAND_SIDE_FULL_PREFIXES)
            or name in _HAND_SIDE_FULL_NAMES
        )
    )


def _expand_side_mask(mask: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    return mask.reshape(*mask.shape, *([1] * (value.ndim - mask.ndim)))


def _apply_global_camera_space_hand_fill(
    stitched: dict[str, torch.Tensor],
    *,
    fill_missing: bool = True,
    smooth_local_shape: bool = True,
) -> None:
    joint = stitched.get("dense_joint_xyz_refined_full")
    vertex = stitched.get("dense_vertex_xyz_refined_full")
    trusted = stitched.get("hand_refined_valid_full")
    if not all(isinstance(value, torch.Tensor) for value in (joint, vertex, trusted)):
        return
    squeeze_batch = joint.ndim == 4
    if squeeze_batch:
        joint, vertex, trusted = joint.unsqueeze(0), vertex.unsqueeze(0), trusted.unsqueeze(0)
    if joint.ndim != 5 or vertex.ndim != 5 or joint.shape[:3] != trusted.shape:
        raise ValueError("stitched camera-space Hand tensors must align on [B,T,2]")
    trusted = trusted.to(dtype=torch.bool)

    def side_field(name: str, dtype: torch.dtype) -> torch.Tensor:
        value = stitched.get(name)
        if not isinstance(value, torch.Tensor):
            return torch.zeros_like(trusted, dtype=dtype)
        value = value.unsqueeze(0) if squeeze_batch else value
        if value.shape != trusted.shape:
            raise ValueError(f"{name} must align with stitched Hand validity")
        return value.to(device=trusted.device, dtype=dtype)

    previous_fill = side_field("hand_missing_fill_full", torch.bool)
    previous_mode = side_field("hand_missing_fill_mode_full", torch.long)
    available = trusted & ~previous_fill if fill_missing else trusted
    if fill_missing:
        joint, vertex, display_valid, computed_mode = _camera_space_smooth_fill(joint, vertex, available)
        previous_mode = torch.where(previous_fill & previous_mode.eq(0), computed_mode, previous_mode)
    else:
        display_valid = available
    new_fill = display_valid & ~available
    final_fill = previous_fill | new_fill
    final_mode = torch.where(
        new_fill & ~previous_fill,
        torch.full_like(previous_mode, HAND_MISSING_FILL_MODE_CROSS_CHUNK), previous_mode
    )
    final_mode = torch.where(final_fill, final_mode, torch.zeros_like(final_mode))
    applied = torch.zeros_like(trusted)
    if smooth_local_shape:
        joint, vertex, applied = smooth_display_hand_local_shape(
            joint, vertex, display_valid, side_field("hand_anchor_exact_mask_full", torch.bool)
        )

    for field, value in (("joint", joint), ("vertex", vertex)):
        value = value[0] if squeeze_batch else value
        stitched[f"dense_{field}_xyz_refined_full"] = value
        for name in (f"dense_{field}_xyz_interpolated_full", f"hand_{field}_xyz_metric_full"):
            if name in stitched:
                stitched[name] = value
        stitched[f"hand_display_{field}_xyz_full"] = value
    for name, value in (
        ("hand_display_valid_full", display_valid),
        ("hand_missing_fill_full", final_fill),
        ("hand_missing_fill_mode_full", final_mode),
        ("hand_display_local_smoothing_applied_full", applied),
    ):
        stitched[name] = value[0] if squeeze_batch else value




def estimate_world_alignment(
    reference_w2c: torch.Tensor,
    current_w2c: torch.Tensor,
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    if reference_w2c.shape != current_w2c.shape or reference_w2c.ndim != 3 or reference_w2c.shape[-2:] != (4, 4):
        raise ValueError("overlap camera poses must have identical [T, 4, 4] shape")
    if valid is None:
        valid = torch.ones(reference_w2c.shape[0], device=reference_w2c.device, dtype=torch.bool)
    else:
        valid = valid.to(device=reference_w2c.device, dtype=torch.bool).clone()
    valid &= torch.isfinite(reference_w2c).all(dim=(-2, -1))
    valid &= torch.isfinite(current_w2c).all(dim=(-2, -1))
    if not bool(valid.any().item()):
        raise ValueError("clip overlap contains no valid camera correspondence")
    reference_c2w = w2c_to_c2w(reference_w2c[valid].to(dtype=torch.float64))
    current_c2w = w2c_to_c2w(current_w2c[valid].to(dtype=torch.float64))
    reference_center = reference_c2w[..., :3, 3]
    current_center = current_c2w[..., :3, 3]

    rotations = reference_c2w[:, :3, :3] @ current_c2w[:, :3, :3].transpose(-1, -2)
    inliers = torch.ones(rotations.shape[0], device=rotations.device, dtype=torch.bool)
    for _ in range(3):
        u, _, vh = torch.linalg.svd(rotations[inliers].mean(dim=0))
        correction = torch.eye(3, device=u.device, dtype=u.dtype)
        correction[-1, -1] = torch.linalg.det(u @ vh)
        rotation = u @ correction @ vh
        angles = _rotation_angle_radians(rotations @ rotation.T)
        median = angles.median()
        threshold = (median + 3 * (angles - median).abs().median()).clamp_min(math.radians(5))
        inliers = angles <= threshold
    translations = reference_center - current_center @ rotation.T
    translation = torch.quantile(translations[inliers], .5, dim=0)
    alignment = torch.eye(4, device=reference_w2c.device, dtype=torch.float64)
    alignment[:3, :3] = rotation
    alignment[:3, 3] = translation
    return alignment.to(dtype=_geometry_dtype(reference_w2c.dtype))


def _rotation_angle_radians(rotation: torch.Tensor) -> torch.Tensor:
    cosine = (rotation.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) * .5
    return torch.acos(cosine.clamp(-1, 1))


def _overlap_camera_errors(
    reference_w2c: torch.Tensor,
    current_w2c: torch.Tensor,
    alignment: torch.Tensor,
    valid: torch.Tensor | None,
) -> torch.Tensor:
    if not bool(torch.isfinite(alignment).all()):
        return reference_w2c.new_full((2, 2), float("inf"), dtype=torch.float64)
    reference = w2c_to_c2w(reference_w2c.double())
    current = alignment.double() @ w2c_to_c2w(current_w2c.double())
    finite = torch.isfinite(reference).all(dim=(-2, -1)) & torch.isfinite(current).all(dim=(-2, -1))
    if valid is not None:
        finite &= valid.to(device=finite.device, dtype=torch.bool)
    reference, current = reference[finite], current[finite]
    position = (reference[:, :3, 3] - current[:, :3, 3]).norm(dim=-1)
    angle = _rotation_angle_radians(reference[:, :3, :3].transpose(-1, -2) @ current[:, :3, :3])
    return torch.quantile(torch.stack((position, angle), dim=-1), position.new_tensor([.5, .9]), dim=0)


def estimate_overlap_metric_scale(
    reference_w2c: torch.Tensor,
    current_w2c: torch.Tensor,
    valid: torch.Tensor | None = None,
    *,
    minimum_camera_displacement_m: float = 0.01,
    minimum_pair_count: int = 3,
    minimum_scale: float = 0.5,
    maximum_scale: float = 2.0,
) -> tuple[float, bool, int]:
    if reference_w2c.shape != current_w2c.shape or reference_w2c.ndim != 3:
        raise ValueError("overlap cameras must have identical [T, 4, 4] shape")
    if reference_w2c.shape[-2:] != (4, 4):
        raise ValueError("overlap cameras must contain 4x4 W2C poses")
    if valid is None:
        valid = torch.ones(reference_w2c.shape[0], dtype=torch.bool)
    valid = valid.detach().cpu().to(dtype=torch.bool)
    valid &= torch.isfinite(reference_w2c.detach().cpu()).all(dim=(-2, -1))
    valid &= torch.isfinite(current_w2c.detach().cpu()).all(dim=(-2, -1))
    if int(valid.sum().item()) < 3:
        return 1.0, False, 0
    reference_center = w2c_to_c2w(reference_w2c[valid].double())[..., :3, 3]
    current_center = w2c_to_c2w(current_w2c[valid].double())[..., :3, 3]
    reference_distance = torch.pdist(reference_center)
    current_distance = torch.pdist(current_center)
    usable = (
        torch.isfinite(reference_distance)
        & torch.isfinite(current_distance)
        & (reference_distance >= float(minimum_camera_displacement_m))
        & (current_distance >= float(minimum_camera_displacement_m))
    )
    pair_count = int(usable.sum().item())
    if pair_count < int(minimum_pair_count):
        return 1.0, False, pair_count
    log_ratio = torch.log(reference_distance[usable] / current_distance[usable])
    median_log_ratio = torch.median(log_ratio)
    median_absolute_deviation = torch.median((log_ratio - median_log_ratio).abs())
    scale = float(torch.exp(median_log_ratio).item())
    reliable = (
        math.isfinite(scale)
        and float(minimum_scale) <= scale <= float(maximum_scale)
        and float(median_absolute_deviation.item()) <= math.log(1.5)
    )
    return (scale if reliable else 1.0), reliable, pair_count


def scale_w2c_world_centers(camera_w2c: torch.Tensor, scale: float) -> torch.Tensor:
    if not math.isfinite(float(scale)) or float(scale) <= 0.0:
        raise ValueError("camera world scale must be finite and positive")
    c2w = w2c_to_c2w(camera_w2c.to(dtype=torch.float64))
    c2w = c2w.clone()
    c2w[..., :3, 3] *= float(scale)
    return c2w_to_w2c(c2w).to(dtype=_geometry_dtype(camera_w2c.dtype))


def apply_world_alignment_to_w2c(
    camera_w2c: torch.Tensor,
    alignment_current_to_reference: torch.Tensor,
) -> torch.Tensor:
    inverse_alignment = torch.linalg.inv(
        alignment_current_to_reference.to(device=camera_w2c.device, dtype=torch.float64)
    )
    aligned = camera_w2c.to(dtype=torch.float64) @ inverse_alignment
    return aligned.to(dtype=_geometry_dtype(camera_w2c.dtype))


def apply_world_alignment_to_points(
    points: torch.Tensor,
    alignment_current_to_reference: torch.Tensor,
) -> torch.Tensor:
    alignment = alignment_current_to_reference.to(device=points.device, dtype=torch.float64)
    transformed = torch.matmul(
        points.to(dtype=torch.float64),
        alignment[:3, :3].transpose(0, 1),
    ) + alignment[:3, 3]
    return transformed.to(dtype=points.dtype)


def symmetric_chamfer_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.ndim != 2 or right.ndim != 2 or left.shape[-1] != 3 or right.shape[-1] != 3:
        raise ValueError("Chamfer inputs must have shape [N, 3] and [M, 3]")
    if left.shape[0] == 0 or right.shape[0] == 0:
        raise ValueError("Chamfer inputs cannot be empty")
    distances = torch.cdist(left.float(), right.float())
    return 0.5 * (distances.amin(dim=1).mean() + distances.amin(dim=0).mean())


def _deterministic_point_subsample(points: torch.Tensor, maximum: int) -> torch.Tensor:
    finite = points[torch.isfinite(points).all(dim=-1)]
    if finite.shape[0] <= maximum:
        return finite
    indices = torch.linspace(0, finite.shape[0] - 1, maximum, device=finite.device).round().long()
    return finite.index_select(0, indices)


def _rotation_vector_matrix(rotation_vector: torch.Tensor) -> torch.Tensor:
    x, y, z = rotation_vector.unbind()
    zero = x.new_zeros(())
    skew = torch.stack(
        (
            torch.stack((zero, -z, y)),
            torch.stack((z, zero, -x)),
            torch.stack((-y, x, zero)),
        )
    )
    return torch.matrix_exp(skew)


def refine_world_alignment_chamfer(
    reference_points: torch.Tensor,
    current_points: torch.Tensor,
    initial_alignment: torch.Tensor,
    *,
    iterations: int = 30,
    maximum_points: int = 512,
) -> tuple[torch.Tensor, float, float]:
    reference = _deterministic_point_subsample(reference_points, maximum_points).float()
    current = _deterministic_point_subsample(current_points, maximum_points).float()
    aligned = apply_world_alignment_to_points(current, initial_alignment).float()
    if reference.shape[0] == 0 or current.shape[0] == 0:
        return initial_alignment, 0.0, 0.0
    before = float(symmetric_chamfer_distance(reference, aligned).item())
    if reference.shape[0] < 16 or current.shape[0] < 16 or iterations <= 0:
        return initial_alignment, before, before
    rotation_vector = torch.zeros(3, device=aligned.device, requires_grad=True)
    translation = torch.zeros(3, device=aligned.device, requires_grad=True)
    optimizer = torch.optim.Adam((rotation_vector, translation), lr=0.02)
    for _ in range(int(iterations)):
        optimizer.zero_grad(set_to_none=True)
        delta_rotation = _rotation_vector_matrix(rotation_vector)
        refined_points = aligned @ delta_rotation.transpose(0, 1) + translation
        chamfer = symmetric_chamfer_distance(reference, refined_points)
        regularization = 0.01 * rotation_vector.square().sum() + 0.01 * translation.square().sum()
        (chamfer + regularization).backward()
        optimizer.step()
        with torch.no_grad():
            rotation_norm = torch.linalg.vector_norm(rotation_vector).clamp_min(1e-8)
            rotation_vector.mul_(min(1.0, 0.10 / float(rotation_norm.item())))
            translation.clamp_(-0.10, 0.10)
    with torch.no_grad():
        delta = torch.eye(4, device=aligned.device, dtype=torch.float32)
        delta[:3, :3] = _rotation_vector_matrix(rotation_vector)
        delta[:3, 3] = translation
        refined_alignment = delta @ initial_alignment.to(device=aligned.device, dtype=torch.float32)
        after = float(
            symmetric_chamfer_distance(
                reference,
                apply_world_alignment_to_points(current, refined_alignment),
            ).item()
        )
    if not math.isfinite(after) or after > before:
        return initial_alignment, before, before
    return refined_alignment.to(dtype=initial_alignment.dtype), before, after


def _smooth_overlap_alpha(position: int, count: int) -> float:
    linear = float(position + 1) / float(count + 1)
    return 0.5 - 0.5 * math.cos(math.pi * linear)


def _blend_root_relative_geometry(
    reference_joint: torch.Tensor,
    current_joint: torch.Tensor,
    reference_vertex: torch.Tensor | None,
    current_vertex: torch.Tensor | None,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    reference_root = reference_joint[..., :1, :]
    current_root = current_joint[..., :1, :]
    root = (1.0 - alpha) * reference_root + alpha * current_root
    joint_local = (1.0 - alpha) * (reference_joint - reference_root) + alpha * (
        current_joint - current_root
    )
    joint = root + joint_local
    if reference_vertex is None or current_vertex is None:
        return joint, None
    vertex_local = (1.0 - alpha) * (reference_vertex - reference_root) + alpha * (
        current_vertex - current_root
    )
    return joint, root + vertex_local


def build_alignment_scene_points(
    outputs: dict[str, Any],
    global_indices_local: torch.Tensor,
    *,
    maximum_points_per_frame: int = 256,
) -> dict[int, torch.Tensor]:
    depth = outputs.get("depth_global")
    intrinsics = outputs.get("intrinsics_global")
    camera = outputs.get("camera_pose_global")
    camera_full = outputs.get("camera_pose_refined_full")
    confidence = outputs.get("depth_conf_global")
    if not all(isinstance(value, torch.Tensor) for value in (depth, intrinsics, camera)):
        return {}
    if depth.ndim != 3 or intrinsics.shape != (depth.shape[0], 3, 3):
        return {}
    if camera.shape != (depth.shape[0], 4, 4) or global_indices_local.numel() != depth.shape[0]:
        return {}
    if (
        isinstance(camera_full, torch.Tensor)
        and camera_full.ndim == 3
        and camera_full.shape[-2:] == (4, 4)
        and bool((global_indices_local < camera_full.shape[0]).all().item())
    ):
        camera = camera_full.index_select(0, global_indices_local.to(dtype=torch.long))
    points_camera = unproject_depth_to_camera_points(
        depth.unsqueeze(0).float(),
        intrinsics.unsqueeze(0).float(),
    )[0]
    result: dict[int, torch.Tensor] = {}
    for slot, local_r_index in enumerate(global_indices_local.tolist()):
        points = points_camera[slot].reshape(-1, 3)
        valid = torch.isfinite(points).all(dim=-1) & (points[:, 2] > 0)
        if isinstance(confidence, torch.Tensor) and confidence.shape == depth.shape:
            conf = confidence[slot].reshape(-1).float()
            valid &= torch.isfinite(conf)
            if bool(valid.any().item()):
                threshold = torch.quantile(conf[valid], 0.25)
                valid &= conf >= threshold
        points = _deterministic_point_subsample(points[valid], maximum_points_per_frame)
        if points.shape[0] == 0:
            continue
        c2w = w2c_to_c2w(camera[slot].float())
        world = points @ c2w[:3, :3].transpose(0, 1) + c2w[:3, 3]
        result[int(local_r_index)] = world.detach().cpu()
    return result


def merge_chunk_metric_depth_intrinsics(
    chunks: Sequence[tuple[torch.Tensor, dict[str, Any]]],
    residual_scales: Sequence[float],
    metric_scale_valids: Sequence[bool] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(chunks) != len(residual_scales) or not chunks:
        raise ValueError("metric scene chunks and residual scales must be non-empty and aligned")
    if metric_scale_valids is None:
        metric_scale_valids = [True] * len(chunks)
    if len(metric_scale_valids) != len(chunks):
        raise ValueError("metric-scale validity must align with metric scene chunks")
    depth_numerator: dict[int, torch.Tensor] = {}
    depth_denominator: dict[int, torch.Tensor] = {}
    confidence_sum: dict[int, torch.Tensor] = {}
    confidence_count: dict[int, torch.Tensor] = {}
    intrinsics_numerator: dict[int, torch.Tensor] = {}
    intrinsics_denominator: dict[int, torch.Tensor] = {}
    depth_shape: tuple[int, ...] | None = None
    for (frame_indices, outputs), residual_scale, chunk_scale_valid in zip(
        chunks,
        residual_scales,
        metric_scale_valids,
        strict=True,
    ):
        depth = outputs.get("depth_metric_global", outputs.get("depth_global"))
        confidence = outputs.get("depth_conf_global")
        intrinsics = outputs.get("intrinsics_global")
        if not all(isinstance(value, torch.Tensor) for value in (depth, intrinsics)):
            raise ValueError("every chunk must contain metric depth and K on its G axis")
        assert isinstance(depth, torch.Tensor)
        assert isinstance(intrinsics, torch.Tensor)
        indices = frame_indices.detach().cpu().to(dtype=torch.long).reshape(-1)
        if depth.ndim not in (3, 4) or depth.shape[0] != indices.numel():
            raise ValueError("chunk metric depth must align with its global frame indices")
        if tuple(intrinsics.shape) != (indices.numel(), 3, 3):
            raise ValueError("chunk K must align with its metric depth")
        if confidence is None:
            confidence = torch.ones_like(depth)
        if not isinstance(confidence, torch.Tensor) or confidence.shape != depth.shape:
            raise ValueError("chunk depth confidence must align with metric depth")
        if depth_shape is None:
            depth_shape = tuple(depth.shape[1:])
        elif tuple(depth.shape[1:]) != depth_shape:
            raise ValueError("all chunk depth maps must share one raster shape")
        metric_depth = depth.detach().cpu().float() * float(residual_scale)
        depth_confidence = confidence.detach().cpu().float()
        chunk_K = intrinsics.detach().cpu().float()
        for slot, frame_index in enumerate(indices.tolist()):
            value = metric_depth[slot]
            conf = depth_confidence[slot]
            valid = (
                torch.isfinite(value)
                & (value > 0)
                & torch.isfinite(conf)
                & bool(chunk_scale_valid)
            )
            weight = torch.where(
                valid,
                torch.nan_to_num(conf, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(1e-6),
                torch.zeros_like(conf),
            )
            weighted_depth = torch.where(valid, value, torch.zeros_like(value)) * weight
            if frame_index not in depth_numerator:
                depth_numerator[frame_index] = weighted_depth
                depth_denominator[frame_index] = weight
                confidence_sum[frame_index] = torch.where(valid, conf, torch.zeros_like(conf))
                confidence_count[frame_index] = valid.to(dtype=torch.float32)
            else:
                depth_numerator[frame_index] += weighted_depth
                depth_denominator[frame_index] += weight
                confidence_sum[frame_index] += torch.where(valid, conf, torch.zeros_like(conf))
                confidence_count[frame_index] += valid.to(dtype=torch.float32)
            K = chunk_K[slot]
            K_valid = bool(torch.isfinite(K).all().item())
            K_weight = float(weight.mean().item()) if bool(valid.any().item()) else 0.0
            if not K_valid:
                K_weight = 0.0
            if K_weight <= 0.0 and K_valid:
                K_weight = 1.0
            weighted_K = K * K_weight if K_valid else torch.zeros_like(K)
            if frame_index not in intrinsics_numerator:
                intrinsics_numerator[frame_index] = weighted_K
                intrinsics_denominator[frame_index] = torch.tensor(K_weight)
            else:
                intrinsics_numerator[frame_index] += weighted_K
                intrinsics_denominator[frame_index] += K_weight

    merged_indices = torch.tensor(sorted(depth_numerator), dtype=torch.long)
    merged_depth = torch.stack(
        [
            depth_numerator[index]
            / depth_denominator[index].clamp_min(1e-6)
            for index in merged_indices.tolist()
        ]
    )
    merged_confidence = torch.stack(
        [
            confidence_sum[index]
            / confidence_count[index].clamp_min(1.0)
            for index in merged_indices.tolist()
        ]
    )
    merged_K = torch.stack(
        [
            intrinsics_numerator[index]
            / intrinsics_denominator[index].clamp_min(1e-6)
            for index in merged_indices.tolist()
        ]
    )
    merged_valid = torch.tensor(
        [
            bool((depth_denominator[index] > 0).any().item())
            for index in merged_indices.tolist()
        ],
        dtype=torch.bool,
    )
    return merged_indices, merged_depth, merged_confidence, merged_K, merged_valid


def resample_intrinsics_nearest(
    global_frame_indices: torch.Tensor,
    intrinsics_global: torch.Tensor,
    *,
    full_frame_count: int,
) -> torch.Tensor:
    indices = global_frame_indices.detach().cpu().to(dtype=torch.long).reshape(-1)
    intrinsics = intrinsics_global.detach().cpu()
    if full_frame_count < 1:
        raise ValueError("full_frame_count must be positive")
    if tuple(intrinsics.shape) != (indices.numel(), 3, 3) or indices.numel() < 1:
        raise ValueError("native G-rate K and frame indices must be non-empty and aligned")
    if bool(((indices < 0) | (indices >= int(full_frame_count))).any().item()):
        raise ValueError("native G frame indices must lie on the R axis")
    if indices.numel() > 1 and not bool((indices[1:] > indices[:-1]).all().item()):
        raise ValueError("native G frame indices must be strictly increasing")
    valid = torch.isfinite(intrinsics).all(dim=(-2, -1))
    valid &= intrinsics[:, 0, 0].abs() > 1e-6
    valid &= intrinsics[:, 1, 1].abs() > 1e-6
    if not bool(valid.any().item()):
        raise ValueError("native G-rate K contains no valid anchor")
    query = torch.arange(full_frame_count, dtype=torch.long).unsqueeze(1)
    distance = (query - indices.unsqueeze(0)).abs().to(dtype=torch.float64)
    distance[:, ~valid] = float("inf")
    nearest = distance.argmin(dim=1)
    return intrinsics.index_select(0, nearest)


def blend_w2c_pose(left: torch.Tensor, right: torch.Tensor, alpha: float) -> torch.Tensor:
    left_c2w = w2c_to_c2w(left.to(dtype=torch.float64))
    right_c2w = w2c_to_c2w(right.to(dtype=torch.float64))
    rotation = slerp_rotation_matrices(
        left_c2w[:3, :3],
        right_c2w[:3, :3],
        float(alpha),
    )
    center = (1.0 - float(alpha)) * left_c2w[:3, 3] + float(alpha) * right_c2w[:3, 3]
    c2w = torch.eye(4, dtype=torch.float64, device=left.device)
    c2w[:3, :3] = rotation
    c2w[:3, 3] = center
    return c2w_to_w2c(c2w).to(dtype=_geometry_dtype(left.dtype))


def _clip_temporal_outputs(
    outputs: dict[str, Any],
    *,
    clip_frames: int,
) -> dict[str, torch.Tensor]:
    return {
        name: value
        for name, value in outputs.items()
        if isinstance(value, torch.Tensor)
        and name not in _REBUILT_CANONICAL_FULL_KEYS
        and value.ndim >= 1
        and value.shape[0] == clip_frames
        and (name.endswith("_full") or name in {"intrinsics_full", "refiner_output_present"})
    }


def stitch_clip_outputs(
    clips: Iterable[tuple[int, dict[str, Any]]],
    *,
    full_frame_count: int,
    alignment_diagnostics: list[dict[str, float]] | None = None,
    estimate_camera_scale: bool = True,
    allow_invalid_camera: bool = False,
) -> tuple[dict[str, torch.Tensor], list[torch.Tensor]]:
    stitched: dict[str, torch.Tensor] = {}
    filled = torch.zeros(full_frame_count, dtype=torch.bool)
    alignments: list[torch.Tensor] = []
    scene_points_by_global_frame: dict[int, torch.Tensor] = {}
    for clip_index, (start, raw_outputs) in enumerate(clips):
        camera = raw_outputs.get("camera_pose_refined_full")
        if not isinstance(camera, torch.Tensor) or camera.ndim != 3 or camera.shape[-2:] != (4, 4):
            raise ValueError("each clip must provide camera_pose_refined_full [T, 4, 4]")
        camera = camera.to(dtype=_geometry_dtype(camera.dtype))
        clip_frames = int(camera.shape[0])
        end = start + clip_frames
        if start < 0 or end > full_frame_count:
            raise ValueError("clip output lies outside the full R axis")
        outputs = {name: value.clone() for name, value in _clip_temporal_outputs(raw_outputs, clip_frames=clip_frames).items()}
        for key in _CAMERA_POSE_FULL_KEYS:
            if key in outputs:
                outputs[key] = outputs[key].to(dtype=_geometry_dtype(outputs[key].dtype))
        raw_scene_points = raw_outputs.get("_alignment_scene_points_by_r", {})
        scene_points_local = (
            {
                int(index): value.detach().cpu().float()
                for index, value in raw_scene_points.items()
                if isinstance(index, int) and isinstance(value, torch.Tensor)
            }
            if isinstance(raw_scene_points, dict)
            else {}
        )
        chamfer_before = 0.0
        chamfer_after = 0.0
        overlap_frame_count = 0
        chamfer_used = False
        chamfer_camera_rejected = False
        camera_errors = torch.zeros(2, 2, dtype=torch.float64)
        camera_unaligned = False
        overlap_residual_scale = 1.0
        overlap_scale_valid = clip_index == 0
        overlap_scale_pair_count = 0
        if clip_index == 0:
            alignment = torch.eye(4, dtype=camera.dtype)
        else:
            overlap_global = torch.arange(start, end)[filled[start:end]]
            if overlap_global.numel() == 0:
                raise ValueError("adjacent clips must overlap on the R axis")
            overlap_local = overlap_global - start
            overlap_frame_count = int(overlap_global.numel())
            current_valid = outputs.get("camera_refined_valid_full")
            reference_valid = stitched.get("camera_refined_valid_full")
            valid = None
            if isinstance(current_valid, torch.Tensor) and isinstance(reference_valid, torch.Tensor):
                valid = current_valid[overlap_local].to(dtype=torch.bool) & reference_valid[overlap_global].to(dtype=torch.bool)
            if estimate_camera_scale:
                (
                    overlap_residual_scale,
                    overlap_scale_valid,
                    overlap_scale_pair_count,
                ) = estimate_overlap_metric_scale(
                    stitched["camera_pose_refined_full"][overlap_global],
                    outputs["camera_pose_refined_full"][overlap_local],
                    valid,
                )
            for key in _CAMERA_POSE_FULL_KEYS:
                if key in outputs:
                    outputs[key] = scale_w2c_world_centers(
                        outputs[key],
                        overlap_residual_scale,
                    )
            scene_points_local = {
                index: points * float(overlap_residual_scale)
                for index, points in scene_points_local.items()
            }
            camera_unaligned = valid is not None and not bool(valid.any())
            if camera_unaligned and allow_invalid_camera:
                pose_alignment = torch.eye(4,dtype=camera.dtype)
                if bool(stitched["camera_refined_valid_full"].any()):
                    for key in ("camera_refined_valid_full","camera_interpolation_valid_full"):
                        if key in outputs: outputs[key]=torch.zeros_like(outputs[key])
                    scene_points_local={}
            else:
                pose_alignment = estimate_world_alignment(
                    stitched["camera_pose_refined_full"][overlap_global],
                    outputs["camera_pose_refined_full"][overlap_local], valid,
                )
            reference_scene: list[torch.Tensor] = []
            current_scene: list[torch.Tensor] = []
            overlap_global_set = {int(index) for index in overlap_global.tolist()}
            for global_index in overlap_global_set:
                reference_points = scene_points_by_global_frame.get(int(global_index))
                if reference_points is not None:
                    reference_scene.append(reference_points)
            for local_index, current_points in scene_points_local.items():
                if start + int(local_index) in overlap_global_set:
                    current_scene.append(current_points)
            if reference_scene and current_scene:
                chamfer_used = True
                alignment, chamfer_before, chamfer_after = refine_world_alignment_chamfer(
                    torch.cat(reference_scene, dim=0),
                    torch.cat(current_scene, dim=0),
                    pose_alignment,
                )
            else:
                alignment = pose_alignment
            baseline_errors = (_overlap_camera_errors(
                stitched["camera_pose_refined_full"][overlap_global],
                outputs["camera_pose_refined_full"][overlap_local], pose_alignment, valid,
            ) if not camera_unaligned else torch.zeros(2,2,dtype=torch.float64))
            camera_errors = (_overlap_camera_errors(
                stitched["camera_pose_refined_full"][overlap_global],
                outputs["camera_pose_refined_full"][overlap_local], alignment, valid,
            ) if not camera_unaligned else baseline_errors)
            tolerance = camera_errors.new_tensor([.001, math.radians(.1)])
            if not bool(torch.isfinite(camera_errors).all()) or bool((camera_errors > baseline_errors + tolerance).any()):
                chamfer_camera_rejected = True
                alignment = pose_alignment
                camera_errors = baseline_errors
                chamfer_after = chamfer_before
            for key in _CAMERA_POSE_FULL_KEYS:
                if key in outputs:
                    outputs[key] = apply_world_alignment_to_w2c(outputs[key], alignment)
        alignments.append(alignment.detach().cpu())
        if alignment_diagnostics is not None:
            alignment_diagnostics.append(
                {
                    "clip_index": float(clip_index),
                    "overlap_frame_count": float(overlap_frame_count),
                    "chamfer_used": float(chamfer_used),
                    "chamfer_before": float(chamfer_before),
                    "chamfer_after": float(chamfer_after),
                    "chamfer_camera_rejected": float(chamfer_camera_rejected),
                    "camera_position_p50_m": float(camera_errors[0, 0]),
                    "camera_position_p90_m": float(camera_errors[1, 0]),
                    "camera_rotation_p50_deg": math.degrees(float(camera_errors[0, 1])),
                    "camera_rotation_p90_deg": math.degrees(float(camera_errors[1, 1])),
                    "overlap_residual_scale": float(overlap_residual_scale),
                    "overlap_scale_valid": float(overlap_scale_valid),
                    "overlap_scale_pair_count": float(overlap_scale_pair_count),
                    "camera_scale_estimation_enabled": float(estimate_camera_scale),
                    "camera_alignment_unavailable": float(camera_unaligned),
                }
            )
        aligned_scene_points = {
            int(local_index): apply_world_alignment_to_points(points, alignment).detach().cpu()
            for local_index, points in scene_points_local.items()
        }

        for name, value in outputs.items():
            if name not in stitched:
                stitched[name] = torch.zeros(
                    (full_frame_count, *value.shape[1:]),
                    dtype=value.dtype,
                )
        overlap_indices = torch.arange(start, end)[filled[start:end]]
        overlap_count = int(overlap_indices.numel())
        overlap_alpha = {
            int(global_index): _smooth_overlap_alpha(position, overlap_count)
            for position, global_index in enumerate(overlap_indices.tolist())
        }
        for local_index, global_index in enumerate(range(start, end)):
            if not filled[global_index]:
                for name, value in outputs.items():
                    stitched[name][global_index] = value[local_index]
                continue
            alpha = overlap_alpha[global_index]
            camera_validity = None
            if "camera_refined_valid_full" in stitched and "camera_refined_valid_full" in outputs:
                camera_validity=(bool(stitched["camera_refined_valid_full"][global_index]),bool(outputs["camera_refined_valid_full"][local_index]))
            reference_hand_valid = stitched.get("hand_refined_valid_full")
            current_hand_valid = outputs.get("hand_refined_valid_full")
            side_validity = None
            if (
                isinstance(reference_hand_valid, torch.Tensor)
                and isinstance(current_hand_valid, torch.Tensor)
            ):
                side_validity = (
                    reference_hand_valid[global_index].to(dtype=torch.bool).clone(),
                    current_hand_valid[local_index].to(dtype=torch.bool).clone(),
                )
                if "hand_missing_fill_full" in stitched:
                    side_validity[0].logical_and_(~stitched["hand_missing_fill_full"][global_index])
                if "hand_missing_fill_full" in outputs:
                    side_validity[1].logical_and_(~outputs["hand_missing_fill_full"][local_index])
            handled_geometry: set[str] = set()
            for suffix in ("base", "interpolated", "refined", "residual"):
                joint_name = f"dense_joint_xyz_{suffix}_full"
                vertex_name = f"dense_vertex_xyz_{suffix}_full"
                if joint_name not in outputs or joint_name not in stitched:
                    continue
                reference_joint = stitched[joint_name][global_index]
                current_joint = outputs[joint_name][local_index]
                reference_vertex = (
                    stitched[vertex_name][global_index]
                    if vertex_name in stitched and vertex_name in outputs
                    else None
                )
                current_vertex = (
                    outputs[vertex_name][local_index]
                    if reference_vertex is not None
                    else None
                )
                blended_joint, blended_vertex = _blend_root_relative_geometry(
                    reference_joint,
                    current_joint,
                    reference_vertex,
                    current_vertex,
                    alpha,
                )
                if side_validity is None:
                    reference_valid = torch.ones(2, dtype=torch.bool)
                    current_valid = torch.ones(2, dtype=torch.bool)
                else:
                    reference_valid, current_valid = side_validity
                use_blend = reference_valid & current_valid
                take_current = ~reference_valid & current_valid
                joint_result = torch.where(
                    _expand_side_mask(use_blend, blended_joint),
                    blended_joint,
                    reference_joint,
                )
                stitched[joint_name][global_index] = torch.where(
                    _expand_side_mask(take_current, current_joint),
                    current_joint,
                    joint_result,
                )
                handled_geometry.add(joint_name)
                if blended_vertex is not None and reference_vertex is not None and current_vertex is not None:
                    vertex_result = torch.where(
                        _expand_side_mask(use_blend, blended_vertex),
                        blended_vertex,
                        reference_vertex,
                    )
                    stitched[vertex_name][global_index] = torch.where(
                        _expand_side_mask(take_current, current_vertex),
                        current_vertex,
                        vertex_result,
                    )
                    handled_geometry.add(vertex_name)
            for name, value in outputs.items():
                if name in handled_geometry:
                    continue
                if name == "hand_missing_fill_full" and side_validity is not None:
                    reference_valid, current_valid = side_validity
                    stitched[name][global_index] = (
                        (stitched[name][global_index] | value[local_index])
                        & ~(reference_valid | current_valid)
                    )
                elif value.dtype is torch.bool:
                    stitched[name][global_index] |= value[local_index]
                elif (
                    side_validity is not None
                    and _is_per_side_hand_full_tensor(name, value)
                ):
                    reference_valid, current_valid = side_validity
                    both_valid = reference_valid & current_valid
                    neither_valid = ~reference_valid & ~current_valid
                    take_current = ~reference_valid & current_valid
                    if value.is_floating_point():
                        blended = (
                            (1.0 - alpha) * stitched[name][global_index]
                            + alpha * value[local_index]
                        )
                        use_blend = _expand_side_mask(
                            both_valid | neither_valid,
                            blended,
                        )
                        use_current = _expand_side_mask(
                            take_current,
                            value[local_index],
                        )
                        composed = torch.where(
                            use_blend,
                            blended,
                            stitched[name][global_index],
                        )
                        stitched[name][global_index] = torch.where(
                            use_current,
                            value[local_index],
                            composed,
                        )
                    else:
                        choose_current = take_current | (
                            (both_valid | neither_valid) & (alpha >= 0.5 - 1e-12)
                        )
                        stitched[name][global_index] = torch.where(
                            _expand_side_mask(
                                choose_current,
                                value[local_index],
                            ),
                            value[local_index],
                            stitched[name][global_index],
                        )
                elif not value.is_floating_point():
                    if alpha >= 0.5 - 1e-12:
                        stitched[name][global_index] = value[local_index]
                elif name in _CAMERA_POSE_FULL_KEYS:
                    if camera_validity is None or all(camera_validity):
                        stitched[name][global_index] = blend_w2c_pose(stitched[name][global_index], value[local_index], alpha)
                    elif camera_validity[1]:
                        stitched[name][global_index] = value[local_index]
                else:
                    stitched[name][global_index] = (
                        (1.0 - alpha) * stitched[name][global_index]
                        + alpha * value[local_index]
                    )
        filled[start:end] = True
        for local_index, points in aligned_scene_points.items():
            global_index = start + int(local_index)
            if not 0 <= global_index < full_frame_count:
                continue
            previous = scene_points_by_global_frame.get(global_index)
            scene_points_by_global_frame[global_index] = (
                points
                if previous is None
                else _deterministic_point_subsample(torch.cat((previous, points), dim=0), 512)
            )
    if not alignments:
        raise ValueError("at least one clip output is required")
    if not bool(filled.all().item()):
        raise RuntimeError("multi-clip stitching left uncovered R frames")
    return stitched, alignments


def run_long_video_inference(
    *,
    project_config: Any,
    model: torch.nn.Module,
    frames: Sequence[torch.Tensor],
    video_path: Path,
    output_dir: Path,
    device: torch.device,
    rate_contract: TemporalRateContract,
    chunk_seconds: float = DEFAULT_CLIP_SECONDS,
    overlap_seconds: float = DEFAULT_OVERLAP_SECONDS,
    save_overlay: bool = True,
    save_overlay_frame_images: bool = True,
    allow_output_overwrite: bool = False,
    provenance: dict[str, Any] | None = None,
    include_debug_tensors: bool = False,
    hand_h_root_smoothing: bool = False,
    upsample_mano: bool = False,
    overlay_marker_count: int = 195,
    root_z_options: RootZOptions | None = None,
    root_intrinsics_input: RootIntrinsicsInput | None = None,
    cli_started_at: float | None = None,
    hand_anchor_options: HandAnchorOptions | None = None,
    hand_depth_scale_options: HandDepthScaleOptions | None = None,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    if len(frames) < 1:
        raise ValueError("video must contain at least one frame")
    image_size_hw = frames.image_size_hw if isinstance(frames, WindowedVideoFrames) else tuple(frames[0].shape[-2:])
    if overlay_marker_count not in (195, 778):
        raise ValueError("overlay marker count must be 195 or 778")
    prepare_inference_output_dir(
        output_dir,
        allow_overwrite=bool(allow_output_overwrite),
    )
    clip_frames = rate_contract.frame_count_for_seconds(float(chunk_seconds))
    overlap_frames = rate_contract.frame_count_for_seconds(float(overlap_seconds))
    windows = build_overlapping_r_windows(
        len(frames),
        clip_frames=clip_frames,
        overlap_frames=overlap_frames,
    )
    clip_metric_chunks: list[tuple[torch.Tensor, dict[str, Any]]] = []
    clip_metadata: list[dict[str, Any]] = []
    clip_hand_root_smoothing: list[dict[str, Any]] = []
    clip_root_intrinsics: list[dict[str, Any] | None] = []
    clip_hand_anchor_filters: list[dict[str, Any]] = []
    clip_hand_depth_scales: list[dict[str, Any]] = []
    graph_active = bool(hand_depth_scale_options is not None and hand_depth_scale_options.enabled
                        and hand_depth_scale_options.overlap_propagation and len(windows)>1)
    depth_samples = []
    raw_cameras = []
    video_global_phase = rate_contract.global_stride // 2
    video_high_indices = build_video_global_high_axis(
        rate_contract,
        frame_count=len(frames),
        windows=windows,
    )
    if isinstance(frames, WindowedVideoFrames):
        frames.configure_cache(window_frames=clip_frames, required_indices=video_high_indices)

    def iter_clips():
        for r_start, r_end_exclusive in windows:
            local_high, local_anchor_phase, first_high_ordinal = (
                select_video_window_high_route(
                    video_high_indices,
                    r_start=r_start,
                    r_end_exclusive=r_end_exclusive,
                    global_stride=rate_contract.global_stride,
                    video_global_phase=video_global_phase,
                )
            )
            source_indices = list(range(r_start, r_end_exclusive))
            smoothing_kwargs = (
                {"hand_h_root_smoothing": True} if hand_h_root_smoothing else {}
            )
            if root_intrinsics_input is not None:
                smoothing_kwargs["root_intrinsics_input"] = root_intrinsics_input
            if hand_anchor_options is not None:
                smoothing_kwargs["hand_anchor_options"] = replace(
                    hand_anchor_options,
                    smooth_local_shape=False,
                ) if hand_anchor_options.smooth_local_shape else hand_anchor_options
            if hand_depth_scale_options is not None:
                smoothing_kwargs["hand_depth_scale_options"] = hand_depth_scale_options
            result = run_clip_forward(
                project_config=project_config,
                model=model,
                frames=frames[r_start:r_end_exclusive],
                high_indices_local=local_high,
                source_frame_indices=source_indices,
                fps=rate_contract.r_fps,
                device=device,
                global_stride=rate_contract.global_stride,
                global_anchor_phase=local_anchor_phase,
                **smoothing_kwargs,
            )
            smoothing_diagnostics = build_hand_h_root_smoothing_payload(
                result,
                enabled=hand_h_root_smoothing,
            )
            if hand_anchor_options is not None:
                clip_hand_anchor_filters.append(hand_anchor_filter_payload(result, hand_anchor_options))
            if result.hand_depth_scale is not None:
                clip_hand_depth_scales.append(result.hand_depth_scale)
            clip_hand_root_smoothing.append(smoothing_diagnostics)
            clip_root_intrinsics.append(build_root_intrinsics_record(
                result.outputs, high_frame_indices=result.high_source_indices,
                source_frame_indices=source_indices,
                image_size_hw=image_size_hw,
                calibration=root_intrinsics_input,
            ))
            clip_outputs = _clip_temporal_outputs(result.outputs, clip_frames=result.full_frame_count)
            if not graph_active:
                clip_outputs["_alignment_scene_points_by_r"] = build_alignment_scene_points(
                    result.outputs,
                    result.global_indices_local,
                )
            global_frame_indices = torch.tensor(
                [r_start + int(index) for index in result.global_indices_local.tolist()],
                dtype=torch.long,
            )
            metric_keys = (("depth_raw_global", "depth_conf_global", "intrinsics_global") if graph_active
                           else ("depth_metric_global", "depth_global", "depth_conf_global", "intrinsics_global"))
            clip_metric_chunks.append((global_frame_indices, {key: result.outputs[key] for key in metric_keys if key in result.outputs}))
            if graph_active:
                depth_samples.append(compact_depth_sample(result.outputs["depth_raw_global"],result.outputs["depth_conf_global"],global_frame_indices))
                raw_cameras.append(result.outputs["camera_pose_raw_global"])

            clip_metadata.append(
                {
                    "r_window": [r_start, r_end_exclusive],
                    "high_indices_local": list(local_high),
                    "global_indices_local": result.global_indices_local.tolist(),
                    "global_frame_indices": global_frame_indices.tolist(),
                    "T_H": int(result.high_indices_local.numel()),
                    "T_G": int(result.global_indices_local.numel()),
                    "T_R": result.full_frame_count,
                    "global_stride": result.global_stride,
                    "global_anchor_phase_local": result.global_anchor_phase,
                    "video_high_ordinal_start": first_high_ordinal,
                    "forward_seconds": result.forward_seconds,
                    "peak_memory_bytes": result.peak_memory_bytes,
                    "peak_reserved_memory_bytes": result.peak_reserved_memory_bytes,
                    "hand_h_root_smoothing_applied_count": int(
                        smoothing_diagnostics["applied"].sum().item()
                    ),
                    "metric_value_pred": _scalar_tensor_value(
                        result.outputs,
                        "metric_value_pred",
                        "metric_value",
                    ),
                    "pred_scene_scale_raw": _scalar_tensor_value(
                        result.outputs,
                        "pred_scene_scale_raw",
                        "s_pred_scene",
                    ),
                    "raw_to_meter_pred": _scalar_tensor_value(
                        result.outputs,
                        "raw_to_meter_pred",
                        "metric_scale_factor",
                        default=1.0,
                    ),
                    "raw_to_meter_fused": _scalar_tensor_value(result.outputs, "raw_to_meter_fused", "raw_to_meter_pred", "metric_scale_factor", default=1.),
                    "hand_depth_scale_seconds": float(result.hand_depth_scale["seconds"]) if result.hand_depth_scale is not None else 0.,
                    "metric_scale_valid": bool(
                        _scalar_tensor_value(
                            result.outputs,
                            "metric_scale_valid",
                            default=0.0,
                        )
                    ),
                }
            )
            yield r_start, clip_outputs
            del result, clip_outputs

    alignment_diagnostics: list[dict[str, float]] = []
    scale_graph = None
    if graph_active:
        with tempfile.TemporaryDirectory(prefix="egofound3r_scale_") as temporary:
            directory=Path(temporary)
            spool_seconds=0.;spool_bytes=0
            for index,(_start,values) in enumerate(iter_clips()):
                write_started=time.perf_counter()
                path=directory/f"{index:06d}.pt"
                torch.save(values,path)
                spool_seconds+=time.perf_counter()-write_started
                spool_bytes+=path.stat().st_size
            solve_started=time.perf_counter()
            edges=[]
            for left in range(len(windows)):
                for right in range(left+1,len(windows)):
                    if windows[right][0]>=windows[left][1]: break
                    edge=estimate_overlap_depth_ratio(depth_samples[left],depth_samples[right])
                    edges.append(dict(edge,left=left,right=right))
            scale_graph=resolve_scale_graph([item["batches"][0] for item in clip_hand_depth_scales],edges)
            scale_graph["timing"]={"spool_write_seconds":spool_seconds,"resolve_seconds":time.perf_counter()-solve_started,
                                   "rebuild_scene_seconds":0.,"spool_bytes":spool_bytes}
            del depth_samples

            def resolved_clips():
                for index,(start,_end) in enumerate(windows):
                    rebuild_started=time.perf_counter()
                    values=torch.load(directory/f"{index:06d}.pt",map_location="cpu",weights_only=True)
                    indices,raw_scene=clip_metric_chunks[index]
                    metadata=clip_metadata[index]
                    metadata["raw_to_meter_local"]=metadata["raw_to_meter_fused"]
                    values,scene=rebuild_chunk_scene(values,raw_scene,raw_cameras[index],metadata,scale_graph["nodes"][index])
                    clip_metric_chunks[index]=(indices,{key:scene[key] for key in ("depth_metric_global","depth_global","depth_conf_global","intrinsics_global")})
                    aligned_scene=dict(scene,depth_global=torch.where(scene["frame_valid"][:,None,None],scene["depth_global"],0.))
                    values["_alignment_scene_points_by_r"]=build_alignment_scene_points(aligned_scene,torch.tensor(metadata["global_indices_local"]))
                    scale_graph["timing"]["rebuild_scene_seconds"]+=time.perf_counter()-rebuild_started
                    yield start,values

            stitched,alignments=stitch_clip_outputs(resolved_clips(),full_frame_count=len(frames),
                alignment_diagnostics=alignment_diagnostics,estimate_camera_scale=False,allow_invalid_camera=True)
    else:
        stitched, alignments = stitch_clip_outputs(
            iter_clips(), full_frame_count=len(frames), alignment_diagnostics=alignment_diagnostics,
        )
    stitched["camera_pose_metric_full"] = stitched.get(
        "camera_pose_interpolated_full",
        stitched["camera_pose_refined_full"],
    )
    stitched["camera_pose_metric_valid_full"] = stitched.get(
        "camera_interpolation_valid_full",
        stitched["camera_refined_valid_full"],
    )
    if hand_anchor_options is not None and hand_anchor_options.enabled and (
        hand_anchor_options.fill_missing or hand_anchor_options.smooth_local_shape
    ):
        _apply_global_camera_space_hand_fill(
            stitched,
            fill_missing=hand_anchor_options.fill_missing,
            smooth_local_shape=hand_anchor_options.smooth_local_shape,
        )
    residual_scales = [
        float(item.get("overlap_residual_scale", 1.0))
        for item in alignment_diagnostics
    ]
    for metadata, diagnostics in zip(
        clip_metadata,
        alignment_diagnostics,
        strict=True,
    ):
        metadata.update(
            {
                "overlap_residual_scale": float(
                    diagnostics.get("overlap_residual_scale", 1.0)
                ),
                "overlap_scale_valid": bool(
                    diagnostics.get("overlap_scale_valid", 0.0)
                ),
                "raw_to_meter_effective": float(metadata["raw_to_meter_fused"])
                * float(diagnostics.get("overlap_residual_scale", 1.0)),
            }
        )
    (
        global_frame_indices,
        depth_metric_global,
        depth_confidence_global,
        intrinsics_K_global,
        depth_metric_valid_global,
    ) = merge_chunk_metric_depth_intrinsics(
        clip_metric_chunks,
        residual_scales,
        [bool(item["metric_scale_valid"]) for item in clip_metadata],
    )
    intrinsics_K_full = resample_intrinsics_nearest(
        global_frame_indices,
        intrinsics_K_global,
        full_frame_count=len(frames),
    )
    high_indices = video_high_indices
    smoothing_applied_count = sum(
        int(item["applied"].sum().item()) for item in clip_hand_root_smoothing
    )
    smoothing_contract = {
        "enabled": bool(hand_h_root_smoothing),
        "mode": clip_hand_root_smoothing[0]["mode"],
        "window_frames_h": clip_hand_root_smoothing[0]["window_frames_h"],
        "min_strength": clip_hand_root_smoothing[0]["min_strength"],
        "max_strength": clip_hand_root_smoothing[0]["max_strength"],
        "applied_count": smoothing_applied_count,
        "clips": clip_hand_root_smoothing,
    }
    payload: dict[str, Any] = {
        "video_path": str(video_path.resolve()),
        "fps": float(rate_contract.r_fps),
        "r_fps": float(rate_contract.r_fps),
        "h_fps": float(rate_contract.h_fps),
        "g_fps": float(rate_contract.g_fps),
        "r_to_h_ratio": float(rate_contract.r_to_h_ratio),
        "global_stride": int(rate_contract.global_stride),
        "global_anchor_phase": int(video_global_phase),
        "chunk_seconds": float(chunk_seconds),
        "overlap_seconds": float(overlap_seconds),
        "clip_frames_r": int(clip_frames),
        "overlap_frames_r": int(overlap_frames),
        "source_frame_indices": torch.arange(len(frames), dtype=torch.long),
        "frame_ids": [str(index) for index in range(len(frames))],
        "timestamps_sec": torch.arange(len(frames), dtype=torch.float64)
        / float(rate_contract.r_fps),
        "image_size_hw": list(image_size_hw),
        "high_frame_indices": torch.as_tensor(high_indices, dtype=torch.long),
        "global_frame_indices": global_frame_indices,
        "clip_metadata": clip_metadata,
        "clip_world_alignments": torch.stack(alignments),
        "clip_alignment_diagnostics": alignment_diagnostics,
        "index_domain_contract": {
            "R": "input video frames",
            "H": "intermediate frames",
            "G": "anchor frames",
        },
        "scale_contract": {
            "hand_xyz": "meter",
            "hand_offset": "meter",
            "contact_distance": "log1p_mm",
            "metric_scale_applies_to_hand": False,
        },
        "T_R": len(frames),
        "T_H_canonical": len(high_indices),
        "clip_count": len(clip_metadata),
        "T_G": int(global_frame_indices.numel()),
        "depth_metric_global": depth_metric_global,
        "depth_metric_valid_global": depth_metric_valid_global,
        "depth_confidence_global": depth_confidence_global,
        "intrinsics_K_global": intrinsics_K_global,
        "intrinsics_K_full": intrinsics_K_full,
        "camera_pose_metric_global": stitched["camera_pose_metric_full"].index_select(
            0,
            global_frame_indices,
        ),
        "hand_h_root_smoothing": smoothing_contract,
    }
    if provenance is not None:
        payload["provenance"] = dict(provenance)
    if hand_anchor_options is not None:
        payload["hand_anchor_filter"] = {"schema_version": "egofound3r.hand_anchor_filter.v1", "clips": clip_hand_anchor_filters}
    if hand_depth_scale_options is not None:
        payload["hand_depth_scale"] = {"schema_version": "egofound3r.hand_depth_scale.v1", "clips": clip_hand_depth_scales}
    if scale_graph is not None:
        payload["overlap_depth_scale"] = scale_graph
    if include_debug_tensors:
        payload.update(stitched)
    payload["depth_global"] = depth_metric_global
    payload["depth_conf_global"] = depth_confidence_global
    payload["intrinsics_global"] = intrinsics_K_global
    payload["intrinsics_full"] = intrinsics_K_full
    payload["camera_pose_global"] = payload["camera_pose_metric_global"]
    aliases = {
        "camera_pose": "camera_pose_refined_full",
        "camera_pose_base": "camera_pose_base_full",
        "joint_xyz": "dense_joint_xyz_refined_full",
        "joint_xyz_base": "dense_joint_xyz_base_full",
        "vertex_xyz": "dense_vertex_xyz_refined_full",
        "vertex_xyz_base": "dense_vertex_xyz_base_full",
        "joint_visibility_logits": "dense_joint_visibility_logits_refined_full",
        "vertex_visibility_logits": "dense_vertex_visibility_logits_refined_full",
        "joint_contact_logits": "dense_joint_contact_logits_refined_full",
        "vertex_contact_logits": "dense_vertex_contact_logits_refined_full",
        "intrinsics": "intrinsics_full",
    }
    for alias, source in aliases.items():
        if source in stitched:
            payload[alias] = stitched[source]
    for name in (
        "hand_missing_fill_full",
        "hand_missing_fill_mode_full",
        "hand_display_valid_full",
        "hand_display_joint_xyz_full",
        "hand_display_vertex_xyz_full",
        "hand_display_local_smoothing_applied_full",
    ):
        if name in stitched:
            payload[name] = stitched[name]
    if "hand_display_joint_xyz_full" in stitched:
        payload["joint_xyz_display"] = stitched["hand_display_joint_xyz_full"]
    if "hand_display_vertex_xyz_full" in stitched:
        payload["vertex_xyz_display"] = stitched["hand_display_vertex_xyz_full"]
    metric_value_by_clip = torch.tensor(
        [float(item["metric_value_pred"]) for item in clip_metadata],
        dtype=torch.float32,
    )
    scene_scale_by_clip = torch.tensor(
        [float(item["pred_scene_scale_raw"]) for item in clip_metadata],
        dtype=torch.float32,
    )
    predicted_factor_by_clip = torch.tensor(
        [float(item["raw_to_meter_pred"]) for item in clip_metadata],
        dtype=torch.float32,
    )
    fused_factor_by_clip = torch.tensor([float(item["raw_to_meter_fused"]) for item in clip_metadata], dtype=torch.float32)
    residual_scale_by_clip = torch.tensor(residual_scales, dtype=torch.float32)
    effective_factor_by_clip = torch.tensor(
        [float(item["raw_to_meter_effective"]) for item in clip_metadata],
        dtype=torch.float32,
    )
    scale_valid_by_clip = torch.tensor(
        [bool(item["metric_scale_valid"]) for item in clip_metadata],
        dtype=torch.bool,
    )
    metric_predictions = build_metric_prediction_contract(
        stitched,
        depth_frame_indices=global_frame_indices,
        world_scope="stitched_video_world",
        depth_metric_global=depth_metric_global,
        depth_metric_valid_global=depth_metric_valid_global,
        depth_confidence_global=depth_confidence_global,
        intrinsics_K_global=intrinsics_K_global,
        intrinsics_K_full=intrinsics_K_full,
        metric_value_pred=metric_value_by_clip,
        pred_scene_scale_raw=scene_scale_by_clip,
        raw_to_meter_pred=predicted_factor_by_clip,
        raw_to_meter_fused=fused_factor_by_clip,
        overlap_residual_scale=residual_scale_by_clip,
        raw_to_meter_effective=effective_factor_by_clip,
        metric_scale_valid=scale_valid_by_clip,
    )
    attach_metric_prediction_contract(payload, metric_predictions)
    attach_root_intrinsics(payload, merge_root_intrinsics_records(
        clip_root_intrinsics, source_frame_indices=list(range(len(frames))),
        high_frame_indices=high_indices, calibration=root_intrinsics_input,
    ))
    if upsample_mano or overlay_marker_count == 778:
        attach_mano778_output(payload)
    apply_root_z_postprocessing(payload, root_z_options)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "inference_output.pt"
    save_marker_inference_output(output_path, payload)
    if save_overlay:
        if isinstance(frames, WindowedVideoFrames):
            source_height, _ = frames.source_size_hw
            display_request = (min(int(source_height), OVERLAY_MAX_HEIGHT), 1 << 20)
            overlay_size_hw = resolve_full_frame_input_size(
                frames.source_size_hw, display_request, mode="keep_aspect"
            )[0]
        else:
            display_request = image_size_hw
            overlay_size_hw = image_size_hw
        _write_optional_overlay(
            project_config=project_config,
            payload=payload,
            rgb_frames_uint8=iter_video_rgb(video_path, frame_count=len(frames), image_size_hw=display_request),
            output_dir=output_dir,
            fps=rate_contract.r_fps,
            marker_count=overlay_marker_count,
            save_frame_images=save_overlay_frame_images,
            overlay_size_hw=overlay_size_hw,
        )
    summary = {
        "T_R": len(frames),
        "T_H_canonical": len(high_indices),
        "pipeline_wall_seconds": time.perf_counter() - started_at,
        "overlay_frame_images": bool(save_overlay_frame_images),
        "R_fps": rate_contract.r_fps,
        "H_fps": rate_contract.h_fps,
        "G_fps": rate_contract.g_fps,
        "global_stride": rate_contract.global_stride,
        "global_anchor_phase": video_global_phase,
        "clip_count": len(clip_metadata),
        "T_G": int(global_frame_indices.numel()),
        "inference_schema_version": payload["inference_schema_version"],
        "metric_scale_valid_clips": int(scale_valid_by_clip.sum().item()),
        "forward_seconds_total": sum(float(item["forward_seconds"]) for item in clip_metadata),
        "hand_h_root_smoothing": {
            "enabled": bool(hand_h_root_smoothing),
            "mode": smoothing_contract["mode"],
            "applied_count": smoothing_applied_count,
        },
        "provenance": provenance,
        "output_path": str(output_path),
        "hand_778_exported": "hand_778" in payload,
        "overlay_marker_count": overlay_marker_count,
        "root_k_source": root_intrinsics_input.source if root_intrinsics_input else "pred",
    }
    if "root_z_postprocessing" in payload:
        summary["root_z_postprocessing"] = {
            key: payload["root_z_postprocessing"][key]
            for key in ("options", "applied_count", "depth_accepted_count")
        }
    if hand_depth_scale_options is not None:
        summary["hand_depth_scale"] = {"enabled": hand_depth_scale_options.enabled,
            "seconds": sum(float(row["seconds"]) for row in clip_hand_depth_scales),
            "raw_to_meter_pred": predicted_factor_by_clip.tolist(), "raw_to_meter_fused": fused_factor_by_clip.tolist(),
            "raw_to_meter_effective": effective_factor_by_clip.tolist()}
    if scale_graph is not None:
        summary["overlap_depth_scale"] = {"enabled":True,"nodes":scale_graph["nodes"],
            "accepted_edges":sum(int(edge["valid"]) for edge in scale_graph["edges"]),"timing":scale_graph["timing"]}
    if isinstance(frames, WindowedVideoFrames):
        summary["video_io"] = dict(frames.stats)
    finished_at = time.perf_counter()
    summary["pipeline_wall_seconds"] = finished_at - started_at
    summary["performance"] = build_inference_performance(
        clip_metadata, device=device, model_dtype=marker_model_floating_dtype(model),
        output_r_frames=len(frames), unique_h_frames=len(high_indices), unique_g_frames=int(global_frame_indices.numel()),
        input_r_fps=rate_contract.r_fps, pipeline_wall_seconds=summary["pipeline_wall_seconds"],
        end_to_end_wall_seconds=None if cli_started_at is None else finished_at - cli_started_at,
        overlay_enabled=save_overlay,
    )
    save_json_atomic(output_dir / "summary.json", summary)
    return summary
