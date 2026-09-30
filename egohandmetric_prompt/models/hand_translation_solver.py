from __future__ import annotations


from dataclasses import dataclass
from enum import IntEnum
from functools import wraps
import math
import torch


def _without_geometry_autocast(function):
    @wraps(function)
    def wrapped(relative_joints: torch.Tensor, *args, **kwargs):
        with torch.autocast(device_type=relative_joints.device.type, enabled=False):
            return function(relative_joints, *args, **kwargs)
    return wrapped


@dataclass(frozen=True, slots=True)
class HandTranslationSolve:
    translation: torch.Tensor
    valid: torch.Tensor
    valid_joint_count: torch.Tensor
    weighted_residual: torch.Tensor
    condition_number: torch.Tensor


@dataclass(frozen=True, slots=True)
class RootDepthFusion:
    depth: torch.Tensor
    analytic_weight: torch.Tensor
    analytic_valid: torch.Tensor
    geometric_quality: torch.Tensor
    condition_quality: torch.Tensor
    prior_agreement: torch.Tensor


@dataclass(frozen=True, slots=True)
class RootSubsetReliability:
    spread_m: torch.Tensor
    relative_spread: torch.Tensor
    valid_count: torch.Tensor
    valid: torch.Tensor
    score: torch.Tensor


class RootTranslationMode(IntEnum):
    INVALID = 0
    NONWRIST_PRIOR = 1
    NONWRIST_ANALYTIC_BLEND = 2
    WRIST_PRIOR = 3
    WRIST_ANALYTIC_BLEND = 4
    WRIST_CORRECTED = 5


@dataclass(frozen=True, slots=True)
class PartialHandTranslationSolve:
    translation: torch.Tensor
    valid: torch.Tensor
    analytic_translation: torch.Tensor
    analytic_valid: torch.Tensor
    valid_joint_count: torch.Tensor
    weighted_residual: torch.Tensor
    condition_number: torch.Tensor
    depth_blend_weight: torch.Tensor
    depth_condition_quality: torch.Tensor
    depth_geometric_quality: torch.Tensor
    depth_prior_agreement: torch.Tensor
    analytic_weighted_residual: torch.Tensor
    analytic_support_count: torch.Tensor
    mode: torch.Tensor
    wrist_used: torch.Tensor
    xy_consensus_valid_count: torch.Tensor
    xy_consensus_gate: torch.Tensor
    xy_consensus_correction_px: torch.Tensor
    xy_consensus_residual: torch.Tensor
    xy_consensus_corrected: torch.Tensor


def _checked_inputs(
    relative_joints: torch.Tensor,
    keypoint_uv: torch.Tensor,
    keypoint_confidence: torch.Tensor,
    keypoint_valid: torch.Tensor,
    normalized_intrinsics: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if relative_joints.ndim < 3 or relative_joints.shape[-1] != 3:
        raise ValueError("relative_joints must have shape [..., J, 3]")
    expected = relative_joints.shape[:-1]
    if keypoint_uv.shape != (*expected, 2):
        raise ValueError("keypoint_uv must have shape [..., J, 2]")
    if keypoint_confidence.shape != expected or keypoint_valid.shape != expected:
        raise ValueError("keypoint confidence/validity must align with relative_joints")
    if normalized_intrinsics.shape != (*relative_joints.shape[:-2], 4):
        raise ValueError("normalized_intrinsics must have shape [..., 4]")
    device = relative_joints.device
    if any(
        value.device != device
        for value in (keypoint_uv, keypoint_confidence, keypoint_valid, normalized_intrinsics)
    ):
        raise ValueError("all translation-solver inputs must share one device")
    work_dtype = torch.float64 if relative_joints.dtype == torch.float64 else torch.float32
    joints = relative_joints.to(dtype=work_dtype)
    uv = keypoint_uv.detach().to(dtype=work_dtype)
    confidence = keypoint_confidence.detach().to(dtype=work_dtype)
    valid = keypoint_valid.detach().to(dtype=torch.bool)
    intrinsics = normalized_intrinsics.detach().to(dtype=work_dtype)
    return joints, uv, confidence, valid, intrinsics


def _bearing_and_weights(
    relative_joints: torch.Tensor,
    keypoint_uv: torch.Tensor,
    keypoint_confidence: torch.Tensor,
    keypoint_valid: torch.Tensor,
    normalized_intrinsics: torch.Tensor,
    *,
    wrist_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    fx, fy, cx, cy = normalized_intrinsics.unbind(dim=-1)
    intrinsics_valid = (
        torch.isfinite(normalized_intrinsics).all(dim=-1)
        & (fx > 1e-6)
        & (fy > 1e-6)
    )
    finite = (
        torch.isfinite(relative_joints).all(dim=-1)
        & torch.isfinite(keypoint_uv).all(dim=-1)
        & torch.isfinite(keypoint_confidence)
    )
    valid = keypoint_valid & finite & intrinsics_valid.unsqueeze(-1)
    safe_uv = torch.where(valid.unsqueeze(-1), keypoint_uv, torch.zeros_like(keypoint_uv))
    bearing = torch.stack(
        (
            (safe_uv[..., 0] - cx.unsqueeze(-1)) / fx.clamp_min(1e-6).unsqueeze(-1),
            (safe_uv[..., 1] - cy.unsqueeze(-1)) / fy.clamp_min(1e-6).unsqueeze(-1),
        ),
        dim=-1,
    )
    weights = torch.where(
        valid,
        torch.nan_to_num(keypoint_confidence, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0),
        torch.zeros_like(keypoint_confidence),
    )
    if weights.shape[-1] > 0:
        wrist_scale = weights.new_ones(weights.shape)
        wrist_scale[..., 0] = float(wrist_weight)
        weights = weights * wrist_scale
    return bearing, weights, valid, intrinsics_valid


@_without_geometry_autocast
def solve_fixed_depth_translation(
    relative_joints: torch.Tensor,
    keypoint_uv: torch.Tensor,
    keypoint_confidence: torch.Tensor,
    keypoint_valid: torch.Tensor,
    normalized_intrinsics: torch.Tensor,
    root_depth: torch.Tensor,
    *,
    wrist_weight: float = 8.0,
    min_valid_joints: int = 2,
    min_joint_depth: float = 1e-4,
) -> HandTranslationSolve:
    joints, uv, confidence, valid, intrinsics = _checked_inputs(
        relative_joints,
        keypoint_uv,
        keypoint_confidence,
        keypoint_valid,
        normalized_intrinsics,
    )
    if root_depth.shape != relative_joints.shape[:-2]:
        raise ValueError("root_depth must align with relative_joints without joint/xyz axes")
    depth = root_depth.to(device=joints.device, dtype=joints.dtype)
    bearing, weights, valid, intrinsics_valid = _bearing_and_weights(
        joints,
        uv,
        confidence,
        valid,
        intrinsics,
        wrist_weight=wrist_weight,
    )
    depth_finite = torch.isfinite(depth)
    safe_depth = torch.where(depth_finite, depth, torch.zeros_like(depth))
    safe_joints = torch.where(valid.unsqueeze(-1), joints, torch.zeros_like(joints))
    joint_depth = safe_joints[..., 2] + safe_depth.unsqueeze(-1)
    valid = (
        valid
        & depth_finite.unsqueeze(-1)
        & torch.isfinite(joint_depth)
        & (joint_depth > float(min_joint_depth))
    )
    weights = torch.where(valid, weights, torch.zeros_like(weights))
    count = (valid & (weights > 0)).sum(dim=-1)
    weight_sum = weights.sum(dim=-1)
    safe_joint_depth = torch.where(valid, joint_depth, torch.ones_like(joint_depth))
    safe_joints = torch.where(valid.unsqueeze(-1), safe_joints, torch.zeros_like(safe_joints))
    vote = safe_joint_depth.unsqueeze(-1) * bearing - safe_joints[..., :2]
    translation_xy = (weights.unsqueeze(-1) * vote).sum(dim=-2) / weight_sum.clamp_min(1e-8).unsqueeze(-1)
    translation = torch.cat((translation_xy, safe_depth.unsqueeze(-1)), dim=-1)
    predicted_bearing = (
        safe_joints[..., :2] + translation_xy.unsqueeze(-2)
    ) / safe_joint_depth.unsqueeze(-1).clamp_min(1e-8)
    residual_per_joint = torch.linalg.vector_norm(predicted_bearing - bearing, dim=-1)
    residual = torch.sqrt(
        (weights * residual_per_joint.square()).sum(dim=-1) / weight_sum.clamp_min(1e-8)
    )
    fx, fy = intrinsics[..., :2].unbind(dim=-1)
    coeff_x = fx.unsqueeze(-1) / safe_joint_depth.clamp_min(1e-8)
    coeff_y = fy.unsqueeze(-1) / safe_joint_depth.clamp_min(1e-8)
    denominator_x = (weights * coeff_x.square()).sum(dim=-1)
    denominator_y = (weights * coeff_y.square()).sum(dim=-1)
    denominator_min = torch.minimum(denominator_x, denominator_y)
    denominator_max = torch.maximum(denominator_x, denominator_y)
    condition = denominator_max / denominator_min.clamp_min(1e-12)
    solve_valid = (
        intrinsics_valid
        & depth_finite
        & (safe_depth > float(min_joint_depth))
        & (count >= int(min_valid_joints))
        & (weight_sum > 0)
        & torch.isfinite(translation).all(dim=-1)
        & torch.isfinite(residual)
        & torch.isfinite(condition)
    )
    translation = torch.where(solve_valid.unsqueeze(-1), translation, torch.zeros_like(translation))
    residual = torch.where(solve_valid, residual, torch.zeros_like(residual))
    condition = torch.where(solve_valid, condition, torch.zeros_like(condition))
    return HandTranslationSolve(
        translation=translation.to(dtype=relative_joints.dtype),
        valid=solve_valid,
        valid_joint_count=count,
        weighted_residual=residual,
        condition_number=condition,
    )


@_without_geometry_autocast
def solve_full_translation(
    relative_joints: torch.Tensor,
    keypoint_uv: torch.Tensor,
    keypoint_confidence: torch.Tensor,
    keypoint_valid: torch.Tensor,
    normalized_intrinsics: torch.Tensor,
    *,
    wrist_weight: float = 8.0,
    min_valid_joints: int = 4,
    damping: float = 1e-6,
    max_condition_number: float = 1e6,
) -> HandTranslationSolve:
    joints, uv, confidence, valid, intrinsics = _checked_inputs(
        relative_joints,
        keypoint_uv,
        keypoint_confidence,
        keypoint_valid,
        normalized_intrinsics,
    )
    bearing, weights, valid, intrinsics_valid = _bearing_and_weights(
        joints,
        uv,
        confidence,
        valid,
        intrinsics,
        wrist_weight=wrist_weight,
    )
    count = (valid & (weights > 0)).sum(dim=-1)
    bx, by = bearing.unbind(dim=-1)
    safe_joints = torch.where(valid.unsqueeze(-1), joints, torch.zeros_like(joints))
    x, y, z = safe_joints.unbind(dim=-1)
    zeros = torch.zeros_like(bx)
    ones = torch.ones_like(bx)
    row_x = torch.stack((ones, zeros, -bx), dim=-1)
    row_y = torch.stack((zeros, ones, -by), dim=-1)
    matrix = torch.stack((row_x, row_y), dim=-2).reshape(*joints.shape[:-2], -1, 3)
    target = torch.stack((bx * z - x, by * z - y), dim=-1).reshape(*joints.shape[:-2], -1)
    row_weights = weights.unsqueeze(-1).expand(*weights.shape, 2).reshape(*weights.shape[:-1], -1)
    with torch.autocast(device_type=joints.device.type, enabled=False):
        weighted_matrix = matrix * row_weights.sqrt().unsqueeze(-1)
        weighted_target = target * row_weights.sqrt()
        normal = weighted_matrix.transpose(-1, -2) @ weighted_matrix
        scale = normal.diagonal(dim1=-2, dim2=-1).mean(dim=-1).clamp_min(1e-8)
        identity = torch.eye(3, device=normal.device, dtype=normal.dtype)
        normal = normal + float(damping) * scale[..., None, None] * identity
        rhs = weighted_matrix.transpose(-1, -2) @ weighted_target.unsqueeze(-1)
        singular_values = torch.linalg.svdvals(normal.detach())
        condition = singular_values[..., 0] / singular_values[..., -1].clamp_min(1e-12)
        pre_solve_valid = (
            torch.isfinite(normal.detach()).all(dim=(-2, -1))
            & torch.isfinite(rhs.detach()).all(dim=(-2, -1))
            & torch.isfinite(condition)
            & (condition <= float(max_condition_number))
        )
        safe_normal = torch.where(
            pre_solve_valid[..., None, None],
            normal,
            identity,
        )
        safe_rhs = torch.where(
            pre_solve_valid[..., None, None],
            rhs,
            torch.zeros_like(rhs),
        )
        solve_result, solve_info = torch.linalg.solve_ex(
            safe_normal,
            safe_rhs,
            check_errors=False,
        )
        translation = solve_result.squeeze(-1)
        predicted = torch.einsum("...rc,...c->...r", matrix, translation)
        weight_sum = row_weights.sum(dim=-1)
        residual = torch.sqrt(
            (row_weights * (predicted - target).square()).sum(dim=-1)
            / weight_sum.clamp_min(1e-8)
        )
    solve_valid = (
        intrinsics_valid
        & (count >= int(min_valid_joints))
        & (weight_sum > 0)
        & pre_solve_valid
        & solve_info.eq(0)
        & torch.isfinite(translation).all(dim=-1)
        & torch.isfinite(residual)
        & torch.isfinite(condition)
        & (condition <= float(max_condition_number))
        & (translation[..., 2] > 0)
    )
    translation = torch.where(solve_valid.unsqueeze(-1), translation, torch.zeros_like(translation))
    residual = torch.where(solve_valid, residual, torch.zeros_like(residual))
    condition = torch.where(solve_valid, condition, torch.zeros_like(condition))
    return HandTranslationSolve(
        translation=translation.to(dtype=relative_joints.dtype),
        valid=solve_valid,
        valid_joint_count=count,
        weighted_residual=residual,
        condition_number=condition,
    )


@_without_geometry_autocast
def solve_wrist_locked_translation(
    relative_joints: torch.Tensor,
    keypoint_uv: torch.Tensor,
    keypoint_confidence: torch.Tensor,
    keypoint_valid: torch.Tensor,
    normalized_intrinsics: torch.Tensor,
    *,
    min_valid_nonroot_joints: int = 3,
    min_depth: float = 1e-4,
    min_denominator: float = 1e-8,
    max_condition_number: float = 1e6,
) -> HandTranslationSolve:
    joints, uv, confidence, valid, intrinsics = _checked_inputs(
        relative_joints,
        keypoint_uv,
        keypoint_confidence,
        keypoint_valid,
        normalized_intrinsics,
    )
    bearing, weights, valid, intrinsics_valid = _bearing_and_weights(
        joints,
        uv,
        confidence,
        valid,
        intrinsics,
        wrist_weight=1.0,
    )
    wrist_valid = valid[..., 0] & (weights[..., 0] > 0)
    wrist_bearing = bearing[..., 0, :]

    safe_joints = torch.where(valid.unsqueeze(-1), joints, torch.zeros_like(joints))
    nonroot_valid = valid & (weights > 0)
    nonroot_valid = nonroot_valid.clone()
    nonroot_valid[..., 0] = False
    pair_weights = weights * weights[..., :1]
    pair_weights = torch.where(
        nonroot_valid & wrist_valid.unsqueeze(-1),
        pair_weights,
        torch.zeros_like(pair_weights),
    )

    bearing_delta = wrist_bearing.unsqueeze(-2) - bearing
    target = bearing * safe_joints[..., 2:].expand_as(bearing) - safe_joints[..., :2]

    def solve_depth(active_weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        active_equation_weights = active_weights.unsqueeze(-1).expand_as(bearing_delta)
        active_denominator = (
            active_equation_weights * bearing_delta.square()
        ).sum(dim=(-2, -1))
        active_numerator = (
            active_equation_weights * bearing_delta * target
        ).sum(dim=(-2, -1))
        active_depth = active_numerator / active_denominator.clamp_min(
            float(min_denominator)
        )
        return active_depth, active_denominator, active_equation_weights

    active_weights = pair_weights
    for _ in range(3):
        depth, _, _ = solve_depth(active_weights)
        candidate_joint_depth = safe_joints[..., 2] + depth.unsqueeze(-1)
        positive_joint = torch.isfinite(candidate_joint_depth) & (
            candidate_joint_depth > float(min_depth)
        )
        active_weights = torch.where(
            nonroot_valid & positive_joint,
            pair_weights,
            torch.zeros_like(pair_weights),
        )
    depth, denominator, equation_weights = solve_depth(active_weights)
    translation = torch.cat(
        (wrist_bearing * depth.unsqueeze(-1), depth.unsqueeze(-1)),
        dim=-1,
    )

    joint_depth = safe_joints[..., 2] + depth.unsqueeze(-1)
    positive_joint = torch.isfinite(joint_depth) & (joint_depth > float(min_depth))
    residual_valid = nonroot_valid & positive_joint
    residual_weights = torch.where(
        residual_valid,
        pair_weights,
        torch.zeros_like(pair_weights),
    )
    predicted_bearing = (
        safe_joints[..., :2] + translation[..., None, :2]
    ) / joint_depth.unsqueeze(-1).clamp_min(float(min_depth))
    residual_per_joint = torch.linalg.vector_norm(predicted_bearing - bearing, dim=-1)
    residual_weight_sum = residual_weights.sum(dim=-1)
    residual = torch.sqrt(
        (residual_weights * residual_per_joint.square()).sum(dim=-1)
        / residual_weight_sum.clamp_min(1e-8)
    )
    valid_nonroot_count = (residual_weights > 0).sum(dim=-1)
    condition = equation_weights.sum(dim=(-2, -1)) / denominator.clamp_min(
        float(min_denominator)
    )
    solve_valid = (
        intrinsics_valid
        & wrist_valid
        & (valid_nonroot_count >= int(min_valid_nonroot_joints))
        & (denominator > float(min_denominator))
        & torch.isfinite(depth)
        & (depth > float(min_depth))
        & torch.isfinite(translation).all(dim=-1)
        & torch.isfinite(residual)
        & torch.isfinite(condition)
        & (condition <= float(max_condition_number))
        & (residual_weight_sum > 0)
    )
    translation = torch.where(
        solve_valid.unsqueeze(-1),
        translation,
        torch.zeros_like(translation),
    )
    residual = torch.where(solve_valid, residual, torch.zeros_like(residual))
    condition = torch.where(solve_valid, condition, torch.zeros_like(condition))
    return HandTranslationSolve(
        translation=translation.to(dtype=relative_joints.dtype),
        valid=solve_valid,
        valid_joint_count=valid_nonroot_count + wrist_valid.to(dtype=torch.long),
        weighted_residual=residual,
        condition_number=condition,
    )


def wrist_ray_translation(
    wrist_uv: torch.Tensor,
    wrist_valid: torch.Tensor,
    normalized_intrinsics: torch.Tensor,
    root_depth: torch.Tensor,
    *,
    min_depth: float = 1e-4,
) -> HandTranslationSolve:
    if wrist_uv.shape[-1] != 2 or wrist_valid.shape != wrist_uv.shape[:-1]:
        raise ValueError("wrist UV/validity shapes are incompatible")
    if normalized_intrinsics.shape != (*wrist_uv.shape[:-1], 4):
        raise ValueError("normalized_intrinsics must align with wrist UV")
    if root_depth.shape != wrist_valid.shape:
        raise ValueError("root_depth must align with wrist validity")
    dtype = torch.float64 if wrist_uv.dtype == torch.float64 else torch.float32
    uv = wrist_uv.detach().to(dtype=dtype)
    intrinsics = normalized_intrinsics.detach().to(dtype=dtype)
    depth = root_depth.to(dtype=dtype)
    fx, fy, cx, cy = intrinsics.unbind(dim=-1)
    valid = (
        wrist_valid.detach().to(dtype=torch.bool)
        & torch.isfinite(uv).all(dim=-1)
        & torch.isfinite(intrinsics).all(dim=-1)
        & torch.isfinite(depth)
        & (fx > 1e-6)
        & (fy > 1e-6)
        & (depth > float(min_depth))
    )
    translation = torch.stack(
        (
            depth * (uv[..., 0] - cx) / fx.clamp_min(1e-6),
            depth * (uv[..., 1] - cy) / fy.clamp_min(1e-6),
            depth,
        ),
        dim=-1,
    )
    translation = torch.where(valid.unsqueeze(-1), translation, torch.zeros_like(translation))
    count = valid.to(dtype=torch.long)
    return HandTranslationSolve(
        translation=translation.to(dtype=wrist_uv.dtype),
        valid=valid,
        valid_joint_count=count,
        weighted_residual=translation.new_zeros(valid.shape),
        condition_number=translation.new_ones(valid.shape),
    )


def _blend_analytic_and_prior_depth(
    analytic: HandTranslationSolve,
    root_depth_prior: torch.Tensor,
    *,
    residual_scale: float,
    min_depth: float,
    quality_low: float = 0.40,
    quality_high: float = 0.65,
    support_saturation: float = 4.0,
    condition_good: float = 100.0,
    condition_max: float = 1e6,
) -> RootDepthFusion:
    if root_depth_prior.shape != analytic.valid.shape:
        raise ValueError("root_depth_prior must align with the analytic solve prefix")
    if residual_scale <= 0:
        raise ValueError("residual_scale must be positive")
    if not 0.0 <= float(quality_low) < float(quality_high) <= 1.0:
        raise ValueError("quality thresholds must satisfy 0 <= low < high <= 1")
    if not math.isfinite(float(support_saturation)) or float(support_saturation) <= 0.0:
        raise ValueError("support_saturation must be finite and positive")
    if not 1.0 <= float(condition_good) < float(condition_max):
        raise ValueError("condition thresholds must satisfy 1 <= good < max")
    work_dtype = torch.float64 if root_depth_prior.dtype == torch.float64 else torch.float32
    prior = root_depth_prior.to(dtype=work_dtype)
    analytic_depth = analytic.translation[..., 2].to(dtype=work_dtype)
    analytic_valid = (
        analytic.valid
        & torch.isfinite(analytic_depth)
        & (analytic_depth > float(min_depth))
    )
    prior_valid = torch.isfinite(prior) & (prior > float(min_depth))
    residual = torch.nan_to_num(
        analytic.weighted_residual.detach().to(dtype=work_dtype),
        nan=float("inf"),
        posinf=float("inf"),
        neginf=float("inf"),
    )
    residual_quality = torch.exp(-float(residual_scale) * residual).clamp(0.0, 1.0)
    support_quality = (
        analytic.valid_joint_count.detach().to(dtype=work_dtype) / float(support_saturation)
    ).clamp(0.0, 1.0)
    safe_prior_for_ratio = prior.clamp_min(float(min_depth))
    safe_analytic_for_ratio = analytic_depth.clamp_min(float(min_depth))
    depth_ratio = torch.minimum(
        safe_analytic_for_ratio / safe_prior_for_ratio,
        safe_prior_for_ratio / safe_analytic_for_ratio,
    ).clamp(0.0, 1.0)
    prior_agreement = torch.where(
        analytic_valid & prior_valid,
        depth_ratio,
        torch.ones_like(depth_ratio),
    ).detach()
    condition = torch.nan_to_num(
        analytic.condition_number.detach().to(dtype=work_dtype),
        nan=float("inf"),
        posinf=float("inf"),
        neginf=float("inf"),
    ).clamp_min(1.0)
    condition_quality = (
        torch.log(condition.new_tensor(float(condition_max)) / condition)
        / math.log(float(condition_max) / float(condition_good))
    ).clamp(0.0, 1.0)
    condition_quality = torch.where(
        analytic_valid,
        condition_quality,
        torch.zeros_like(condition_quality),
    ).detach()
    geometric_quality = residual_quality * torch.sqrt(support_quality) * condition_quality
    geometric_quality = torch.where(
        analytic_valid,
        geometric_quality,
        torch.zeros_like(geometric_quality),
    ).detach()
    safe_prior = torch.where(prior_valid, prior, torch.zeros_like(prior))
    safe_analytic = torch.where(
        analytic_valid,
        analytic_depth,
        torch.zeros_like(analytic_depth),
    )
    both_valid = analytic_valid & prior_valid
    transition = (
        (geometric_quality - float(quality_low))
        / float(quality_high - quality_low)
    ).clamp(0.0, 1.0)
    smooth_weight = transition.square() * (3.0 - 2.0 * transition)
    analytic_weight = torch.where(
        both_valid,
        smooth_weight,
        analytic_valid.to(dtype=geometric_quality.dtype),
    ).detach()
    depth = torch.where(
        both_valid,
        analytic_weight * safe_analytic + (1.0 - analytic_weight) * safe_prior,
        torch.where(analytic_valid, safe_analytic, safe_prior),
    )
    depth_valid = analytic_valid | prior_valid
    depth = torch.where(depth_valid, depth, torch.zeros_like(depth))
    return RootDepthFusion(
        depth=depth.to(dtype=root_depth_prior.dtype),
        analytic_weight=analytic_weight.to(dtype=root_depth_prior.dtype),
        analytic_valid=analytic_valid,
        geometric_quality=geometric_quality.to(dtype=root_depth_prior.dtype),
        condition_quality=condition_quality.to(dtype=root_depth_prior.dtype),
        prior_agreement=prior_agreement.to(dtype=root_depth_prior.dtype),
    )


def _translation_reprojection_diagnostics(
    relative_joints: torch.Tensor,
    keypoint_uv: torch.Tensor,
    keypoint_confidence: torch.Tensor,
    keypoint_valid: torch.Tensor,
    normalized_intrinsics: torch.Tensor,
    translation: torch.Tensor,
    translation_valid: torch.Tensor,
    *,
    min_depth: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    joints, uv, confidence, valid, intrinsics = _checked_inputs(
        relative_joints,
        keypoint_uv,
        keypoint_confidence,
        keypoint_valid,
        normalized_intrinsics,
    )
    bearing, weights, valid, _ = _bearing_and_weights(
        joints,
        uv,
        confidence,
        valid,
        intrinsics,
        wrist_weight=1.0,
    )
    candidate = translation.to(dtype=joints.dtype)
    candidate_finite = torch.isfinite(candidate).all(dim=-1) & translation_valid
    safe_candidate = torch.where(
        candidate_finite.unsqueeze(-1),
        candidate,
        torch.zeros_like(candidate),
    )
    safe_joints = torch.where(valid.unsqueeze(-1), joints, torch.zeros_like(joints))
    joint_depth = safe_joints[..., 2] + safe_candidate[..., None, 2]
    active = (
        valid
        & (weights > 0)
        & candidate_finite.unsqueeze(-1)
        & torch.isfinite(joint_depth)
        & (joint_depth > float(min_depth))
    )
    safe_depth = torch.where(active, joint_depth, torch.ones_like(joint_depth))
    predicted_bearing = (
        safe_joints[..., :2] + safe_candidate[..., None, :2]
    ) / safe_depth.unsqueeze(-1).clamp_min(float(min_depth))
    per_joint = torch.linalg.vector_norm(predicted_bearing - bearing, dim=-1)
    active_weights = torch.where(active, weights, torch.zeros_like(weights))
    weight_sum = active_weights.sum(dim=-1)
    residual = torch.sqrt(
        (active_weights * per_joint.square()).sum(dim=-1)
        / weight_sum.clamp_min(1e-8)
    )
    count = (active_weights > 0).sum(dim=-1)
    residual = torch.where(
        candidate_finite & (count > 0) & torch.isfinite(residual),
        residual,
        torch.zeros_like(residual),
    )
    return residual.to(dtype=relative_joints.dtype), count


def _per_joint_reprojection_residual(
    relative_joints: torch.Tensor,
    keypoint_uv: torch.Tensor,
    keypoint_confidence: torch.Tensor,
    keypoint_valid: torch.Tensor,
    normalized_intrinsics: torch.Tensor,
    translation: torch.Tensor,
    translation_valid: torch.Tensor,
    *,
    min_depth: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    joints, uv, confidence, valid, intrinsics = _checked_inputs(
        relative_joints,
        keypoint_uv,
        keypoint_confidence,
        keypoint_valid,
        normalized_intrinsics,
    )
    if translation.shape != (*joints.shape[:-2], 3):
        raise ValueError("translation must align with the solver prefix")
    if translation_valid.shape != joints.shape[:-2]:
        raise ValueError("translation_valid must align with the solver prefix")
    _, weights, valid, _ = _bearing_and_weights(
        joints,
        uv,
        confidence,
        valid,
        intrinsics,
        wrist_weight=1.0,
    )
    candidate = translation.to(dtype=joints.dtype)
    candidate_valid = translation_valid.detach().to(dtype=torch.bool) & torch.isfinite(
        candidate
    ).all(dim=-1)
    safe_candidate = torch.where(
        candidate_valid.unsqueeze(-1),
        candidate,
        torch.zeros_like(candidate),
    )
    safe_joints = torch.where(valid.unsqueeze(-1), joints, torch.zeros_like(joints))
    camera_xyz = safe_joints + safe_candidate.unsqueeze(-2)
    active = (
        valid
        & candidate_valid.unsqueeze(-1)
        & torch.isfinite(camera_xyz).all(dim=-1)
        & (camera_xyz[..., 2] > float(min_depth))
        & (weights > 0)
    )
    safe_depth = torch.where(active, camera_xyz[..., 2], torch.ones_like(camera_xyz[..., 2]))
    fx, fy, cx, cy = intrinsics.unbind(dim=-1)
    predicted_uv = torch.stack(
        (
            fx.unsqueeze(-1) * camera_xyz[..., 0] / safe_depth + cx.unsqueeze(-1),
            fy.unsqueeze(-1) * camera_xyz[..., 1] / safe_depth + cy.unsqueeze(-1),
        ),
        dim=-1,
    )
    residual = torch.linalg.vector_norm(predicted_uv - uv, dim=-1)
    residual = torch.where(active & torch.isfinite(residual), residual, torch.zeros_like(residual))
    return residual, active, weights


@torch.no_grad()
def _one_pass_huber_confidence(
    relative_joints: torch.Tensor,
    keypoint_uv: torch.Tensor,
    keypoint_confidence: torch.Tensor,
    keypoint_valid: torch.Tensor,
    normalized_intrinsics: torch.Tensor,
    initial: HandTranslationSolve,
    *,
    huber_delta: float,
    min_depth: float,
) -> torch.Tensor:
    if not math.isfinite(float(huber_delta)) or float(huber_delta) <= 0.0:
        raise ValueError("huber_delta must be finite and positive")
    residual, active, _ = _per_joint_reprojection_residual(
        relative_joints,
        keypoint_uv,
        keypoint_confidence,
        keypoint_valid,
        normalized_intrinsics,
        initial.translation,
        initial.valid,
        min_depth=float(min_depth),
    )
    scale = torch.minimum(
        torch.ones_like(residual),
        residual.new_tensor(float(huber_delta)) / residual.clamp_min(1e-8),
    )
    scale = torch.where(
        initial.valid.unsqueeze(-1) & active,
        scale,
        torch.ones_like(scale),
    ).detach()
    return keypoint_confidence.detach() * scale.to(dtype=keypoint_confidence.dtype)


def _bounded_wrist_consensus_translation(
    relative_joints: torch.Tensor,
    keypoint_uv: torch.Tensor,
    keypoint_confidence: torch.Tensor,
    keypoint_valid: torch.Tensor,
    normalized_intrinsics: torch.Tensor,
    wrist_translation: HandTranslationSolve,
    *,
    image_size_hw: tuple[int, int],
    min_nonwrist_joints: int = 3,
    huber_delta: float = 0.02,
    wrist_hard_lock_confidence: float = 0.8,
    wrist_min_confidence: float = 0.25,
    nonwrist_correction_strength: float = 0.25,
    max_correction_short_side_fraction: float = 0.0125,
    min_depth: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    image_height, image_width = (int(image_size_hw[0]), int(image_size_hw[1]))
    if image_height <= 0 or image_width <= 0:
        raise ValueError("image_size_hw must contain positive integers")
    if not 0.0 <= float(wrist_min_confidence) < float(wrist_hard_lock_confidence) <= 1.0:
        raise ValueError("wrist confidence thresholds are invalid")
    if int(min_nonwrist_joints) <= 0:
        raise ValueError("min_nonwrist_joints must be positive")
    if not math.isfinite(float(max_correction_short_side_fraction)) or float(
        max_correction_short_side_fraction
    ) <= 0.0:
        raise ValueError("max correction fraction must be finite and positive")
    if not math.isfinite(float(nonwrist_correction_strength)) or not (
        0.0 <= float(nonwrist_correction_strength) <= 1.0
    ):
        raise ValueError("nonwrist correction strength must be finite and inside [0, 1]")

    joints, uv, confidence, valid, intrinsics = _checked_inputs(
        relative_joints,
        keypoint_uv,
        keypoint_confidence,
        keypoint_valid,
        normalized_intrinsics,
    )
    bearing, weights, valid, intrinsics_valid = _bearing_and_weights(
        joints,
        uv,
        confidence,
        valid,
        intrinsics,
        wrist_weight=8.0,
    )
    root = wrist_translation.translation.to(dtype=joints.dtype)
    root_valid = wrist_translation.valid & torch.isfinite(root).all(dim=-1)
    depth = root[..., 2]
    safe_joints = torch.where(valid.unsqueeze(-1), joints, torch.zeros_like(joints))
    joint_depth = safe_joints[..., 2] + depth.unsqueeze(-1)
    active = (
        valid
        & root_valid.unsqueeze(-1)
        & torch.isfinite(joint_depth)
        & (joint_depth > float(min_depth))
        & (weights > 0)
    )
    nonwrist_active = active.clone()
    nonwrist_active[..., 0] = False
    nonwrist_count = nonwrist_active.sum(dim=-1)
    active_weights = torch.where(active, weights, torch.zeros_like(weights))
    votes = joint_depth.unsqueeze(-1) * bearing - safe_joints[..., :2]

    def weighted_xy(candidate_weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weight_sum = candidate_weights.sum(dim=-1)
        xy = (candidate_weights.unsqueeze(-1) * votes).sum(dim=-2) / weight_sum.clamp_min(
            1e-8
        ).unsqueeze(-1)
        return xy, weight_sum

    initial_xy, initial_weight_sum = weighted_xy(active_weights)
    initial_translation = torch.cat((initial_xy, depth.unsqueeze(-1)), dim=-1)
    initial_valid = (
        root_valid
        & intrinsics_valid
        & active[..., 0]
        & (nonwrist_count >= int(min_nonwrist_joints))
        & (initial_weight_sum > 0)
        & torch.isfinite(initial_translation).all(dim=-1)
    )
    with torch.no_grad():
        initial_residual, initial_active, _ = _per_joint_reprojection_residual(
            relative_joints,
            keypoint_uv,
            keypoint_confidence,
            keypoint_valid,
            normalized_intrinsics,
            initial_translation.to(dtype=relative_joints.dtype),
            initial_valid,
            min_depth=float(min_depth),
        )
    huber_scale = torch.minimum(
        torch.ones_like(initial_residual),
        initial_residual.new_tensor(float(huber_delta)) / initial_residual.clamp_min(1e-8),
    ).detach()
    robust_weights = torch.where(
        initial_active,
        active_weights * huber_scale,
        torch.zeros_like(active_weights),
    )
    consensus_xy, robust_weight_sum = weighted_xy(robust_weights)
    consensus_translation = torch.cat((consensus_xy, depth.unsqueeze(-1)), dim=-1)
    with torch.no_grad():
        final_residual_per_joint, final_active, _ = _per_joint_reprojection_residual(
            relative_joints,
            keypoint_uv,
            keypoint_confidence,
            keypoint_valid,
            normalized_intrinsics,
            consensus_translation.to(dtype=relative_joints.dtype),
            initial_valid,
            min_depth=float(min_depth),
        )
    final_weights = torch.where(final_active, robust_weights, torch.zeros_like(robust_weights))
    final_weight_sum = final_weights.sum(dim=-1)
    consensus_residual = torch.sqrt(
        (final_weights * final_residual_per_joint.square()).sum(dim=-1)
        / final_weight_sum.clamp_min(1e-8)
    )

    fx, fy, cx, cy = intrinsics.unbind(dim=-1)
    safe_depth = depth.clamp_min(float(min_depth))
    consensus_uv = torch.stack(
        (
            fx * consensus_xy[..., 0] / safe_depth + cx,
            fy * consensus_xy[..., 1] / safe_depth + cy,
        ),
        dim=-1,
    )
    wrist_uv = uv[..., 0, :]
    pixel_scale = root.new_tensor(
        (float(max(image_width - 1, 1)), float(max(image_height - 1, 1)))
    )
    raw_delta_px = (consensus_uv - wrist_uv) * pixel_scale
    raw_delta_norm = torch.linalg.vector_norm(raw_delta_px, dim=-1)
    correction_cap = float(max_correction_short_side_fraction) * float(
        min(image_height, image_width)
    )
    cap_scale = torch.minimum(
        torch.ones_like(raw_delta_norm),
        raw_delta_norm.new_tensor(correction_cap) / raw_delta_norm.clamp_min(1e-8),
    ).detach()
    clipped_delta_px = raw_delta_px * cap_scale.unsqueeze(-1)

    wrist_confidence = confidence[..., 0].clamp(0.0, 1.0)
    q_wrist = (
        (float(wrist_hard_lock_confidence) - wrist_confidence)
        / float(wrist_hard_lock_confidence - wrist_min_confidence)
    ).clamp(0.0, 1.0)
    q_support = (nonwrist_count.to(dtype=joints.dtype) / 4.0).clamp(0.0, 1.0)
    q_residual = torch.exp(-consensus_residual / float(huber_delta)).clamp(0.0, 1.0)
    eligible = (
        initial_valid
        & (robust_weight_sum > 0)
        & (final_weight_sum > 0)
        & torch.isfinite(consensus_translation).all(dim=-1)
        & torch.isfinite(consensus_residual)
        & torch.isfinite(raw_delta_px).all(dim=-1)
        & (wrist_confidence >= float(wrist_min_confidence))
        & (wrist_confidence < float(wrist_hard_lock_confidence))
    )
    gate = torch.where(
        eligible,
        float(nonwrist_correction_strength) * q_wrist * q_support * q_residual,
        torch.zeros_like(q_wrist),
    ).detach()
    applied_delta_px = gate.unsqueeze(-1) * clipped_delta_px
    corrected_uv = wrist_uv + applied_delta_px / pixel_scale
    corrected_bearing = torch.stack(
        (
            (corrected_uv[..., 0] - cx) / fx.clamp_min(1e-6),
            (corrected_uv[..., 1] - cy) / fy.clamp_min(1e-6),
        ),
        dim=-1,
    )
    corrected_translation = torch.cat(
        (corrected_bearing * depth.unsqueeze(-1), depth.unsqueeze(-1)),
        dim=-1,
    )
    correction_norm = torch.linalg.vector_norm(applied_delta_px, dim=-1)
    corrected = eligible & (correction_norm > 0) & torch.isfinite(corrected_translation).all(dim=-1)
    translation = torch.where(corrected.unsqueeze(-1), corrected_translation, root)
    return (
        translation.to(dtype=relative_joints.dtype),
        corrected,
        torch.where(eligible, nonwrist_count, torch.zeros_like(nonwrist_count)),
        gate.to(dtype=relative_joints.dtype),
        torch.where(corrected, correction_norm, torch.zeros_like(correction_norm)).to(
            dtype=relative_joints.dtype
        ).detach(),
        torch.where(eligible, consensus_residual, torch.zeros_like(consensus_residual)).to(
            dtype=relative_joints.dtype
        ).detach(),
    )


@_without_geometry_autocast
def solve_partial_observation_translation(
    relative_joints: torch.Tensor,
    keypoint_uv: torch.Tensor,
    keypoint_confidence: torch.Tensor,
    keypoint_valid: torch.Tensor,
    normalized_intrinsics: torch.Tensor,
    root_depth_prior: torch.Tensor,
    *,
    analytic_residual_scale: float = 10.0,
    robust_huber_delta: float = 0.02,
    wrist_min_valid_nonroot_joints: int = 3,
    nonwrist_min_analytic_joints: int = 2,
    wrist_nonwrist_correction_strength: float = 0.25,
    min_depth: float = 1e-4,
    image_size_hw: tuple[int, int] | None = None,
) -> PartialHandTranslationSolve:
    relative_joints, keypoint_uv, keypoint_confidence, keypoint_valid, normalized_intrinsics = _checked_inputs(
        relative_joints, keypoint_uv, keypoint_confidence, keypoint_valid, normalized_intrinsics,
    )
    root_depth_prior = root_depth_prior.to(dtype=relative_joints.dtype)

    expected_prefix = relative_joints.shape[:-2]
    if root_depth_prior.shape != expected_prefix:
        raise ValueError("root_depth_prior must align with relative_joints without joint/xyz axes")
    if relative_joints.shape[-2] < 1:
        raise ValueError("partial-observation translation requires at least one joint slot")

    with torch.no_grad():
        wrist_initial = solve_wrist_locked_translation(
            relative_joints,
            keypoint_uv,
            keypoint_confidence,
            keypoint_valid,
            normalized_intrinsics,
            min_valid_nonroot_joints=int(wrist_min_valid_nonroot_joints),
            min_depth=float(min_depth),
        )
    wrist_robust_confidence = _one_pass_huber_confidence(
        relative_joints,
        keypoint_uv,
        keypoint_confidence,
        keypoint_valid,
        normalized_intrinsics,
        wrist_initial,
        huber_delta=float(robust_huber_delta),
        min_depth=float(min_depth),
    )
    wrist_analytic = solve_wrist_locked_translation(
        relative_joints,
        keypoint_uv,
        wrist_robust_confidence,
        keypoint_valid,
        normalized_intrinsics,
        min_valid_nonroot_joints=int(wrist_min_valid_nonroot_joints),
        min_depth=float(min_depth),
    )
    wrist_fusion = _blend_analytic_and_prior_depth(
        wrist_analytic,
        root_depth_prior,
        residual_scale=float(analytic_residual_scale),
        min_depth=float(min_depth),
    )
    wrist_depth = wrist_fusion.depth
    wrist_blend = wrist_fusion.analytic_weight
    wrist_analytic_valid = wrist_fusion.analytic_valid
    wrist_condition_quality = wrist_fusion.condition_quality
    wrist_solution = wrist_ray_translation(
        keypoint_uv[..., 0, :],
        keypoint_valid[..., 0],
        normalized_intrinsics,
        wrist_depth,
        min_depth=float(min_depth),
    )
    if image_size_hw is None:
        wrist_translation = wrist_solution.translation
        wrist_corrected = torch.zeros_like(wrist_solution.valid)
        xy_consensus_count = torch.zeros_like(wrist_solution.valid_joint_count)
        xy_consensus_gate = wrist_depth.new_zeros(wrist_depth.shape)
        xy_consensus_correction_px = wrist_depth.new_zeros(wrist_depth.shape)
        xy_consensus_residual = wrist_depth.new_zeros(wrist_depth.shape)
    else:
        (
            wrist_translation,
            wrist_corrected,
            xy_consensus_count,
            xy_consensus_gate,
            xy_consensus_correction_px,
            xy_consensus_residual,
        ) = _bounded_wrist_consensus_translation(
            relative_joints,
            keypoint_uv,
            keypoint_confidence,
            keypoint_valid,
            normalized_intrinsics,
            wrist_solution,
            image_size_hw=image_size_hw,
            min_nonwrist_joints=int(wrist_min_valid_nonroot_joints),
            huber_delta=float(robust_huber_delta),
            nonwrist_correction_strength=float(wrist_nonwrist_correction_strength),
            min_depth=float(min_depth),
        )

    prior_work = root_depth_prior.to(
        dtype=torch.float64 if root_depth_prior.dtype == torch.float64 else torch.float32
    )
    relative_z_work = relative_joints[..., 2].to(dtype=prior_work.dtype)
    prior_valid = torch.isfinite(prior_work) & (prior_work > float(min_depth))
    prior_joint_depth = relative_z_work + prior_work.unsqueeze(-1)
    nonwrist_analytic_joint_valid = keypoint_valid & (
        ~prior_valid.unsqueeze(-1)
        | (torch.isfinite(prior_joint_depth) & (prior_joint_depth > float(min_depth)))
    )
    with torch.no_grad():
        full_initial = solve_full_translation(
            relative_joints,
            keypoint_uv,
            keypoint_confidence,
            nonwrist_analytic_joint_valid,
            normalized_intrinsics,
            min_valid_joints=int(nonwrist_min_analytic_joints),
            damping=1e-10,
        )
    full_robust_confidence = _one_pass_huber_confidence(
        relative_joints,
        keypoint_uv,
        keypoint_confidence,
        nonwrist_analytic_joint_valid,
        normalized_intrinsics,
        full_initial,
        huber_delta=float(robust_huber_delta),
        min_depth=float(min_depth),
    )
    full_analytic_raw = solve_full_translation(
        relative_joints,
        keypoint_uv,
        full_robust_confidence,
        nonwrist_analytic_joint_valid,
        normalized_intrinsics,
        min_valid_joints=int(nonwrist_min_analytic_joints),
        damping=1e-10,
    )
    analytic_joint_depth = (
        relative_z_work + full_analytic_raw.translation[..., 2].to(dtype=prior_work.dtype).unsqueeze(-1)
    )
    analytic_positive = (
        nonwrist_analytic_joint_valid
        & torch.isfinite(analytic_joint_depth)
        & (analytic_joint_depth > float(min_depth))
        & torch.isfinite(keypoint_confidence)
        & (keypoint_confidence > 0)
    )
    analytic_positive_count = analytic_positive.sum(dim=-1)
    full_analytic_valid = full_analytic_raw.valid & (
        analytic_positive_count >= int(nonwrist_min_analytic_joints)
    )
    full_analytic = HandTranslationSolve(
        translation=torch.where(
            full_analytic_valid.unsqueeze(-1),
            full_analytic_raw.translation,
            torch.zeros_like(full_analytic_raw.translation),
        ),
        valid=full_analytic_valid,
        valid_joint_count=analytic_positive_count,
        weighted_residual=torch.where(
            full_analytic_valid,
            full_analytic_raw.weighted_residual,
            torch.zeros_like(full_analytic_raw.weighted_residual),
        ),
        condition_number=torch.where(
            full_analytic_valid,
            full_analytic_raw.condition_number,
            torch.zeros_like(full_analytic_raw.condition_number),
        ),
    )
    full_fusion = _blend_analytic_and_prior_depth(
        full_analytic,
        root_depth_prior,
        residual_scale=float(analytic_residual_scale),
        min_depth=float(min_depth),
    )
    full_depth = full_fusion.depth
    full_blend = full_fusion.analytic_weight
    full_analytic_valid = full_fusion.analytic_valid
    full_condition_quality = full_fusion.condition_quality
    partial_solution = solve_fixed_depth_translation(
        relative_joints,
        keypoint_uv,
        keypoint_confidence,
        keypoint_valid,
        normalized_intrinsics,
        full_depth,
        wrist_weight=1.0,
        min_valid_joints=1,
        min_joint_depth=float(min_depth),
    )

    use_wrist = wrist_solution.valid
    use_partial = ~use_wrist & partial_solution.valid
    valid = use_wrist | use_partial
    translation = torch.where(
        use_wrist.unsqueeze(-1),
        wrist_translation,
        partial_solution.translation,
    )
    translation = torch.where(valid.unsqueeze(-1), translation, torch.zeros_like(translation))

    selected_analytic = torch.where(
        use_wrist.unsqueeze(-1),
        wrist_analytic.translation,
        full_analytic.translation,
    )
    selected_analytic_valid = torch.where(
        use_wrist,
        wrist_analytic_valid,
        full_analytic_valid,
    ) & valid
    selected_blend = torch.where(use_wrist, wrist_blend, full_blend)
    selected_blend = torch.where(valid, selected_blend, torch.zeros_like(selected_blend))
    selected_condition_quality = torch.where(
        use_wrist,
        wrist_condition_quality,
        full_condition_quality,
    )
    selected_condition_quality = torch.where(
        valid,
        selected_condition_quality,
        torch.zeros_like(selected_condition_quality),
    )
    selected_geometric_quality = torch.where(
        use_wrist,
        wrist_fusion.geometric_quality,
        full_fusion.geometric_quality,
    )
    selected_geometric_quality = torch.where(
        valid,
        selected_geometric_quality,
        torch.zeros_like(selected_geometric_quality),
    )
    selected_prior_agreement = torch.where(
        use_wrist,
        wrist_fusion.prior_agreement,
        full_fusion.prior_agreement,
    )
    selected_prior_agreement = torch.where(
        valid,
        selected_prior_agreement,
        torch.zeros_like(selected_prior_agreement),
    )
    selected_analytic_residual = torch.where(
        use_wrist & wrist_analytic_valid,
        wrist_analytic.weighted_residual,
        torch.where(
            use_partial & full_analytic_valid,
            full_analytic.weighted_residual,
            torch.zeros_like(full_analytic.weighted_residual),
        ),
    ).detach()
    selected_analytic_support = torch.where(
        use_wrist & wrist_analytic_valid,
        wrist_analytic.valid_joint_count,
        torch.where(
            use_partial & full_analytic_valid,
            full_analytic.valid_joint_count,
            torch.zeros_like(full_analytic.valid_joint_count),
        ),
    ).detach()
    selected_condition = torch.where(
        use_wrist & wrist_analytic_valid,
        wrist_analytic.condition_number,
        torch.where(
            use_partial & full_analytic_valid,
            full_analytic.condition_number,
            torch.ones_like(full_analytic.condition_number),
        ),
    )
    selected_condition = torch.where(
        valid,
        selected_condition,
        torch.zeros_like(selected_condition),
    )
    residual, observed_count = _translation_reprojection_diagnostics(
        relative_joints,
        keypoint_uv,
        keypoint_confidence,
        keypoint_valid,
        normalized_intrinsics,
        translation,
        valid,
        min_depth=float(min_depth),
    )

    mode = torch.full(expected_prefix, int(RootTranslationMode.INVALID), device=valid.device, dtype=torch.long)
    mode = torch.where(
        use_partial,
        torch.full_like(mode, int(RootTranslationMode.NONWRIST_PRIOR)),
        mode,
    )
    mode = torch.where(
        use_partial & full_analytic_valid,
        torch.full_like(mode, int(RootTranslationMode.NONWRIST_ANALYTIC_BLEND)),
        mode,
    )
    mode = torch.where(
        use_wrist,
        torch.full_like(mode, int(RootTranslationMode.WRIST_PRIOR)),
        mode,
    )
    mode = torch.where(
        use_wrist & wrist_analytic_valid,
        torch.full_like(mode, int(RootTranslationMode.WRIST_ANALYTIC_BLEND)),
        mode,
    )
    mode = torch.where(
        use_wrist & wrist_corrected,
        torch.full_like(mode, int(RootTranslationMode.WRIST_CORRECTED)),
        mode,
    )
    return PartialHandTranslationSolve(
        translation=translation,
        valid=valid,
        analytic_translation=selected_analytic,
        analytic_valid=selected_analytic_valid,
        valid_joint_count=observed_count,
        weighted_residual=residual,
        condition_number=selected_condition,
        depth_blend_weight=selected_blend,
        depth_condition_quality=selected_condition_quality,
        depth_geometric_quality=selected_geometric_quality,
        depth_prior_agreement=selected_prior_agreement,
        analytic_weighted_residual=selected_analytic_residual,
        analytic_support_count=selected_analytic_support,
        mode=mode,
        wrist_used=use_wrist,
        xy_consensus_valid_count=torch.where(
            use_wrist,
            xy_consensus_count,
            torch.zeros_like(xy_consensus_count),
        ),
        xy_consensus_gate=torch.where(
            use_wrist,
            xy_consensus_gate,
            torch.zeros_like(xy_consensus_gate),
        ),
        xy_consensus_correction_px=torch.where(
            use_wrist,
            xy_consensus_correction_px,
            torch.zeros_like(xy_consensus_correction_px),
        ),
        xy_consensus_residual=torch.where(
            use_wrist,
            xy_consensus_residual,
            torch.zeros_like(xy_consensus_residual),
        ),
        xy_consensus_corrected=use_wrist & wrist_corrected,
    )


@torch.no_grad()
@_without_geometry_autocast
def root_subset_reliability(
    relative_joints: torch.Tensor,
    keypoint_uv: torch.Tensor,
    keypoint_confidence: torch.Tensor,
    keypoint_valid: torch.Tensor,
    normalized_intrinsics: torch.Tensor,
    reference: PartialHandTranslationSolve,
    *,
    relative_scale: float = 0.05,
    min_depth: float = 1e-4,
) -> RootSubsetReliability:
    if not math.isfinite(relative_scale) or relative_scale <= 0:
        raise ValueError("subset relative scale must be positive finite")
    joints, uv, confidence, valid, K = _checked_inputs(
        relative_joints, keypoint_uv, keypoint_confidence, keypoint_valid, normalized_intrinsics,
    )
    initial = HandTranslationSolve(
        reference.analytic_translation, reference.analytic_valid,
        reference.analytic_support_count, reference.analytic_weighted_residual,
        reference.condition_number,
    )
    robust = _one_pass_huber_confidence(
        joints, uv, confidence, valid, K, initial, huber_delta=0.02, min_depth=min_depth,
    )
    prefix, joint_count, folds = joints.shape[:-2], joints.shape[-2], 3
    ids = torch.arange(joint_count, device=joints.device)
    fold_ids = torch.arange(folds, device=joints.device)
    keep = ((ids[None, :] - 1) % folds != fold_ids[:, None]) | (ids[None, :] == 0)
    subset_valid = valid.unsqueeze(-2) & keep
    subset_joints = joints.unsqueeze(-3).expand(*prefix, folds, joint_count, 3)
    subset_uv = uv.unsqueeze(-3).expand(*prefix, folds, joint_count, 2)
    subset_confidence = robust.unsqueeze(-2).expand(*prefix, folds, joint_count)
    subset_K = K.unsqueeze(-2).expand(*prefix, folds, 4)
    wrist = solve_wrist_locked_translation(
        subset_joints, subset_uv, subset_confidence, subset_valid, subset_K,
    )
    full = solve_full_translation(
        subset_joints, subset_uv, subset_confidence, subset_valid, subset_K,
        min_valid_joints=2, damping=1e-10,
    )
    use_wrist = reference.wrist_used.unsqueeze(-1)
    z = torch.where(use_wrist, wrist.translation[..., 2], full.translation[..., 2])
    positive = ((subset_joints[..., 2] + z.unsqueeze(-1) > min_depth)
                & subset_valid & (subset_confidence > 0)).sum(-1) >= 2
    eligible = (torch.where(use_wrist, wrist.valid, full.valid & positive)
                & reference.analytic_valid.unsqueeze(-1) & torch.isfinite(z) & (z > min_depth))
    count = eligible.sum(-1)
    safe_z = torch.where(eligible, z, torch.zeros_like(z))
    mean = safe_z.sum(-1) / count.clamp_min(1)
    variance = torch.where(eligible, (safe_z - mean.unsqueeze(-1)).square(), torch.zeros_like(z)).sum(-1)
    spread = torch.sqrt(variance / count.clamp_min(1))
    spread_valid = reference.analytic_valid & (count >= 2)
    relative = spread / reference.analytic_translation[..., 2].clamp_min(min_depth)
    score = count.to(joints.dtype) / folds * torch.exp(-(relative / relative_scale).square())
    score = torch.where(spread_valid, score, torch.full_like(score, 0.5))
    score = torch.where(reference.valid, score, torch.zeros_like(score)).clamp(0, 1)
    return RootSubsetReliability(
        torch.where(spread_valid, spread, torch.zeros_like(spread)),
        torch.where(spread_valid, relative, torch.zeros_like(relative)),
        count, spread_valid, score,
    )
