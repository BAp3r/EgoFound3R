from __future__ import annotations


from dataclasses import dataclass
from enum import IntEnum
import torch
from torch import nn
from egohandmetric_prompt.multirate_contracts import MultiRateFrameMap


HAND_ROOT_SMOOTHING_MODE_NONE = "none"


HAND_ROOT_SMOOTHING_MODE_CONFIDENCE_THREE_TAP = "root_confidence_three_tap"


HAND_ROOT_SMOOTHING_WINDOW_FRAMES = 3


HAND_ROOT_SMOOTHING_MIN_STRENGTH = 0.15


HAND_ROOT_SMOOTHING_MAX_STRENGTH = 0.65


@dataclass(frozen=True, slots=True)
class HighRateCameraOutput:
    extrinsic_vector_base: torch.Tensor
    extrinsic_vector_refined: torch.Tensor
    camera_pose: torch.Tensor
    residual_vector: torch.Tensor
    anchor_exact_mask: torch.Tensor
    camera_refined_valid: torch.Tensor


@dataclass(frozen=True, slots=True)
class HighRateHandRootSmoothingOutput:
    joint_xyz: torch.Tensor
    vertex_xyz: torch.Tensor
    root_xyz_raw: torch.Tensor
    root_xyz_smoothed: torch.Tensor
    root_delta: torch.Tensor
    applied_mask: torch.Tensor


def smooth_high_rate_hand_root(
    joint_xyz: torch.Tensor,
    vertex_xyz: torch.Tensor,
    anchor_valid: torch.Tensor,
    anchor_quality: torch.Tensor,
    *,
    min_strength: float = HAND_ROOT_SMOOTHING_MIN_STRENGTH,
    max_strength: float = HAND_ROOT_SMOOTHING_MAX_STRENGTH,
    min_depth: float = 1e-6,
) -> HighRateHandRootSmoothingOutput:
    if joint_xyz.ndim != 5 or joint_xyz.shape[2] != 2 or joint_xyz.shape[-1] != 3:
        raise ValueError("joint_xyz must have shape [B, T_H, 2, J, 3]")
    if joint_xyz.shape[3] < 1:
        raise ValueError("joint_xyz must contain the Root joint")
    if vertex_xyz.ndim != 5 or vertex_xyz.shape[2] != 2 or vertex_xyz.shape[-1] != 3:
        raise ValueError("vertex_xyz must have shape [B, T_H, 2, V, 3]")
    if vertex_xyz.shape[:3] != joint_xyz.shape[:3]:
        raise ValueError("joint_xyz and vertex_xyz must share [B, T_H, side]")
    expected = joint_xyz.shape[:3]
    if tuple(anchor_valid.shape) != expected or anchor_valid.dtype is not torch.bool:
        raise ValueError("anchor_valid must be bool with shape [B, T_H, 2]")
    if tuple(anchor_quality.shape) != expected or not anchor_quality.is_floating_point():
        raise ValueError("anchor_quality must be floating point with shape [B, T_H, 2]")
    tensors = (joint_xyz, vertex_xyz, anchor_valid, anchor_quality)
    if any(value.device != joint_xyz.device for value in tensors):
        raise ValueError("all H-rate Root smoothing inputs must share a device")
    if not joint_xyz.is_floating_point() or not vertex_xyz.is_floating_point():
        raise ValueError("joint_xyz and vertex_xyz must be floating point")
    if not 0.0 <= float(min_strength) <= float(max_strength) <= 1.0:
        raise ValueError("smoothing strengths must satisfy 0 <= min <= max <= 1")
    if not float(min_depth) > 0.0:
        raise ValueError("min_depth must be positive")

    root_raw = joint_xyz[..., 0, :]
    applied_mask = torch.zeros(expected, device=joint_xyz.device, dtype=torch.bool)
    root_delta = torch.zeros_like(root_raw)
    if joint_xyz.shape[1] < HAND_ROOT_SMOOTHING_WINDOW_FRAMES:
        return HighRateHandRootSmoothingOutput(
            joint_xyz=joint_xyz,
            vertex_xyz=vertex_xyz,
            root_xyz_raw=root_raw,
            root_xyz_smoothed=root_raw,
            root_delta=root_delta,
            applied_mask=applied_mask,
        )

    work_dtype = torch.float64 if joint_xyz.dtype is torch.float64 else torch.float32
    root_work = root_raw.to(dtype=work_dtype)
    valid = (
        anchor_valid
        & torch.isfinite(root_work).all(dim=-1)
        & root_work[..., 2].gt(float(min_depth))
    )
    quality = anchor_quality.detach().to(device=joint_xyz.device, dtype=work_dtype)
    quality = torch.where(
        valid & torch.isfinite(quality),
        quality.clamp(0.0, 1.0),
        torch.zeros_like(quality),
    )
    safe_z = torch.where(valid, root_work[..., 2], torch.ones_like(root_work[..., 2]))
    representation = torch.stack(
        (
            root_work[..., 0] / safe_z,
            root_work[..., 1] / safe_z,
            safe_z.log(),
        ),
        dim=-1,
    )
    representation = torch.where(
        valid.unsqueeze(-1),
        representation,
        torch.zeros_like(representation),
    )

    previous_quality = quality[:, :-2]
    center_quality = quality[:, 1:-1]
    next_quality = quality[:, 2:]
    denominator = previous_quality + 2.0 * center_quality + next_quality
    support = (
        valid[:, :-2]
        & valid[:, 1:-1]
        & valid[:, 2:]
        & denominator.gt(torch.finfo(work_dtype).eps)
    )
    candidate = (
        previous_quality.unsqueeze(-1) * representation[:, :-2]
        + 2.0 * center_quality.unsqueeze(-1) * representation[:, 1:-1]
        + next_quality.unsqueeze(-1) * representation[:, 2:]
    ) / denominator.clamp_min(torch.finfo(work_dtype).eps).unsqueeze(-1)
    strength = float(min_strength) + (
        float(max_strength) - float(min_strength)
    ) * (1.0 - center_quality)
    smoothed_representation = representation[:, 1:-1] + strength.unsqueeze(-1) * (
        candidate - representation[:, 1:-1]
    )
    smoothed_z = smoothed_representation[..., 2].exp()
    root_candidate = torch.stack(
        (
            smoothed_representation[..., 0] * smoothed_z,
            smoothed_representation[..., 1] * smoothed_z,
            smoothed_z,
        ),
        dim=-1,
    )
    support &= torch.isfinite(root_candidate).all(dim=-1) & smoothed_z.gt(float(min_depth))
    center_delta = torch.where(
        support.unsqueeze(-1),
        root_candidate - root_work[:, 1:-1],
        torch.zeros_like(root_candidate),
    )
    root_delta = torch.cat(
        (
            torch.zeros_like(root_work[:, :1]),
            center_delta,
            torch.zeros_like(root_work[:, -1:]),
        ),
        dim=1,
    ).to(dtype=joint_xyz.dtype)
    applied_mask = torch.cat(
        (
            torch.zeros_like(valid[:, :1]),
            support,
            torch.zeros_like(valid[:, -1:]),
        ),
        dim=1,
    )
    root_smoothed = root_raw + root_delta
    return HighRateHandRootSmoothingOutput(
        joint_xyz=joint_xyz + root_delta.unsqueeze(-2),
        vertex_xyz=vertex_xyz + root_delta.unsqueeze(-2),
        root_xyz_raw=root_raw,
        root_xyz_smoothed=root_smoothed,
        root_delta=root_delta,
        applied_mask=applied_mask,
    )


class HandRefinedMode(IntEnum):
    NO_ANCHOR_INVALID = 0
    SINGLE_ANCHOR_HOLD = 1
    ONE_SIDED_EXTRAPOLATION = 2
    BRACKETED_INTERPOLATION = 3
    EXACT_ANCHOR = 4


@dataclass(frozen=True, slots=True)
class SparseHandQueryMap:
    left_high: torch.Tensor
    right_high: torch.Tensor
    previous_left_high: torch.Tensor
    next_right_high: torch.Tensor
    alpha: torch.Tensor
    exact_mask: torch.Tensor
    bracketed_mask: torch.Tensor
    extrapolation_mask: torch.Tensor
    valid: torch.Tensor
    support_confidence: torch.Tensor
    mode: torch.Tensor


@dataclass(frozen=True, slots=True)
class TemporalInterpolationQueryMap:
    interval_steps: torch.Tensor
    high_anchor_indices: torch.Tensor
    query_left_high: torch.Tensor
    query_right_high: torch.Tensor
    query_alpha: torch.Tensor
    output_present: torch.Tensor
    high_anchor_mask: torch.Tensor
    global_anchor_mask: torch.Tensor
    global_route_indices: torch.Tensor

    def validate(self, frame_map: MultiRateFrameMap) -> None:
        batch_size, high_frames = frame_map.high_frame_present.shape
        full_frames = self.output_present.shape[1]
        expected = {
            "interval_steps": (batch_size, max(high_frames - 1, 0)),
            "high_anchor_indices": (batch_size, high_frames),
            "query_left_high": (batch_size, full_frames),
            "query_right_high": (batch_size, full_frames),
            "query_alpha": (batch_size, full_frames),
            "output_present": (batch_size, full_frames),
            "high_anchor_mask": (batch_size, full_frames),
            "global_anchor_mask": (batch_size, full_frames),
            "global_route_indices": (batch_size, full_frames),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if tuple(value.shape) != shape:
                raise ValueError(f"query_map.{name} must have shape {shape}")
            if value.device != frame_map.high_frame_present.device:
                raise ValueError(f"query_map.{name} must share the frame-map device")
        for name in (
            "interval_steps",
            "high_anchor_indices",
            "query_left_high",
            "query_right_high",
            "global_route_indices",
        ):
            if getattr(self, name).dtype is not torch.long:
                raise ValueError(f"query_map.{name} must have dtype torch.long")
        for name in ("output_present", "high_anchor_mask", "global_anchor_mask"):
            if getattr(self, name).dtype is not torch.bool:
                raise ValueError(f"query_map.{name} must have dtype torch.bool")
        if self.query_alpha.dtype not in (torch.float32, torch.float64):
            raise ValueError("query_map.query_alpha must be float32 or float64")
        if bool((self.interval_steps < 1).any().item()):
            raise ValueError("query_map.interval_steps must contain positive integers")
        if bool((self.high_anchor_mask & ~self.output_present).any().item()):
            raise ValueError("high anchors must be present on the R axis")
        if bool((self.global_anchor_mask & ~self.high_anchor_mask).any().item()):
            raise ValueError("every G anchor must also be an H anchor")


@dataclass(frozen=True, slots=True)
class FullRateRefinerOutput:
    query_map: TemporalInterpolationQueryMap
    camera_pose_base: torch.Tensor
    camera_pose_refined: torch.Tensor
    camera_residual_6d: torch.Tensor
    camera_anchor_exact_mask: torch.Tensor
    camera_valid: torch.Tensor
    hand_base: dict[str, torch.Tensor]
    hand_refined: dict[str, torch.Tensor]
    hand_residual: dict[str, torch.Tensor]
    hand_anchor_exact_mask: torch.Tensor
    hand_valid: torch.Tensor
    root_translation_valid: torch.Tensor
    relative_hand_valid: torch.Tensor
    hand_support_confidence: torch.Tensor
    hand_mode: torch.Tensor

    def high_rate_camera(self) -> HighRateCameraOutput:
        base_pose = gather_full_rate_at_high(self.camera_pose_base, self.query_map)
        refined_pose = gather_full_rate_at_high(self.camera_pose_refined, self.query_map)
        residual = gather_full_rate_at_high(self.camera_residual_6d, self.query_map)
        anchor_mask = gather_full_rate_at_high(
            self.camera_anchor_exact_mask,
            self.query_map,
        ).to(dtype=torch.bool)
        valid = gather_full_rate_at_high(self.camera_valid, self.query_map).to(dtype=torch.bool)
        base_vector = torch.cat(
            (matrix_to_rotation_6d(base_pose[..., :3, :3]), base_pose[..., :3, 3]),
            dim=-1,
        )
        refined_vector = torch.cat(
            (matrix_to_rotation_6d(refined_pose[..., :3, :3]), refined_pose[..., :3, 3]),
            dim=-1,
        )
        return HighRateCameraOutput(
            extrinsic_vector_base=base_vector,
            extrinsic_vector_refined=refined_vector,
            camera_pose=refined_pose,
            residual_vector=residual,
            anchor_exact_mask=anchor_mask,
            camera_refined_valid=valid,
        )

    def to_public_dict(self) -> dict[str, torch.Tensor]:
        result = {
            "camera_pose_base_full": self.camera_pose_base,
            "camera_pose_interpolated_full": self.camera_pose_base,
            "camera_pose_refined_full": self.camera_pose_refined,
            "camera_residual_6d_full": self.camera_residual_6d,
            "camera_anchor_exact_mask_full": self.camera_anchor_exact_mask,
            "camera_refined_valid_full": self.camera_valid,
            "camera_interpolation_valid_full": self.camera_valid,
            "hand_anchor_exact_mask_full": self.hand_anchor_exact_mask,
            "hand_refined_valid_full": self.hand_valid,
            "root_translation_valid_full": self.root_translation_valid,
            "relative_hand_valid_full": self.relative_hand_valid,
            "hand_refined_support_confidence_full": self.hand_support_confidence,
            "hand_refined_mode_full": self.hand_mode,
            "refiner_interval_steps": self.query_map.interval_steps,
            "refiner_high_anchor_indices": self.query_map.high_anchor_indices,
            "refiner_output_present": self.query_map.output_present,
            "refiner_high_anchor_mask_full": self.query_map.high_anchor_mask,
            "refiner_global_anchor_mask_full": self.query_map.global_anchor_mask,
        }
        public_names = {
            "joint_xyz": "dense_joint_xyz",
            "vertex_xyz": "dense_vertex_xyz",
            "joint_visibility_logits": "dense_joint_visibility_logits",
            "vertex_visibility_logits": "dense_vertex_visibility_logits",
            "joint_contact_logits": "dense_joint_contact_logits",
            "vertex_contact_logits": "dense_vertex_contact_logits",
            "joint_contact_log_conf": "dense_joint_contact_log_conf",
            "vertex_contact_log_conf": "dense_vertex_contact_log_conf",
            "joint_log_conf": "dense_joint_log_conf",
            "vertex_log_conf": "dense_vertex_log_conf",
            "joint_contact_log_distance": "dense_joint_contact_log_distance",
            "vertex_contact_log_distance": "dense_vertex_contact_log_distance",
        }
        for field_name, public_name in public_names.items():
            if field_name in self.hand_base:
                result[f"{public_name}_base_full"] = self.hand_base[field_name]
                result[f"{public_name}_interpolated_full"] = self.hand_base[field_name]
                result[f"{public_name}_refined_full"] = self.hand_refined[field_name]
            if field_name in self.hand_residual:
                result[f"{public_name}_residual_full"] = self.hand_residual[field_name]
        return result


def matrix_to_rotation_6d(rotation: torch.Tensor) -> torch.Tensor:
    if rotation.ndim < 2 or rotation.shape[-2:] != (3, 3):
        raise ValueError("rotation must have shape [..., 3, 3]")
    return rotation[..., :, :2].transpose(-1, -2).reshape(*rotation.shape[:-2], 6)


def clip_local_metric_w2c(
    camera_pose_w2c: torch.Tensor,
    present: torch.Tensor,
    *,
    translation_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    if camera_pose_w2c.ndim != 4 or camera_pose_w2c.shape[-2:] != (4, 4):
        raise ValueError("camera_pose_w2c must have shape [B, T, 4, 4]")
    batch_size, frames = camera_pose_w2c.shape[:2]
    if frames == 0:
        raise ValueError("camera_pose_w2c must contain at least one temporal anchor")
    if tuple(present.shape) != (batch_size, frames) or present.dtype is not torch.bool:
        raise ValueError("present must be bool with shape [B, T]")
    if present.device != camera_pose_w2c.device:
        raise ValueError("present and camera_pose_w2c must share a device")
    if translation_scale is None:
        translation_scale = torch.ones(
            batch_size,
            device=camera_pose_w2c.device,
            dtype=camera_pose_w2c.dtype,
        )
    if tuple(translation_scale.shape) != (batch_size,):
        raise ValueError("translation_scale must have shape [B]")
    if translation_scale.device != camera_pose_w2c.device:
        raise ValueError("translation_scale and camera_pose_w2c must share a device")

    geometry_dtype = _geometry_dtype(camera_pose_w2c.dtype)
    pose = camera_pose_w2c.to(dtype=geometry_dtype)
    valid = present & torch.isfinite(pose).all(dim=(-2, -1))
    axis = torch.arange(frames, device=pose.device, dtype=torch.long).unsqueeze(0)
    reference_index = torch.where(valid, axis, torch.full_like(axis, frames)).amin(dim=1)
    reference_available = reference_index.lt(frames)
    safe_reference = reference_index.clamp(0, max(frames - 1, 0)).unsqueeze(1)
    reference_pose = _gather_time_axis(pose, safe_reference)[:, 0]
    with torch.autocast(device_type=pose.device.type, enabled=False):
        relative = torch.matmul(pose, w2c_to_c2w(reference_pose).unsqueeze(1))
    scale = translation_scale.detach().to(device=pose.device, dtype=pose.dtype)
    scale_valid = torch.isfinite(scale) & (scale > 0)
    safe_scale = torch.where(scale_valid, scale, torch.ones_like(scale))
    relative = relative.clone()
    relative[..., :3, 3] = relative[..., :3, 3] * safe_scale[:, None, None]
    output_valid = valid & reference_available.unsqueeze(1)
    identity = _identity_pose(
        (batch_size, frames),
        device=pose.device,
        dtype=pose.dtype,
    )
    relative = torch.where(output_valid[..., None, None], relative, identity)
    reference_mask = axis.eq(reference_index.unsqueeze(1)) & output_valid
    relative = torch.where(reference_mask[..., None, None], identity, relative)
    return relative


def _identity_pose(
    shape: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    eye = torch.eye(4, device=device, dtype=dtype)
    return eye.reshape(*((1,) * len(shape)), 4, 4).expand(*shape, 4, 4)


def _pose_from_rotation_translation(rotation: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    top = torch.cat((rotation, translation.unsqueeze(-1)), dim=-1)
    bottom = torch.zeros_like(top[..., :1, :])
    bottom[..., 0, 3] = 1.0
    return torch.cat((top, bottom), dim=-2)


def w2c_to_c2w(w2c_pose: torch.Tensor) -> torch.Tensor:
    if w2c_pose.ndim < 2 or w2c_pose.shape[-2:] != (4, 4):
        raise ValueError("w2c_pose must have shape [..., 4, 4]")
    w2c_pose = w2c_pose.to(dtype=_geometry_dtype(w2c_pose.dtype))
    rotation_cw = w2c_pose[..., :3, :3]
    translation_cw = w2c_pose[..., :3, 3]
    rotation_wc = rotation_cw.transpose(-1, -2)
    with torch.autocast(device_type=w2c_pose.device.type, enabled=False):
        center_w = -torch.matmul(rotation_wc, translation_cw.unsqueeze(-1)).squeeze(-1)
    return _pose_from_rotation_translation(rotation_wc, center_w)


def c2w_to_w2c(c2w_pose: torch.Tensor) -> torch.Tensor:
    if c2w_pose.ndim < 2 or c2w_pose.shape[-2:] != (4, 4):
        raise ValueError("c2w_pose must have shape [..., 4, 4]")
    c2w_pose = c2w_pose.to(dtype=_geometry_dtype(c2w_pose.dtype))
    rotation_wc = c2w_pose[..., :3, :3]
    center_w = c2w_pose[..., :3, 3]
    rotation_cw = rotation_wc.transpose(-1, -2)
    with torch.autocast(device_type=c2w_pose.device.type, enabled=False):
        translation_cw = -torch.matmul(rotation_cw, center_w.unsqueeze(-1)).squeeze(-1)
    return _pose_from_rotation_translation(rotation_cw, translation_cw)


def _normalize_quaternion(quaternion: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if quaternion.ndim < 1 or quaternion.shape[-1] != 4:
        raise ValueError("quaternion must have shape [..., 4] in xyzw order")
    identity = torch.zeros_like(quaternion)
    identity[..., 3] = 1.0
    norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    normalized = quaternion / norm.clamp_min(eps)
    normalized = torch.where(norm > eps, normalized, identity)
    return torch.where(normalized[..., 3:4] < 0.0, -normalized, normalized)


def rotation_matrix_to_quaternion(rotation: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if rotation.ndim < 2 or rotation.shape[-2:] != (3, 3):
        raise ValueError("rotation must have shape [..., 3, 3]")
    matrix = rotation
    m00, m01, m02 = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2]
    m10, m11, m12 = matrix[..., 1, 0], matrix[..., 1, 1], matrix[..., 1, 2]
    m20, m21, m22 = matrix[..., 2, 0], matrix[..., 2, 1], matrix[..., 2, 2]
    quaternion_abs = torch.sqrt(
        torch.stack(
            (
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ),
            dim=-1,
        ).clamp_min(eps)
    )
    candidates_wxyz = torch.stack(
        (
            torch.stack((quaternion_abs[..., 0].square(), m21 - m12, m02 - m20, m10 - m01), dim=-1),
            torch.stack((m21 - m12, quaternion_abs[..., 1].square(), m01 + m10, m02 + m20), dim=-1),
            torch.stack((m02 - m20, m01 + m10, quaternion_abs[..., 2].square(), m12 + m21), dim=-1),
            torch.stack((m10 - m01, m02 + m20, m12 + m21, quaternion_abs[..., 3].square()), dim=-1),
        ),
        dim=-2,
    )
    candidates_wxyz = candidates_wxyz / (2.0 * quaternion_abs.unsqueeze(-1)).clamp_min(eps)
    best_index = quaternion_abs.argmax(dim=-1, keepdim=True).unsqueeze(-1)
    quaternion_wxyz = candidates_wxyz.gather(-2, best_index.expand(*best_index.shape[:-1], 4)).squeeze(-2)
    quaternion_xyzw = quaternion_wxyz[..., (1, 2, 3, 0)]
    return _normalize_quaternion(quaternion_xyzw, eps=eps)


def quaternion_to_rotation_matrix(quaternion: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    quaternion = _normalize_quaternion(quaternion, eps=eps)
    x, y, z, w = quaternion.unbind(dim=-1)
    xx, yy, zz = x.square(), y.square(), z.square()
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return torch.stack(
        (
            1.0 - 2.0 * (yy + zz),
            2.0 * (xy - wz),
            2.0 * (xz + wy),
            2.0 * (xy + wz),
            1.0 - 2.0 * (xx + zz),
            2.0 * (yz - wx),
            2.0 * (xz - wy),
            2.0 * (yz + wx),
            1.0 - 2.0 * (xx + yy),
        ),
        dim=-1,
    ).reshape(*quaternion.shape[:-1], 3, 3)


def quaternion_slerp(
    left: torch.Tensor,
    right: torch.Tensor,
    alpha: torch.Tensor | float,
    eps: float = 1e-8,
) -> torch.Tensor:
    if left.shape != right.shape:
        raise ValueError("left and right quaternions must have identical shapes")
    left = _normalize_quaternion(left, eps=eps)
    right = _normalize_quaternion(right.to(device=left.device, dtype=left.dtype), eps=eps)
    alpha_tensor = torch.as_tensor(alpha, device=left.device, dtype=left.dtype)
    if alpha_tensor.ndim > left.ndim:
        raise ValueError("alpha has too many dimensions for quaternion interpolation")
    while alpha_tensor.ndim < left.ndim - 1:
        alpha_tensor = alpha_tensor.unsqueeze(0)
    if alpha_tensor.ndim == left.ndim - 1:
        alpha_tensor = alpha_tensor.unsqueeze(-1)
    if alpha_tensor.shape[-1] != 1:
        raise ValueError("alpha must broadcast to quaternion leading dimensions")
    try:
        torch.broadcast_shapes(alpha_tensor.shape, (*left.shape[:-1], 1))
    except RuntimeError as error:
        raise ValueError("alpha must broadcast to quaternion leading dimensions") from error
    alpha_tensor = alpha_tensor.clamp(0.0, 1.0)
    dot = (left * right).sum(dim=-1, keepdim=True)
    right = torch.where(dot < 0.0, -right, right)
    dot = (left * right).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    angle_eps = max(float(eps), float(torch.finfo(left.dtype).eps))
    theta = torch.acos(dot.clamp(-1.0 + angle_eps, 1.0 - angle_eps))
    sin_theta = torch.sin(theta)
    linear = _normalize_quaternion((1.0 - alpha_tensor) * left + alpha_tensor * right, eps=eps)
    slerp = (
        torch.sin((1.0 - alpha_tensor) * theta) / sin_theta.clamp_min(angle_eps) * left
        + torch.sin(alpha_tensor * theta) / sin_theta.clamp_min(angle_eps) * right
    )
    return _normalize_quaternion(torch.where((dot > 0.9995) | (sin_theta.abs() < angle_eps), linear, slerp), eps=eps)


def _geometry_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype


def slerp_rotation_matrices(
    left: torch.Tensor,
    right: torch.Tensor,
    alpha: torch.Tensor | float,
) -> torch.Tensor:
    if left.shape != right.shape or left.ndim < 2 or left.shape[-2:] != (3, 3):
        raise ValueError("left and right rotations must have identical shape [..., 3, 3]")
    work_dtype = _geometry_dtype(left.dtype)
    left_work = left.to(dtype=work_dtype)
    right_work = right.to(device=left.device, dtype=work_dtype)
    result = quaternion_to_rotation_matrix(
        quaternion_slerp(
            rotation_matrix_to_quaternion(left_work),
            rotation_matrix_to_quaternion(right_work),
            alpha,
        )
    )
    return result.to(dtype=left.dtype)


def build_temporal_interpolation_query_map(
    frame_map: MultiRateFrameMap,
    *,
    interpolation_factor: int = 1,
    interval_steps: torch.Tensor | None = None,
) -> TemporalInterpolationQueryMap:
    frame_map.validate()
    if int(interpolation_factor) < 1:
        raise ValueError("interpolation_factor must be a positive integer")
    high_present = frame_map.high_frame_present.detach()
    batch_size, high_frames = high_present.shape
    device = high_present.device
    if interval_steps is None:
        steps = torch.full(
            (batch_size, max(high_frames - 1, 0)),
            int(interpolation_factor),
            device=device,
            dtype=torch.long,
        )
    else:
        if not isinstance(interval_steps, torch.Tensor):
            raise TypeError("interval_steps must be a tensor when provided")
        if tuple(interval_steps.shape) != (batch_size, max(high_frames - 1, 0)):
            raise ValueError("interval_steps must have shape [B, T_H - 1]")
        if interval_steps.device != device:
            raise ValueError("interval_steps and frame_map must share a device")
        steps = interval_steps.detach().to(dtype=torch.long)
    if bool((steps < 1).any().item()):
        raise ValueError("interval_steps must contain positive integers")

    high_anchor_indices = torch.cat(
        (
            torch.zeros(batch_size, 1, device=device, dtype=torch.long),
            steps.cumsum(dim=1),
        ),
        dim=1,
    )
    last_anchor = torch.where(
        high_present,
        high_anchor_indices,
        torch.full_like(high_anchor_indices, -1),
    ).amax(dim=1)
    if bool((last_anchor < 0).any().item()):
        raise ValueError("each clip must contain at least one H frame")
    full_frames = int(last_anchor.max().item()) + 1
    output_axis = torch.arange(full_frames, device=device, dtype=torch.long)
    query_axis = output_axis.view(1, full_frames, 1)
    anchor_axis = high_anchor_indices.unsqueeze(1)
    high_slot_axis = torch.arange(high_frames, device=device, dtype=torch.long).view(1, 1, high_frames)
    present = high_present.unsqueeze(1)

    raw_left = torch.where(
        present & anchor_axis.le(query_axis),
        high_slot_axis,
        torch.full_like(high_slot_axis, -1),
    ).amax(dim=2)
    raw_right = torch.where(
        present & anchor_axis.ge(query_axis),
        high_slot_axis,
        torch.full_like(high_slot_axis, high_frames),
    ).amin(dim=2)
    left_exists = raw_left.ge(0)
    right_exists = raw_right.lt(high_frames)
    safe_left = raw_left.clamp(0, high_frames - 1)
    safe_right = raw_right.clamp(0, high_frames - 1)
    query_left = torch.where(left_exists, safe_left, safe_right)
    query_right = torch.where(right_exists, safe_right, safe_left)
    left_position = high_anchor_indices.gather(1, query_left)
    right_position = high_anchor_indices.gather(1, query_right)
    span = right_position - left_position
    alpha = torch.where(
        span.gt(0),
        (output_axis.unsqueeze(0) - left_position).to(dtype=torch.float32)
        / span.clamp_min(1).to(dtype=torch.float32),
        torch.zeros(batch_size, full_frames, device=device, dtype=torch.float32),
    ).clamp(0.0, 1.0)

    anchor_matches = present & anchor_axis.eq(query_axis)
    high_anchor_mask = anchor_matches.any(dim=2)
    between = left_exists & right_exists & query_right.eq(query_left + 1)
    if high_frames > 1:
        pair_valid = frame_map.high_temporal_pair_valid.gather(
            1,
            query_left.clamp(max=high_frames - 2),
        )
    else:
        pair_valid = torch.zeros(batch_size, full_frames, device=device, dtype=torch.bool)
    within_clip = output_axis.unsqueeze(0).le(last_anchor.unsqueeze(1))
    output_present = within_clip & (high_anchor_mask | (between & pair_valid))

    global_frames = frame_map.global_frame_present.shape[1]
    if global_frames == 0:
        global_anchor_mask = torch.zeros_like(output_present)
        global_route_indices = torch.zeros_like(query_left)
    else:
        global_high = frame_map.global_anchor_indices.clamp(0, high_frames - 1)
        global_positions = high_anchor_indices.gather(1, global_high)
        global_matches = (
            frame_map.global_frame_present.unsqueeze(1)
            & global_positions.unsqueeze(1).eq(query_axis)
        )
        global_anchor_mask = global_matches.any(dim=2)
        global_route_indices = global_matches.to(dtype=torch.long).argmax(dim=2)

    result = TemporalInterpolationQueryMap(
        interval_steps=steps,
        high_anchor_indices=high_anchor_indices,
        query_left_high=query_left,
        query_right_high=query_right,
        query_alpha=alpha,
        output_present=output_present,
        high_anchor_mask=high_anchor_mask,
        global_anchor_mask=global_anchor_mask,
        global_route_indices=global_route_indices,
    )
    result.validate(frame_map)
    return result


def _gather_time_axis(value: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if value.ndim < 2 or indices.ndim != 2 or value.shape[0] != indices.shape[0]:
        raise ValueError("time gather requires value [B, T, ...] and indices [B, Q]")
    safe = indices.to(device=value.device, dtype=torch.long).clamp(0, value.shape[1] - 1)
    tail = value.shape[2:]
    view = safe.reshape(safe.shape[0], safe.shape[1], *([1] * len(tail)))
    return value.gather(1, view.expand(safe.shape[0], safe.shape[1], *tail))


def gather_full_rate_at_high(
    value_full: torch.Tensor,
    query_map: TemporalInterpolationQueryMap,
) -> torch.Tensor:
    gathered = _gather_time_axis(value_full, query_map.high_anchor_indices)
    high_present = query_map.high_anchor_indices.le(value_full.shape[1] - 1)
    view = high_present.reshape(*high_present.shape, *([1] * (gathered.ndim - 2)))
    return torch.where(view, gathered, torch.zeros_like(gathered))


def build_sparse_hand_query_map(
    frame_map: MultiRateFrameMap,
    query_map: TemporalInterpolationQueryMap,
    hand_anchor_valid_high: torch.Tensor,
    hand_anchor_quality_high: torch.Tensor | None = None,
) -> SparseHandQueryMap:
    batch_size, high_frames = frame_map.high_frame_present.shape
    full_frames = query_map.output_present.shape[1]
    expected = (batch_size, high_frames, 2)
    if tuple(hand_anchor_valid_high.shape) != expected or hand_anchor_valid_high.dtype is not torch.bool:
        raise ValueError(f"hand_anchor_valid_high must be bool with shape {expected}")
    if hand_anchor_valid_high.device != query_map.output_present.device:
        raise ValueError("hand anchors and interpolation query map must share a device")
    if hand_anchor_quality_high is None:
        hand_anchor_quality_high = torch.ones(
            expected,
            device=hand_anchor_valid_high.device,
            dtype=torch.float32,
        )
    if tuple(hand_anchor_quality_high.shape) != expected:
        raise ValueError(f"hand_anchor_quality_high must have shape {expected}")
    quality = torch.nan_to_num(
        hand_anchor_quality_high.detach().float(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).clamp(0.0, 1.0)
    valid = (
        hand_anchor_valid_high.detach()
        & frame_map.high_frame_present.unsqueeze(-1)
        & torch.isfinite(hand_anchor_quality_high)
        & (quality > 0)
    )

    positions = query_map.high_anchor_indices[:, None, :].expand(-1, 2, -1).reshape(-1, high_frames)
    anchor_valid = valid.permute(0, 2, 1).reshape(-1, high_frames)
    anchor_quality = quality.permute(0, 2, 1).reshape(-1, high_frames)
    side_batches = positions.shape[0]
    query_axis = torch.arange(
        full_frames,
        device=positions.device,
        dtype=torch.long,
    ).view(1, full_frames, 1)
    position_grid = positions.unsqueeze(1)
    slot_grid = torch.arange(high_frames, device=positions.device, dtype=torch.long).view(
        1, 1, high_frames
    )
    valid_grid = anchor_valid.unsqueeze(1)

    raw_left = torch.where(
        valid_grid & position_grid.le(query_axis),
        slot_grid,
        torch.full_like(slot_grid, -1),
    ).amax(dim=-1)
    raw_right = torch.where(
        valid_grid & position_grid.ge(query_axis),
        slot_grid,
        torch.full_like(slot_grid, high_frames),
    ).amin(dim=-1)
    has_left = raw_left.ge(0)
    has_right = raw_right.lt(high_frames)
    safe_left = raw_left.clamp(0, high_frames - 1)
    safe_right = raw_right.clamp(0, high_frames - 1)
    left = torch.where(has_left, safe_left, safe_right)
    right = torch.where(has_right, safe_right, safe_left)
    left_position = positions.gather(1, left)
    right_position = positions.gather(1, right)

    previous = torch.where(
        valid_grid & position_grid.lt(left_position.unsqueeze(-1)),
        slot_grid,
        torch.full_like(slot_grid, -1),
    ).amax(dim=-1)
    following = torch.where(
        valid_grid & position_grid.gt(right_position.unsqueeze(-1)),
        slot_grid,
        torch.full_like(slot_grid, high_frames),
    ).amin(dim=-1)
    previous = torch.where(previous.ge(0), previous, left)
    following = torch.where(following.lt(high_frames), following, right)

    anchor_count = anchor_valid.sum(dim=-1, keepdim=True)
    exact = (
        valid_grid & position_grid.eq(query_axis)
    ).any(dim=-1)
    bracketed = has_left & has_right & left_position.lt(query_axis[..., 0]) & right_position.gt(
        query_axis[..., 0]
    )
    extrapolation = has_left.logical_xor(has_right) & anchor_count.ge(2)
    single = anchor_count.eq(1).expand(-1, full_frames) & ~exact
    output_present = query_map.output_present[:, None, :].expand(-1, 2, -1).reshape(
        side_batches,
        full_frames,
    )
    exact &= output_present
    bracketed &= output_present
    extrapolation &= output_present
    single &= output_present
    support_valid = output_present & anchor_count.ge(1)

    span = (right_position - left_position).clamp_min(1)
    alpha = torch.where(
        bracketed,
        (query_axis[..., 0] - left_position).float() / span.float(),
        torch.zeros(side_batches, full_frames, device=positions.device),
    ).clamp(0.0, 1.0)
    left_quality = anchor_quality.gather(1, left)
    right_quality = anchor_quality.gather(1, right)
    nearest = torch.where(has_left, left, right)
    nearest_position = positions.gather(1, nearest)
    nearest_quality = anchor_quality.gather(1, nearest)
    second = torch.where(has_left, previous, following)
    second_position = positions.gather(1, second)
    neighbor_span = (nearest_position - second_position).abs().clamp_min(1)
    query_distance = (query_axis[..., 0] - nearest_position).abs()

    support_confidence = torch.zeros_like(alpha)
    support_confidence = torch.where(exact, nearest_quality, support_confidence)
    bracket_quality = torch.sqrt((left_quality * right_quality).clamp_min(0.0))
    bracket_quality = bracket_quality * torch.exp(-0.05 * (span.float() - 1.0).clamp_min(0.0))
    support_confidence = torch.where(bracketed, bracket_quality, support_confidence)
    extrapolation_quality = 0.5 * nearest_quality / (
        1.0 + query_distance.float() / neighbor_span.float()
    )
    support_confidence = torch.where(
        extrapolation,
        extrapolation_quality,
        support_confidence,
    )
    support_confidence = torch.where(single, 0.1 * nearest_quality, support_confidence)
    support_confidence = torch.where(support_valid, support_confidence, torch.zeros_like(support_confidence))

    mode = torch.full_like(left, int(HandRefinedMode.NO_ANCHOR_INVALID))
    mode = torch.where(single, int(HandRefinedMode.SINGLE_ANCHOR_HOLD), mode)
    mode = torch.where(
        extrapolation,
        int(HandRefinedMode.ONE_SIDED_EXTRAPOLATION),
        mode,
    )
    mode = torch.where(bracketed, int(HandRefinedMode.BRACKETED_INTERPOLATION), mode)
    mode = torch.where(exact, int(HandRefinedMode.EXACT_ANCHOR), mode)

    def unflatten(value: torch.Tensor) -> torch.Tensor:
        return value.reshape(batch_size, 2, full_frames).permute(0, 2, 1)

    return SparseHandQueryMap(
        left_high=unflatten(left),
        right_high=unflatten(right),
        previous_left_high=unflatten(previous),
        next_right_high=unflatten(following),
        alpha=unflatten(alpha),
        exact_mask=unflatten(exact),
        bracketed_mask=unflatten(bracketed),
        extrapolation_mask=unflatten(extrapolation),
        valid=unflatten(support_valid),
        support_confidence=unflatten(support_confidence),
        mode=unflatten(mode),
    )


def _sparse_hand_interpolation(
    values: torch.Tensor,
    anchor_positions: torch.Tensor,
    sparse_map: SparseHandQueryMap,
    *,
    clamp_to_support: bool,
) -> torch.Tensor:
    batch_size, high_frames, side_count = values.shape[:3]
    if side_count != 2 or tuple(anchor_positions.shape) != (batch_size, high_frames):
        raise ValueError("hand values and anchor positions must align on [B, T_H, 2]")
    tail = values.shape[3:]
    flattened = values.permute(0, 2, 1, *range(3, values.ndim)).reshape(
        batch_size * 2,
        high_frames,
        *tail,
    )
    positions = anchor_positions[:, None, :].expand(-1, 2, -1).reshape(batch_size * 2, high_frames)

    def flat_index(value: torch.Tensor) -> torch.Tensor:
        return value.permute(0, 2, 1).reshape(batch_size * 2, -1)

    left = flat_index(sparse_map.left_high)
    right = flat_index(sparse_map.right_high)
    previous = flat_index(sparse_map.previous_left_high)
    following = flat_index(sparse_map.next_right_high)
    left_value = _gather_time_axis(flattened, left)
    right_value = _gather_time_axis(flattened, right)
    previous_value = _gather_time_axis(flattened, previous)
    following_value = _gather_time_axis(flattened, following)
    left_position = positions.gather(1, left)
    right_position = positions.gather(1, right)
    previous_position = positions.gather(1, previous)
    following_position = positions.gather(1, following)
    tail_dims = values.ndim - 3

    def expand(value: torch.Tensor) -> torch.Tensor:
        return value.reshape(*value.shape, *([1] * tail_dims))

    span = (right_position - left_position).clamp_min(1)
    secant = (right_value - left_value) / expand(span.to(dtype=values.dtype))
    left_slope = torch.where(
        expand(previous.ne(left)),
        (right_value - previous_value)
        / expand((right_position - previous_position).clamp_min(1).to(dtype=values.dtype)),
        secant,
    )
    right_slope = torch.where(
        expand(following.ne(right)),
        (following_value - left_value)
        / expand((following_position - left_position).clamp_min(1).to(dtype=values.dtype)),
        secant,
    )
    alpha = expand(flat_index(sparse_map.alpha).to(dtype=values.dtype))
    alpha2 = alpha.square()
    alpha3 = alpha2 * alpha
    span_value = expand(span.to(dtype=values.dtype))
    bracketed_value = (
        (2.0 * alpha3 - 3.0 * alpha2 + 1.0) * left_value
        + (alpha3 - 2.0 * alpha2 + alpha) * span_value * left_slope
        + (-2.0 * alpha3 + 3.0 * alpha2) * right_value
        + (alpha3 - alpha2) * span_value * right_slope
    )

    full_frames = left.shape[1]
    query_axis = torch.arange(full_frames, device=values.device, dtype=torch.long).view(1, -1)
    after = query_axis.gt(left_position)
    nearest_index = torch.where(after, left, right)
    second_index = torch.where(after, previous, following)
    nearest_value = _gather_time_axis(flattened, nearest_index)
    second_value = _gather_time_axis(flattened, second_index)
    nearest_position = positions.gather(1, nearest_index)
    second_position = positions.gather(1, second_index)
    derivative_denominator = nearest_position - second_position
    derivative_denominator = torch.where(
        derivative_denominator.ne(0),
        derivative_denominator,
        torch.ones_like(derivative_denominator),
    )
    derivative = (nearest_value - second_value) / expand(
        derivative_denominator.to(dtype=values.dtype)
    )
    raw_delta = query_axis - nearest_position
    max_delta = (nearest_position - second_position).abs().clamp_min(1)
    limited_delta = raw_delta.sign() * torch.minimum(raw_delta.abs(), max_delta)
    extrapolated_value = nearest_value + derivative * expand(limited_delta.to(dtype=values.dtype))

    bracketed = expand(flat_index(sparse_map.bracketed_mask))
    extrapolation = expand(flat_index(sparse_map.extrapolation_mask))
    valid = expand(flat_index(sparse_map.valid))
    result = torch.where(bracketed, bracketed_value, nearest_value)
    result = torch.where(extrapolation, extrapolated_value, result)
    if clamp_to_support:
        lower = torch.minimum(left_value, right_value)
        upper = torch.maximum(left_value, right_value)
        result = torch.where(bracketed, result.clamp(min=lower, max=upper), result)
        extrap_lower = torch.minimum(nearest_value, second_value)
        extrap_upper = torch.maximum(nearest_value, second_value)
        result = torch.where(
            extrapolation,
            result.clamp(min=extrap_lower, max=extrap_upper),
            result,
        )
    result = torch.where(valid, result, torch.zeros_like(result))
    return result.reshape(batch_size, 2, full_frames, *tail).permute(
        0,
        2,
        1,
        *range(3, 3 + len(tail)),
    )


def _gather_hand_time_axis(value: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if value.ndim < 3 or value.shape[2] != 2:
        raise ValueError("hand gather requires value [B, T, 2, ...]")
    if indices.ndim != 3 or indices.shape[0] != value.shape[0] or indices.shape[2] != 2:
        raise ValueError("hand gather indices must have shape [B, Q, 2]")
    batch_size, high_frames = value.shape[:2]
    tail = value.shape[3:]
    flattened = value.permute(0, 2, 1, *range(3, value.ndim)).reshape(
        batch_size * 2,
        high_frames,
        *tail,
    )
    flat_indices = indices.permute(0, 2, 1).reshape(batch_size * 2, -1)
    gathered = _gather_time_axis(flattened, flat_indices)
    return gathered.reshape(batch_size, 2, indices.shape[1], *tail).permute(
        0,
        2,
        1,
        *range(3, 3 + len(tail)),
    )


def _sparse_query_geometry(
    anchor_positions: torch.Tensor,
    anchor_present: torch.Tensor,
    *,
    full_frames: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, anchor_count = anchor_positions.shape
    device = anchor_positions.device
    if anchor_count == 0:
        empty_index = torch.zeros(batch_size, full_frames, device=device, dtype=torch.long)
        return (
            empty_index,
            empty_index,
            torch.zeros(batch_size, full_frames, device=device, dtype=torch.float32),
            torch.zeros(batch_size, full_frames, device=device, dtype=torch.bool),
        )
    query = torch.arange(full_frames, device=device, dtype=torch.long).view(1, full_frames, 1)
    positions = anchor_positions.unsqueeze(1)
    slots = torch.arange(anchor_count, device=device, dtype=torch.long).view(1, 1, anchor_count)
    present = anchor_present.unsqueeze(1)
    raw_left = torch.where(
        present & positions.le(query),
        slots,
        torch.full_like(slots, -1),
    ).amax(dim=2)
    raw_right = torch.where(
        present & positions.ge(query),
        slots,
        torch.full_like(slots, anchor_count),
    ).amin(dim=2)
    has_left = raw_left.ge(0)
    has_right = raw_right.lt(anchor_count)
    safe_left = raw_left.clamp(0, anchor_count - 1)
    safe_right = raw_right.clamp(0, anchor_count - 1)
    left = torch.where(has_left, safe_left, safe_right)
    right = torch.where(has_right, safe_right, safe_left)
    left_position = anchor_positions.gather(1, left)
    right_position = anchor_positions.gather(1, right)
    span = right_position - left_position
    alpha = torch.where(
        span.gt(0),
        (query[..., 0] - left_position).to(dtype=torch.float32)
        / span.clamp_min(1).to(dtype=torch.float32),
        torch.zeros(batch_size, full_frames, device=device, dtype=torch.float32),
    ).clamp(0.0, 1.0)
    return left, right, alpha, has_left | has_right


def _cubic_sparse_interpolation(
    values: torch.Tensor,
    positions: torch.Tensor,
    present: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    alpha: torch.Tensor,
    *,
    clamp_to_endpoints: bool = False,
) -> torch.Tensor:
    anchor_count = values.shape[1]
    if anchor_count == 0:
        return values.new_zeros(values.shape[0], left.shape[1], *values.shape[2:])
    left_value = _gather_time_axis(values, left)
    right_value = _gather_time_axis(values, right)
    previous = (left - 1).clamp_min(0)
    following = (right + 1).clamp_max(anchor_count - 1)
    previous_value = _gather_time_axis(values, previous)
    following_value = _gather_time_axis(values, following)
    left_position = positions.gather(1, left)
    right_position = positions.gather(1, right)
    previous_position = positions.gather(1, previous)
    following_position = positions.gather(1, following)
    previous_valid = present.gather(1, previous) & previous.ne(left)
    following_valid = present.gather(1, following) & following.ne(right)
    tail_dims = values.ndim - 2

    def expand(value: torch.Tensor) -> torch.Tensor:
        return value.reshape(*value.shape, *([1] * tail_dims))

    secant_span = expand((right_position - left_position).clamp_min(1).to(dtype=values.dtype))
    secant = (right_value - left_value) / secant_span
    left_slope = torch.where(
        expand(previous_valid),
        (right_value - previous_value)
        / expand((right_position - previous_position).clamp_min(1).to(dtype=values.dtype)),
        secant,
    )
    right_slope = torch.where(
        expand(following_valid),
        (following_value - left_value)
        / expand((following_position - left_position).clamp_min(1).to(dtype=values.dtype)),
        secant,
    )
    u = expand(alpha.to(dtype=values.dtype))
    u2 = u.square()
    u3 = u2 * u
    span = expand((right_position - left_position).to(dtype=values.dtype))
    interpolated = (
        (2.0 * u3 - 3.0 * u2 + 1.0) * left_value
        + (u3 - 2.0 * u2 + u) * span * left_slope
        + (-2.0 * u3 + 3.0 * u2) * right_value
        + (u3 - u2) * span * right_slope
    )
    same = left.eq(right)
    interpolated = torch.where(expand(same), left_value, interpolated)
    if clamp_to_endpoints:
        lower = torch.minimum(left_value, right_value)
        upper = torch.maximum(left_value, right_value)
        interpolated = interpolated.clamp(min=lower, max=upper)
    return interpolated


def _quaternion_multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    lx, ly, lz, lw = left.unbind(dim=-1)
    rx, ry, rz, rw = right.unbind(dim=-1)
    return torch.stack(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ),
        dim=-1,
    )


def _quaternion_inverse(value: torch.Tensor) -> torch.Tensor:
    normalized = _normalize_quaternion(value)
    return torch.cat((-normalized[..., :3], normalized[..., 3:]), dim=-1)


def _quaternion_log(value: torch.Tensor) -> torch.Tensor:
    normalized = _normalize_quaternion(value)
    vector = normalized[..., :3]
    vector_norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    angle = torch.atan2(vector_norm, normalized[..., 3:].clamp(-1.0, 1.0))
    return vector * angle / vector_norm.clamp_min(1e-8)


def _quaternion_exp(value: torch.Tensor) -> torch.Tensor:
    angle = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    scale = torch.sin(angle) / angle.clamp_min(1e-8)
    scale = torch.where(angle < 1e-6, 1.0 - angle.square() / 6.0, scale)
    return _normalize_quaternion(torch.cat((value * scale, torch.cos(angle)), dim=-1))


def _continuous_quaternion_controls(
    rotations: torch.Tensor,
    present: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    quaternions = rotation_matrix_to_quaternion(rotations)
    anchor_count = quaternions.shape[1]
    if anchor_count == 0:
        return quaternions, quaternions
    continuous: list[torch.Tensor] = [quaternions[:, 0]]
    for index in range(1, anchor_count):
        current = quaternions[:, index]
        previous = continuous[-1]
        current = torch.where((previous * current).sum(dim=-1, keepdim=True) < 0.0, -current, current)
        continuous.append(current)
    quaternion = torch.stack(continuous, dim=1)
    controls: list[torch.Tensor] = []
    for index in range(anchor_count):
        current = quaternion[:, index]
        if index == 0 or index == anchor_count - 1:
            controls.append(current)
            continue
        has_neighbors = present[:, index - 1] & present[:, index] & present[:, index + 1]
        inverse = _quaternion_inverse(current)
        left_delta = _quaternion_log(_quaternion_multiply(inverse, quaternion[:, index - 1]))
        right_delta = _quaternion_log(_quaternion_multiply(inverse, quaternion[:, index + 1]))
        control = _quaternion_multiply(current, _quaternion_exp(-0.25 * (left_delta + right_delta)))
        controls.append(torch.where(has_neighbors.unsqueeze(-1), control, current))
    return quaternion, torch.stack(controls, dim=1)


def _squad_sparse_interpolation(
    rotations: torch.Tensor,
    present: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    if rotations.shape[1] == 0:
        return torch.eye(3, device=rotations.device, dtype=rotations.dtype).reshape(1, 1, 3, 3).expand(
            rotations.shape[0], left.shape[1], -1, -1
        )
    work_dtype = _geometry_dtype(rotations.dtype)
    quaternion, controls = _continuous_quaternion_controls(rotations.to(dtype=work_dtype), present)
    left_quaternion = _gather_time_axis(quaternion, left)
    right_quaternion = _gather_time_axis(quaternion, right)
    left_control = _gather_time_axis(controls, left)
    right_control = _gather_time_axis(controls, right)
    endpoint_curve = quaternion_slerp(left_quaternion, right_quaternion, alpha)
    control_curve = quaternion_slerp(left_control, right_control, alpha)
    blend = 2.0 * alpha * (1.0 - alpha)
    squad = quaternion_slerp(endpoint_curve, control_curve, blend)
    squad = torch.where(left.eq(right).unsqueeze(-1), left_quaternion, squad)
    return quaternion_to_rotation_matrix(squad).to(dtype=rotations.dtype)


class DeterministicTemporalInterpolator(nn.Module):
    _XYZ_FIELDS = ("joint_xyz", "vertex_xyz")
    _INTERPOLATED_LOGIT_FIELDS = (
        "joint_visibility_logits",
        "vertex_visibility_logits",
        "joint_contact_logits",
        "vertex_contact_logits",
    )
    _BASE_ONLY_FIELDS = (
        "joint_contact_log_conf",
        "vertex_contact_log_conf",
        "joint_log_conf",
        "vertex_log_conf",
        "joint_contact_log_distance",
        "vertex_contact_log_distance",
    )

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def _masked(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        view = mask.reshape(*mask.shape, *([1] * (value.ndim - mask.ndim)))
        return torch.where(view, value, torch.zeros_like(value))

    def _camera_base(
        self,
        camera_pose_global: torch.Tensor,
        frame_map: MultiRateFrameMap,
        query_map: TemporalInterpolationQueryMap,
        camera_anchor_valid_global: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        camera_pose_global = camera_pose_global.to(dtype=_geometry_dtype(camera_pose_global.dtype))
        batch_size = camera_pose_global.shape[0]
        full_frames = query_map.output_present.shape[1]
        global_frames = camera_pose_global.shape[1]
        if global_frames == 0:
            identity = _identity_pose(
                (batch_size, full_frames),
                device=camera_pose_global.device,
                dtype=camera_pose_global.dtype,
            )
            return (
                identity,
                torch.zeros_like(query_map.output_present),
                identity,
                torch.zeros_like(query_map.output_present),
            )
        global_high = frame_map.global_anchor_indices.clamp(0, query_map.high_anchor_indices.shape[1] - 1)
        global_positions = query_map.high_anchor_indices.gather(1, global_high)
        global_present = camera_anchor_valid_global & frame_map.global_frame_present
        left, right, alpha, sparse_valid = _sparse_query_geometry(
            global_positions,
            global_present,
            full_frames=full_frames,
        )
        geometry_dtype = _geometry_dtype(camera_pose_global.dtype)
        c2w_global = w2c_to_c2w(camera_pose_global.to(dtype=geometry_dtype))
        rotation = _squad_sparse_interpolation(
            c2w_global[..., :3, :3],
            global_present,
            left,
            right,
            alpha,
        )
        center = _cubic_sparse_interpolation(
            c2w_global[..., :3, 3],
            global_positions,
            global_present,
            left,
            right,
            alpha,
        )
        base_c2w = _pose_from_rotation_translation(rotation, center)
        valid = sparse_valid & query_map.output_present
        identity = _identity_pose(
            (batch_size, full_frames),
            device=camera_pose_global.device,
            dtype=geometry_dtype,
        )
        base_c2w = torch.where(valid[..., None, None], base_c2w, identity)
        base_w2c = c2w_to_w2c(base_c2w)
        query_axis = torch.arange(
            full_frames,
            device=global_positions.device,
            dtype=torch.long,
        ).view(1, full_frames, 1)
        anchor_matches = (
            global_present.unsqueeze(1)
            & global_positions.unsqueeze(1).eq(query_axis)
        )
        anchor_exact_mask = anchor_matches.any(dim=-1) & query_map.output_present
        anchor_route_indices = anchor_matches.to(dtype=torch.long).argmax(dim=-1)
        anchor_pose = _gather_time_axis(camera_pose_global, anchor_route_indices)
        base_w2c = torch.where(
            anchor_exact_mask[..., None, None],
            anchor_pose,
            base_w2c,
        )
        return base_w2c, valid, anchor_pose, anchor_exact_mask

    def _hand_base(
        self,
        hand_outputs_high: dict[str, torch.Tensor],
        query_map: TemporalInterpolationQueryMap,
        sparse_map: SparseHandQueryMap,
    ) -> dict[str, torch.Tensor]:
        required = (*self._XYZ_FIELDS, *self._INTERPOLATED_LOGIT_FIELDS)
        missing = [name for name in required if name not in hand_outputs_high]
        if missing:
            raise ValueError(f"hand_outputs_high is missing required fields: {missing}")
        joint_high = hand_outputs_high["joint_xyz"]
        vertex_high = hand_outputs_high["vertex_xyz"]
        root_high = joint_high[..., :1, :]
        joint_local_high = joint_high - root_high
        vertex_local_high = vertex_high - root_high.detach()
        root_base = _sparse_hand_interpolation(
            root_high,
            query_map.high_anchor_indices,
            sparse_map,
            clamp_to_support=False,
        )
        joint_local_base = _sparse_hand_interpolation(
            joint_local_high,
            query_map.high_anchor_indices,
            sparse_map,
            clamp_to_support=True,
        )
        vertex_local_base = _sparse_hand_interpolation(
            vertex_local_high,
            query_map.high_anchor_indices,
            sparse_map,
            clamp_to_support=True,
        )
        geometry_base = {
            "joint_xyz": root_base + joint_local_base,
            "vertex_xyz": root_base.detach() + vertex_local_base,
        }
        base: dict[str, torch.Tensor] = {}
        fields = (*required, *(name for name in self._BASE_ONLY_FIELDS if name in hand_outputs_high))
        for name in fields:
            value = hand_outputs_high[name]
            if value.shape[:3] != (*query_map.high_anchor_indices.shape, 2):
                raise ValueError(f"hand_outputs_high[{name!r}] must align with [B, T_H, 2]")
            interpolated = geometry_base.get(name)
            if interpolated is None:
                interpolated = _sparse_hand_interpolation(
                    value,
                    query_map.high_anchor_indices,
                    sparse_map,
                    clamp_to_support=True,
                )
            interpolated = self._masked(interpolated, query_map.output_present)
            anchor_value = _gather_hand_time_axis(value, sparse_map.left_high)
            base[name] = torch.where(
                sparse_map.exact_mask.reshape(
                    *sparse_map.exact_mask.shape,
                    *([1] * (value.ndim - 3)),
                ),
                anchor_value,
                interpolated,
            )
        return base

    def forward(
        self,
        *,
        camera_pose_global: torch.Tensor,
        camera_anchor_valid_global: torch.Tensor | None = None,
        hand_outputs_high: dict[str, torch.Tensor],
        hand_anchor_valid_high: torch.Tensor | None = None,
        hand_anchor_quality_high: torch.Tensor | None = None,
        frame_map: MultiRateFrameMap,
        query_map: TemporalInterpolationQueryMap | None = None,
    ) -> FullRateRefinerOutput:
        batch_size, high_frames = frame_map.high_frame_present.shape
        if high_frames <= 0:
            raise ValueError("frame_map must contain at least one H frame")
        if tuple(frame_map.high_frame_present.shape) != (batch_size, high_frames):
            raise ValueError("frame_map H axes must align")
        if camera_pose_global.ndim != 4 or camera_pose_global.shape[0] != batch_size or camera_pose_global.shape[-2:] != (4, 4):
            raise ValueError("camera_pose_global must have shape [B, T_G, 4, 4]")
        if camera_pose_global.shape[1] != frame_map.global_frame_present.shape[1]:
            raise ValueError("camera_pose_global and frame_map G axes must align")
        if camera_anchor_valid_global is None:
            camera_anchor_valid_global = frame_map.global_frame_present
        if (
            tuple(camera_anchor_valid_global.shape) != tuple(frame_map.global_frame_present.shape)
            or camera_anchor_valid_global.dtype is not torch.bool
        ):
            raise ValueError("camera_anchor_valid_global must be bool with shape [B, T_G]")
        camera_anchor_valid_global = camera_anchor_valid_global.to(
            device=camera_pose_global.device,
            dtype=torch.bool,
        )
        if frame_map.high_frame_present.device != camera_pose_global.device:
            raise ValueError("frame_map and interpolation inputs must share a device")
        if query_map is None:
            query_map = build_temporal_interpolation_query_map(frame_map, interpolation_factor=1)
        query_map.validate(frame_map)
        if hand_anchor_valid_high is None:
            hand_anchor_valid_high = frame_map.high_frame_present.unsqueeze(-1).expand(-1, -1, 2)
        if tuple(hand_anchor_valid_high.shape) != (batch_size, high_frames, 2):
            raise ValueError("hand_anchor_valid_high must have shape [B, T_H, 2]")
        hand_anchor_valid_high = hand_anchor_valid_high.to(
            device=camera_pose_global.device,
            dtype=torch.bool,
        )
        joint_xyz = hand_outputs_high.get("joint_xyz")
        if not isinstance(joint_xyz, torch.Tensor):
            raise ValueError("hand_outputs_high must contain joint_xyz")
        if joint_xyz.device != camera_pose_global.device:
            raise ValueError("camera and hand interpolation inputs must share a device")
        if hand_anchor_quality_high is None:
            hand_anchor_quality_high = torch.ones(
                batch_size,
                high_frames,
                2,
                device=camera_pose_global.device,
                dtype=joint_xyz.dtype,
            )
        if tuple(hand_anchor_quality_high.shape) != (batch_size, high_frames, 2):
            raise ValueError("hand_anchor_quality_high must have shape [B, T_H, 2]")
        hand_anchor_quality_high = hand_anchor_quality_high.to(
            device=camera_pose_global.device,
            dtype=joint_xyz.dtype,
        )
        sparse_hand_map = build_sparse_hand_query_map(
            frame_map,
            query_map,
            hand_anchor_valid_high,
            hand_anchor_quality_high,
        )

        camera_base, camera_valid, _, camera_anchor_exact_mask = self._camera_base(
            camera_pose_global,
            frame_map,
            query_map,
            camera_anchor_valid_global,
        )

        hand_base = self._hand_base(hand_outputs_high, query_map, sparse_hand_map)
        high_root = hand_outputs_high["joint_xyz"][..., :1, :]
        relative_joint_high = hand_outputs_high["joint_xyz"] - high_root
        relative_vertex_high = hand_outputs_high["vertex_xyz"] - high_root.detach()
        relative_joint_proposal = _cubic_sparse_interpolation(
            relative_joint_high,
            query_map.high_anchor_indices,
            frame_map.high_frame_present,
            query_map.query_left_high,
            query_map.query_right_high,
            query_map.query_alpha,
            clamp_to_endpoints=True,
        )
        relative_vertex_proposal = _cubic_sparse_interpolation(
            relative_vertex_high,
            query_map.high_anchor_indices,
            frame_map.high_frame_present,
            query_map.query_left_high,
            query_map.query_right_high,
            query_map.query_alpha,
            clamp_to_endpoints=True,
        )
        absolute_support = sparse_hand_map.valid[..., None, None]
        hand_base["joint_xyz"] = torch.where(
            absolute_support,
            hand_base["joint_xyz"],
            relative_joint_proposal,
        )
        hand_base["vertex_xyz"] = torch.where(
            absolute_support,
            hand_base["vertex_xyz"],
            relative_vertex_proposal,
        )

        camera_residual_6d = camera_base.new_zeros(*camera_base.shape[:2], 9)
        hand_residual = {name: torch.zeros_like(base) for name, base in hand_base.items()}
        hand_refined = dict(hand_base)
        root_translation_valid = sparse_hand_map.valid & query_map.output_present.unsqueeze(-1)
        relative_hand_valid = (
            query_map.output_present.unsqueeze(-1).expand(-1, -1, 2)
            & torch.isfinite(hand_base["joint_xyz"]).all(dim=(-2, -1))
            & torch.isfinite(hand_base["vertex_xyz"]).all(dim=(-2, -1))
        )
        return FullRateRefinerOutput(
            query_map=query_map,
            camera_pose_base=camera_base,
            camera_pose_refined=camera_base,
            camera_residual_6d=camera_residual_6d,
            camera_anchor_exact_mask=camera_anchor_exact_mask,
            camera_valid=camera_valid,
            hand_base=hand_base,
            hand_refined=hand_refined,
            hand_residual=hand_residual,
            hand_anchor_exact_mask=sparse_hand_map.exact_mask,
            hand_valid=root_translation_valid & relative_hand_valid,
            root_translation_valid=root_translation_valid,
            relative_hand_valid=relative_hand_valid,
            hand_support_confidence=sparse_hand_map.support_confidence,
            hand_mode=sparse_hand_map.mode,
        )
