from dataclasses import asdict, dataclass
import math
import torch
from .models.camera_temporal_refiner import (
    DeterministicTemporalInterpolator,
    HandRefinedMode,
)


HAND_MISSING_FILL_MODE_BRACKETED = 1


HAND_MISSING_FILL_MODE_SINGLE_ANCHOR = 2


HAND_MISSING_FILL_MODE_CROSS_CHUNK = 3


HAND_MISSING_FILL_MODE_NO_ANCHOR = 4


HAND_ANCHOR_REASON_ROOT_UV_TEMPORAL_SPIKE = 16


_ROOT_UV_ISOLATED_RESIDUAL = 0.06


_ROOT_UV_NEIGHBOR_SPAN = 0.12


_ROOT_UV_ONE_SIDED_JUMP = 0.12


_ROOT_UV_LOW_CONFIDENCE = 0.5


@dataclass(frozen=True)
class HandAnchorOptions:
    enabled: bool = True
    min_confidence: float = .1
    max_extrapolation_seconds: float = .2
    fill_missing: bool = True
    smooth_local_shape: bool = True
    smooth_root_uv: bool = True

    def validate(self):
        if not math.isfinite(self.min_confidence) or not 0 < self.min_confidence <= 1:
            raise ValueError("hand anchor confidence must be in (0, 1]")
        if not math.isfinite(self.max_extrapolation_seconds) or self.max_extrapolation_seconds < 0:
            raise ValueError("hand extrapolation seconds must be finite and nonnegative")


def add_hand_anchor_arguments(parser):
    import argparse
    parser.add_argument("--hand-anchor-filter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hand-anchor-min-confidence", type=float, default=.1)
    parser.add_argument("--hand-max-extrapolation-seconds", type=float, default=.2)
    parser.add_argument(
        "--hand-fill-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="fill a missing camera-space hand whenever a valid same-side H anchor exists",
    )
    parser.add_argument(
        "--hand-local-smooth",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="lightly smooth the root-relative shape of the final R-rate hand output over three frames",
    )
    parser.add_argument(
        "--root-uv-smooth",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="correct low-confidence isolated wrist UV spikes before root solving; raw heatmap outputs stay untouched",
    )


def hand_anchor_options_from_args(args):
    options = HandAnchorOptions(
        enabled=args.hand_anchor_filter,
        min_confidence=args.hand_anchor_min_confidence,
        max_extrapolation_seconds=args.hand_max_extrapolation_seconds,
        fill_missing=bool(getattr(args, "hand_fill_missing", True)),
        smooth_local_shape=bool(getattr(args, "hand_local_smooth", True)),
        smooth_root_uv=bool(getattr(args, "root_uv_smooth", True)),
    )
    options.validate()
    if options.enabled and getattr(args, "smooth_hand_root_h", False):
        raise ValueError("Disable H Root smoothing when filtering raw H anchors; use --root-z-smooth instead")
    return options


def select_hand_anchors(root, raw_valid, confidence, times, wrist_uv, wrist_valid, options):
    options.validate()
    if root.ndim != 4 or root.shape[-2:] != (2, 3):
        raise ValueError("H roots must have shape [B,H,2,3]")
    if raw_valid.shape != root.shape[:-1] or confidence.shape != raw_valid.shape or times.shape != root.shape[:2]:
        raise ValueError("H validity, confidence and timestamps must align")
    if not bool(torch.isfinite(times).all()) or bool((times[:, 1:] <= times[:, :-1]).any()):
        raise ValueError("H timestamps must be finite and increasing")
    if wrist_uv.shape != (*raw_valid.shape, 2) or wrist_valid.shape != raw_valid.shape:
        raise ValueError("wrist UV and validity must align with H sides")
    finite = torch.isfinite(root).all(-1) & (root[..., 2] > 1e-4) & torch.isfinite(confidence)
    eligible = raw_valid & finite & (confidence >= options.min_confidence)
    reasons = torch.where(raw_valid & finite, 0, 1).to(torch.uint8)
    reasons |= (raw_valid & (~torch.isfinite(confidence) | (confidence < options.min_confidence))).to(torch.uint8) * 2
    collision = wrist_valid.all(-1) & torch.isfinite(wrist_uv).all(dim=(-1, -2))
    collision &= (wrist_uv[..., 0, :] - wrist_uv[..., 1, :]).norm(dim=-1) < .025
    for side in range(2):
        weak = (confidence[..., side] < .5) & (confidence[..., 1-side] > 2 * confidence[..., side])
        reject = collision & weak & raw_valid[..., 1-side]
        eligible[..., side] &= ~reject
        reasons[..., side] |= reject.to(torch.uint8) * 4
    spikes = torch.zeros_like(eligible)
    for batch in range(root.shape[0]):
        for side in range(2):
            indices = torch.where(eligible[batch, :, side])[0]
            previous, current, following = indices[:-2], indices[1:-1], indices[2:]
            t0, t1, t2 = (times[batch, selected] for selected in (previous, current, following))
            z0, z1, z2 = (root[batch, selected, side, 2].double() for selected in (previous, current, following))
            estimate = torch.exp(torch.lerp(z0.log(), z2.log(), (t1-t0).double()/(t2-t0).double()))
            isolated = (t1-t0 <= .25) & (t2-t1 <= .25) & ((z1-z0)*(z2-z1) < 0)
            isolated &= (z0-z2).abs() < .1
            isolated &= (z1-estimate).abs() > .05
            isolated &= (z1/estimate).log().abs() > math.log(1.35)
            spikes[batch, current, side] = isolated
    reasons |= spikes.to(torch.uint8) * 8
    uv_spikes = torch.zeros_like(eligible)
    for batch in range(root.shape[0]):
        for side in range(2):
            uv = wrist_uv[batch, :, side].float()
            side_valid = eligible[batch, :, side] & wrist_valid[batch, :, side]
            side_valid &= torch.isfinite(uv).all(dim=-1)
            indices = torch.where(side_valid)[0]
            if indices.numel() < 2:
                continue
            previous, current = indices[:-1], indices[1:]
            dt = times[batch, current] - times[batch, previous]
            jump = torch.linalg.vector_norm(uv[current] - uv[previous], dim=-1)
            one_sided = (
                (dt <= .25)
                & (jump > _ROOT_UV_ONE_SIDED_JUMP)
                & (confidence[batch, current, side] < _ROOT_UV_LOW_CONFIDENCE)
            )
            uv_spikes[batch, current, side] |= one_sided
            if indices.numel() >= 3:
                previous, current, following = indices[:-2], indices[1:-1], indices[2:]
                t0, t1, t2 = (times[batch, selected] for selected in (previous, current, following))
                ratio = ((t1 - t0) / (t2 - t0).clamp_min(1e-6)).float().unsqueeze(-1)
                predicted = torch.lerp(uv[previous], uv[following], ratio)
                residual = torch.linalg.vector_norm(uv[current] - predicted, dim=-1)
                neighbor_span = torch.linalg.vector_norm(uv[previous] - uv[following], dim=-1)
                isolated = (
                    (t1 - t0 <= .25)
                    & (t2 - t1 <= .25)
                    & (neighbor_span < _ROOT_UV_NEIGHBOR_SPAN)
                    & (residual > _ROOT_UV_ISOLATED_RESIDUAL)
                    & (confidence[batch, current, side] < _ROOT_UV_LOW_CONFIDENCE)
                )
                uv_spikes[batch, current, side] |= isolated
    reasons |= uv_spikes.to(torch.uint8) * HAND_ANCHOR_REASON_ROOT_UV_TEMPORAL_SPIKE
    return eligible & ~spikes & ~uv_spikes, reasons


def filter_hand_outputs(outputs, frame_map, query_map, fps, options):
    if options is None or not options.enabled:
        return outputs
    options.validate()
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("inference FPS must be positive")
    required = ("dense_joint_xyz", "dense_vertex_xyz", "root_translation_valid",
                "root_translation_reported_confidence", "predicted_keypoint_uv", "predicted_keypoint_mask")
    missing = [key for key in required if key not in outputs]
    if missing:
        raise ValueError(f"Hand anchor filtering requires raw diagnostics: {missing}")
    raw_valid = outputs["root_translation_valid"] & frame_map.high_frame_present.unsqueeze(-1)
    geometry_valid = torch.isfinite(outputs["dense_joint_xyz"]).all(dim=(-1, -2))
    geometry_valid &= torch.isfinite(outputs["dense_vertex_xyz"]).all(dim=(-1, -2))
    confidence = outputs["root_translation_reported_confidence"].float()
    times = query_map.high_anchor_indices.double() / fps
    accepted, reasons = select_hand_anchors(
        outputs["dense_joint_xyz"][..., 0, :], raw_valid & geometry_valid, confidence, times,
        outputs.get("root_solver_keypoint_uv", outputs["predicted_keypoint_uv"])[..., 0, :],
        outputs.get("root_solver_keypoint_mask", outputs["predicted_keypoint_mask"])[..., 0],
        options,
    )
    raw_anchor = outputs.get("interpolation_hand_anchor_valid_high")
    reuse_existing = (
        isinstance(raw_anchor, torch.Tensor)
        and tuple(raw_anchor.shape) == tuple(accepted.shape)
        and torch.equal(raw_anchor.to(device=accepted.device, dtype=torch.bool), accepted)
        and all(
            name in outputs
            for name in (
                "dense_joint_xyz_refined_full",
                "dense_vertex_xyz_refined_full",
                "hand_refined_valid_full",
                "root_translation_valid_full",
            )
        )
    )
    result = dict(outputs)
    if reuse_existing:
        hand_valid = outputs["hand_refined_valid_full"].to(dtype=torch.bool)
        root_translation_valid = outputs["root_translation_valid_full"].to(dtype=torch.bool)
        hand_support_confidence = outputs.get(
            "hand_refined_support_confidence_full",
            torch.zeros_like(outputs["hand_refined_valid_full"], dtype=torch.float32),
        )
        hand_mode = outputs.get(
            "hand_refined_mode_full",
            torch.zeros_like(outputs["hand_refined_valid_full"], dtype=torch.long),
        )
    else:
        interpolator = DeterministicTemporalInterpolator()
        fields = (*interpolator._XYZ_FIELDS, *interpolator._INTERPOLATED_LOGIT_FIELDS, *interpolator._BASE_ONLY_FIELDS)
        high = {field: torch.nan_to_num(outputs[f"dense_{field}"], nan=0., posinf=0., neginf=0.)
                for field in fields if f"dense_{field}" in outputs}
        rebuilt = interpolator(
            camera_pose_global=outputs["camera_pose_metric_global"],
            camera_anchor_valid_global=outputs["camera_pose_metric_valid_global"],
            hand_outputs_high=high, hand_anchor_valid_high=accepted,
            hand_anchor_quality_high=confidence.clamp(0, 1), frame_map=frame_map, query_map=query_map,
        )
        for key, value in rebuilt.to_public_dict().items():
            if key.startswith(("dense_", "hand_", "root_translation_valid_full", "relative_hand_valid_full")):
                result[key] = value
        hand_valid = rebuilt.hand_valid
        root_translation_valid = rebuilt.root_translation_valid
        hand_support_confidence = rebuilt.hand_support_confidence
        hand_mode = rebuilt.hand_mode
    queries = torch.arange(query_map.output_present.shape[1], device=times.device).double() / fps
    distance = (queries[None, :, None, None] - times[:, None, :, None]).abs()
    nearest = torch.where(accepted[:, None], distance, torch.inf).amin(dim=2)
    one_sided = (hand_mode == int(HandRefinedMode.ONE_SIDED_EXTRAPOLATION)) | (hand_mode == int(HandRefinedMode.SINGLE_ANCHOR_HOLD))
    unsupported = (
        one_sided
        & (nearest > options.max_extrapolation_seconds + 1e-8)
    )
    interpolated_root = result["dense_joint_xyz_refined_full"][..., 0, :]
    unsupported |= hand_valid & (~torch.isfinite(interpolated_root).all(-1) | (interpolated_root[..., 2] <= 1e-4))
    unsupported = unsupported.to(dtype=torch.bool)
    hand_valid = hand_valid & ~unsupported
    root_translation_valid = root_translation_valid & ~unsupported
    hand_support_confidence = torch.where(unsupported, 0., hand_support_confidence)
    hand_mode = torch.where(unsupported, int(HandRefinedMode.NO_ANCHOR_INVALID), hand_mode)
    root = result["dense_joint_xyz_refined_full"][..., :1, :]
    for field in ("joint_xyz", "vertex_xyz"):
        public_name = f"dense_{field}_refined_full"
        value = result[public_name]
        value = torch.where(unsupported[..., None, None], value-root, value)
        result[public_name] = value
    result["hand_refined_valid_full"] = hand_valid
    result["root_translation_valid_full"] = root_translation_valid
    result["hand_refined_support_confidence_full"] = hand_support_confidence
    result["hand_refined_mode_full"] = hand_mode
    result.update(
        interpolation_hand_anchor_valid_high=accepted,
        interpolation_hand_anchor_quality_high=torch.where(accepted, confidence, 0.),
        hand_anchor_filter_raw_valid_high=raw_valid,
        hand_anchor_filter_rejection_reason_high=reasons,
        hand_anchor_filter_unsupported_full=unsupported,
    )
    if options.fill_missing:
        accepted_side = accepted & frame_map.high_frame_present.unsqueeze(-1)
        high_count = accepted_side.sum(dim=1, keepdim=True)
        safe_left = query_map.query_left_high.clamp(0, accepted_side.shape[1] - 1)
        safe_right = query_map.query_right_high.clamp(0, accepted_side.shape[1] - 1)
        left_valid = accepted_side.gather(
            1, safe_left.unsqueeze(-1).expand(-1, -1, accepted_side.shape[-1])
        )
        right_valid = accepted_side.gather(
            1, safe_right.unsqueeze(-1).expand(-1, -1, accepted_side.shape[-1])
        )
        output_present = query_map.output_present.unsqueeze(-1).expand_as(left_valid)
        missing_fill = output_present & high_count.gt(0) & ~(left_valid & right_valid)
        fill_mode = torch.zeros_like(missing_fill, dtype=torch.long)
        fill_mode = torch.where(
            missing_fill & high_count.eq(1),
            torch.full_like(fill_mode, 2),
            fill_mode,
        )
        fill_mode = torch.where(
            missing_fill & high_count.gt(1),
            torch.full_like(fill_mode, 1),
            fill_mode,
        )
        filled_joint, filled_vertex, filled_valid, filled_mode = _camera_space_smooth_fill(
            result["dense_joint_xyz_refined_full"],
            result["dense_vertex_xyz_refined_full"],
            hand_valid,
        )
        new_fill = filled_valid & ~hand_valid & output_present
        result["dense_joint_xyz_refined_full"] = torch.where(
            new_fill[..., None, None], filled_joint, result["dense_joint_xyz_refined_full"]
        )
        result["dense_vertex_xyz_refined_full"] = torch.where(
            new_fill[..., None, None], filled_vertex, result["dense_vertex_xyz_refined_full"]
        )
        missing_fill |= new_fill
        fill_mode = torch.where(new_fill, filled_mode, fill_mode)
    else:
        missing_fill = torch.zeros_like(hand_valid)
        fill_mode = torch.zeros_like(hand_valid, dtype=torch.long)
    display_valid = hand_valid | missing_fill
    result["hand_missing_fill_full"] = missing_fill
    result["hand_missing_fill_mode_full"] = fill_mode
    result["hand_display_valid_full"] = display_valid
    result["hand_display_joint_xyz_full"] = result["dense_joint_xyz_refined_full"]
    result["hand_display_vertex_xyz_full"] = result["dense_vertex_xyz_refined_full"]
    if options.smooth_local_shape:
        display_joint, display_vertex, applied = smooth_display_hand_local_shape(
            result["hand_display_joint_xyz_full"],
            result["hand_display_vertex_xyz_full"],
            display_valid,
            result.get("hand_anchor_exact_mask_full"),
        )
        result["dense_joint_xyz_refined_full"] = display_joint
        result["dense_vertex_xyz_refined_full"] = display_vertex
        result["hand_display_joint_xyz_full"] = display_joint
        result["hand_display_vertex_xyz_full"] = display_vertex
        result["hand_display_local_smoothing_applied_full"] = applied
    else:
        result["hand_display_local_smoothing_applied_full"] = torch.zeros_like(display_valid)
    for field in ("joint", "vertex"):
        final_geometry = result[f"dense_{field}_xyz_refined_full"]
        for name in (f"dense_{field}_xyz_interpolated_full", f"hand_{field}_xyz_metric_full"):
            if name in result:
                result[name] = final_geometry
    return result


def _camera_space_smooth_fill(
    joint_xyz: torch.Tensor,
    vertex_xyz: torch.Tensor,
    trusted_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if joint_xyz.ndim != 5 or vertex_xyz.ndim != 5 or trusted_valid.ndim != 3:
        raise ValueError("camera-space fill expects [B,T,2,...] tensors")
    if joint_xyz.shape[:3] != vertex_xyz.shape[:3] or joint_xyz.shape[:3] != trusted_valid.shape:
        raise ValueError("camera-space fill tensors must align on [B,T,2]")
    batch_size, frame_count, side_count = trusted_valid.shape
    filled_joint = joint_xyz.clone()
    filled_vertex = vertex_xyz.clone()
    fill_mask = torch.zeros_like(trusted_valid)
    fill_mode = torch.zeros_like(trusted_valid, dtype=torch.long)
    for batch_index in range(batch_size):
        for side_index in range(side_count):
            valid = trusted_valid[batch_index, :, side_index].clone()
            valid &= torch.isfinite(joint_xyz[batch_index, :, side_index]).all(dim=(-2, -1))
            valid &= torch.isfinite(vertex_xyz[batch_index, :, side_index]).all(dim=(-2, -1))
            valid_indices = torch.where(valid)[0]
            if valid_indices.numel() == 0:
                fill_mode[batch_index, :, side_index] = HAND_MISSING_FILL_MODE_NO_ANCHOR
                continue
            for frame_index in range(frame_count):
                if bool(valid[frame_index]):
                    continue
                previous = valid_indices[valid_indices < frame_index]
                following = valid_indices[valid_indices > frame_index]
                if previous.numel() and following.numel():
                    left = int(previous[-1].item())
                    right = int(following[0].item())
                    alpha = (frame_index - left) / float(max(right - left, 1))
                    smooth_alpha = alpha * alpha * (3.0 - 2.0 * alpha)
                    left_root = joint_xyz[batch_index, left, side_index, 0]
                    right_root = joint_xyz[batch_index, right, side_index, 0]
                    left_joint_local = joint_xyz[batch_index, left, side_index] - left_root
                    right_joint_local = joint_xyz[batch_index, right, side_index] - right_root
                    left_vertex_local = vertex_xyz[batch_index, left, side_index] - left_root.detach()
                    right_vertex_local = vertex_xyz[batch_index, right, side_index] - right_root.detach()
                    root = torch.lerp(left_root, right_root, smooth_alpha)
                    joint_local = torch.lerp(left_joint_local, right_joint_local, smooth_alpha)
                    vertex_local = torch.lerp(left_vertex_local, right_vertex_local, smooth_alpha)
                    filled_joint[batch_index, frame_index, side_index] = root + joint_local
                    filled_vertex[batch_index, frame_index, side_index] = root.detach() + vertex_local
                    fill_mode[batch_index, frame_index, side_index] = HAND_MISSING_FILL_MODE_BRACKETED
                else:
                    nearest = int((previous[-1] if previous.numel() else following[0]).item())
                    filled_joint[batch_index, frame_index, side_index] = joint_xyz[batch_index, nearest, side_index]
                    filled_vertex[batch_index, frame_index, side_index] = vertex_xyz[batch_index, nearest, side_index]
                    fill_mode[batch_index, frame_index, side_index] = HAND_MISSING_FILL_MODE_SINGLE_ANCHOR
                fill_mask[batch_index, frame_index, side_index] = True
    display_valid = trusted_valid | fill_mask
    return filled_joint, filled_vertex, display_valid, fill_mode


def smooth_display_hand_local_shape(
    joint_xyz: torch.Tensor,
    vertex_xyz: torch.Tensor,
    valid: torch.Tensor,
    exact_anchor: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if joint_xyz.ndim != 5 or vertex_xyz.ndim != 5 or valid.ndim != 3:
        raise ValueError("display Hand smoothing expects [B,T,2,...] tensors")
    if joint_xyz.shape[:3] != vertex_xyz.shape[:3] or joint_xyz.shape[:3] != valid.shape:
        raise ValueError("display Hand smoothing tensors must align on [B,T,2]")
    if exact_anchor is None:
        exact_anchor = torch.zeros_like(valid)
    if exact_anchor.shape != valid.shape:
        raise ValueError("exact_anchor must align with valid")
    smoothed_joint = joint_xyz.clone()
    smoothed_vertex = vertex_xyz.clone()
    applied = torch.zeros_like(valid)
    for batch_index in range(valid.shape[0]):
        for frame_index in range(1, valid.shape[1] - 1):
            support = valid[batch_index, frame_index - 1] & valid[batch_index, frame_index] & valid[batch_index, frame_index + 1]
            support &= ~exact_anchor[batch_index, frame_index]
            for side_index in torch.where(support)[0].tolist():
                root = joint_xyz[batch_index, frame_index, side_index, 0]
                local_joint = torch.stack(
                    (
                        joint_xyz[batch_index, frame_index - 1, side_index] - joint_xyz[batch_index, frame_index - 1, side_index, :1],
                        joint_xyz[batch_index, frame_index, side_index] - root,
                        joint_xyz[batch_index, frame_index + 1, side_index] - joint_xyz[batch_index, frame_index + 1, side_index, :1],
                    ),
                    dim=0,
                )
                local_vertex = torch.stack(
                    (
                        vertex_xyz[batch_index, frame_index - 1, side_index] - joint_xyz[batch_index, frame_index - 1, side_index, :1].detach(),
                        vertex_xyz[batch_index, frame_index, side_index] - root.detach(),
                        vertex_xyz[batch_index, frame_index + 1, side_index] - joint_xyz[batch_index, frame_index + 1, side_index, :1].detach(),
                    ),
                    dim=0,
                )
                weights = joint_xyz.new_tensor([0.25, 0.5, 0.25]).reshape(3, 1, 1)
                smoothed_joint[batch_index, frame_index, side_index] = root + (local_joint * weights).sum(dim=0)
                smoothed_vertex[batch_index, frame_index, side_index] = root.detach() + (local_vertex * weights).sum(dim=0)
                applied[batch_index, frame_index, side_index] = True
    return smoothed_joint, smoothed_vertex, applied


def hand_anchor_filter_payload(result, options):
    if options is None:
        return None
    payload = {"schema_version": "egofound3r.hand_anchor_filter.v1", "options": asdict(options),
               "reason_bits": {"invalid_geometry_or_root": 1, "low_confidence": 2, "wrist_collision": 4, "isolated_depth_spike": 8,
                                "root_uv_temporal_spike": HAND_ANCHOR_REASON_ROOT_UV_TEMPORAL_SPIKE}}
    if options.enabled:
        outputs = result.outputs
        payload.update(frame_indices=result.high_source_indices,
                       raw_valid=outputs["hand_anchor_filter_raw_valid_high"],
                       accepted=outputs["interpolation_hand_anchor_valid_high"],
                       rejection_reason=outputs["hand_anchor_filter_rejection_reason_high"],
                       unsupported_full=outputs["hand_anchor_filter_unsupported_full"])
        if "root_uv_smoothing_applied" in outputs:
            payload["root_uv_smoothing_applied_high"] = outputs["root_uv_smoothing_applied"]
    return payload
