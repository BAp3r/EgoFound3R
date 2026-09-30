from __future__ import annotations


import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any
import torch
from egohandmetric_prompt.metric_geometry import (
    scene_average_distance_from_depth,
)
from egohandmetric_prompt.models.camera_temporal_refiner import (
    _geometry_dtype,
    clip_local_metric_w2c,
)


INFERENCE_METRIC_SCHEMA_VERSION = "egofound3r.metric_inference.v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_value(repo_root: Path, *arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = completed.stdout.strip()
    return value or None


def build_inference_provenance(
    *,
    config_path: Path,
    checkpoint_path: Path,
    video_path: Path,
    decoded_video_fps: float,
    effective_r_fps: float,
    model_input_size_hw: tuple[int, int],
    source_video_size_hw: tuple[int, int] | None,
    temporal_architecture_id: str,
    command_argv: list[str],
    repo_root: Path,
) -> dict[str, Any]:
    config = config_path.resolve()
    checkpoint = checkpoint_path.resolve()
    video = video_path.resolve()
    commit = _git_value(repo_root, "rev-parse", "HEAD")
    branch = _git_value(repo_root, "branch", "--show-current")
    dirty_status = _git_value(repo_root, "status", "--porcelain")
    return {
        "schema_version": INFERENCE_METRIC_SCHEMA_VERSION,
        "config_path": str(config),
        "config_sha256": file_sha256(config),
        "checkpoint_path": str(checkpoint),
        "checkpoint_size_bytes": int(checkpoint.stat().st_size),
        "video_path": str(video),
        "temporal_architecture_id": str(temporal_architecture_id),
        "git_commit": commit,
        "git_branch": branch,
        "git_dirty": bool(dirty_status),
        "decoded_video_fps": float(decoded_video_fps),
        "effective_r_fps": float(effective_r_fps),
        "model_input_size_hw": [
            int(model_input_size_hw[0]),
            int(model_input_size_hw[1]),
        ],
        "source_video_size_hw": (
            None
            if source_video_size_hw is None
            else [int(source_video_size_hw[0]), int(source_video_size_hw[1])]
        ),
        "command_argv": [str(value) for value in command_argv],
    }


def prepare_inference_output_dir(
    output_dir: Path,
    *,
    allow_overwrite: bool,
) -> None:
    managed_paths = (
        output_dir / "inference_output.pt",
        output_dir / "summary.json",
        output_dir / "overlay.mp4",
        output_dir / "overlay_frames",
        output_dir / "overlay_multimode.mp4",
        output_dir / "overlay_multimode_frames",
    )
    existing = [path for path in managed_paths if path.exists()]
    if existing and not allow_overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            "inference output already exists; use a new --output-dir or "
            f"explicit --allow-output-overwrite: {joined}"
        )
    if allow_overwrite:
        for path in existing:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)


def save_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _first_tensor(values: Mapping[str, Any], *names: str) -> torch.Tensor | None:
    for name in names:
        value = values.get(name)
        if isinstance(value, torch.Tensor):
            return value
    return None


def _scale_vector(
    value: torch.Tensor | None,
    *,
    count: int,
    default: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    if value is None:
        return torch.full((count,), default, dtype=dtype)
    result = value.detach().cpu().to(dtype=dtype).reshape(-1)
    if result.numel() != count:
        raise ValueError(f"metric-scale value must contain {count} element(s)")
    return result


def _apply_batch_scale(tensor: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if tensor.shape[0] != scale.shape[0]:
        raise ValueError(
            f"cannot apply metric scale {tuple(scale.shape)} to tensor {tuple(tensor.shape)}"
        )
    return tensor * scale.to(device=tensor.device, dtype=tensor.dtype).reshape(
        tensor.shape[0],
        *([1] * (tensor.ndim - 1)),
    )


def reconstruct_multirate_metric_scene_outputs(
    outputs: Mapping[str, Any],
    *,
    global_frame_present: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> dict[str, Any]:
    if eps <= 0:
        raise ValueError("eps must be positive")
    depth_raw = _first_tensor(outputs, "depth_raw_global", "depth_global")
    intrinsics = _first_tensor(outputs, "intrinsics_K_global", "intrinsics_global")
    camera_raw = _first_tensor(outputs, "camera_pose_raw_global", "camera_pose_global")
    metric_value = _first_tensor(outputs, "metric_value_pred", "metric_value")
    if depth_raw is None or depth_raw.ndim not in (4, 5):
        raise ValueError("multi-rate metric reconstruction requires raw G-rate depth")
    if depth_raw.ndim == 5 and depth_raw.shape[-1] != 1:
        raise ValueError("five-dimensional G-rate depth must end in one channel")
    batch_size, global_frames = depth_raw.shape[:2]
    if intrinsics is None or tuple(intrinsics.shape) != (batch_size, global_frames, 3, 3):
        raise ValueError("multi-rate metric reconstruction requires G-rate K [B, T_G, 3, 3]")
    if camera_raw is None or tuple(camera_raw.shape) != (batch_size, global_frames, 4, 4):
        raise ValueError(
            "multi-rate metric reconstruction requires raw G-rate W2C [B, T_G, 4, 4]"
        )
    if metric_value is None or tuple(metric_value.shape) != (batch_size,):
        raise ValueError("multi-rate metric reconstruction requires metric_value [B]")
    depth_raw = depth_raw.to(dtype=_geometry_dtype(depth_raw.dtype))
    intrinsics = intrinsics.to(dtype=_geometry_dtype(intrinsics.dtype))
    camera_raw = camera_raw.to(dtype=_geometry_dtype(camera_raw.dtype))

    if global_frame_present is None:
        present = torch.ones(
            batch_size,
            global_frames,
            device=depth_raw.device,
            dtype=torch.bool,
        )
    else:
        if tuple(global_frame_present.shape) != (batch_size, global_frames):
            raise ValueError("global_frame_present must have shape [B, T_G]")
        present = global_frame_present.to(device=depth_raw.device, dtype=torch.bool)
    depth_values_raw = depth_raw.squeeze(-1) if depth_raw.ndim == 5 else depth_raw
    depth_frame_support = (
        torch.isfinite(depth_values_raw) & (depth_values_raw > 0)
    ).flatten(2).any(dim=2)
    intrinsics_frame_support = torch.isfinite(intrinsics).all(dim=(-2, -1))
    intrinsics_frame_support &= intrinsics[..., 0, 0].abs() > float(eps)
    intrinsics_frame_support &= intrinsics[..., 1, 1].abs() > float(eps)
    camera_frame_support = torch.isfinite(camera_raw).all(dim=(-2, -1))
    camera_frame_support &= torch.linalg.det(camera_raw.float()).abs() > float(eps)
    scene_support_valid = (
        present
        & depth_frame_support
        & intrinsics_frame_support
        & camera_frame_support
    ).any(dim=1)

    scene_scale = _first_tensor(
        outputs,
        "interpolation_pred_scene_scale_global",
        "pred_scene_scale_raw",
        "s_pred_scene",
    )
    if scene_scale is None:
        scene_scale = scene_average_distance_from_depth(
            depth_raw.detach(),
            intrinsics.detach(),
            camera_raw.detach(),
            present.unsqueeze(-1).unsqueeze(-1),
        )
    if tuple(scene_scale.shape) != (batch_size,):
        raise ValueError("predicted raw scene scale must have shape [B]")
    scene_scale = scene_scale.detach().to(device=depth_raw.device, dtype=depth_raw.dtype)
    metric_value = metric_value.detach().to(device=depth_raw.device, dtype=depth_raw.dtype)
    metric_value_effective = _first_tensor(outputs, "metric_value_effective")
    if metric_value_effective is None:
        metric_value_effective = metric_value
    if tuple(metric_value_effective.shape) != (batch_size,):
        raise ValueError("effective metric value must have shape [B]")
    metric_value_effective = metric_value_effective.detach().to(device=depth_raw.device, dtype=depth_raw.dtype)

    factor = _first_tensor(
        outputs,
        "interpolation_scene_metric_scale_factor",
        "raw_to_meter_pred",
        "metric_scale_factor",
    )
    if factor is None:
        factor = metric_value_effective / scene_scale.clamp_min(float(eps))
    if tuple(factor.shape) != (batch_size,):
        raise ValueError("raw-to-meter factor must have shape [B]")
    factor = factor.detach().to(device=depth_raw.device, dtype=depth_raw.dtype)

    scale_valid = _first_tensor(
        outputs,
        "interpolation_scene_metric_scale_valid",
        "metric_scale_valid",
    )
    derived_valid = (
        scene_support_valid
        & torch.isfinite(scene_scale)
        & (scene_scale > float(eps))
        & torch.isfinite(metric_value_effective)
        & (metric_value_effective > 0)
        & torch.isfinite(factor)
        & (factor > 0)
    )
    if scale_valid is None:
        scale_valid = derived_valid
    else:
        if tuple(scale_valid.shape) != (batch_size,):
            raise ValueError("metric scale validity must have shape [B]")
        scale_valid = scale_valid.to(device=depth_raw.device, dtype=torch.bool) & derived_valid
    safe_factor = torch.where(scale_valid, factor, torch.ones_like(factor)).detach()

    metric_depth = _apply_batch_scale(depth_raw.detach(), safe_factor)
    camera_anchor_valid = (
        present
        & scale_valid.unsqueeze(-1)
        & camera_frame_support
    )
    depth_values = metric_depth.squeeze(-1) if metric_depth.ndim == 5 else metric_depth
    depth_metric_valid = (
        camera_anchor_valid
        & torch.isfinite(depth_values).flatten(2).any(dim=2)
        & (torch.isfinite(depth_values) & (depth_values > 0)).flatten(2).any(dim=2)
    )
    metric_camera = clip_local_metric_w2c(
        camera_raw.detach(),
        camera_anchor_valid,
        translation_scale=safe_factor,
    )

    reconstructed = dict(outputs)
    reconstructed.update(
        {
            "depth_raw_global": depth_raw,
            "camera_pose_raw_global": camera_raw,
            "depth_metric_global": metric_depth,
            "depth_metric_valid_global": depth_metric_valid,
            "camera_pose_metric_global": metric_camera,
            "camera_pose_metric_valid_global": camera_anchor_valid,
            "metric_value_pred": metric_value,
            "pred_scene_scale_raw": scene_scale,
            "raw_to_meter_pred": safe_factor,
            "s_pred_scene": scene_scale,
            "metric_scale_factor": safe_factor,
            "metric_scale_valid": scale_valid.detach(),
            "depth_global": metric_depth,
            "camera_pose_global": metric_camera,
            "intrinsics_global": intrinsics,
            "intrinsics_K_global": intrinsics,
        }
    )
    return reconstructed


def build_metric_prediction_contract(
    outputs: Mapping[str, Any],
    *,
    depth_frame_indices: torch.Tensor,
    world_scope: str,
    depth_metric_global: torch.Tensor | None = None,
    depth_metric_valid_global: torch.Tensor | None = None,
    depth_confidence_global: torch.Tensor | None = None,
    intrinsics_K_global: torch.Tensor | None = None,
    intrinsics_K_full: torch.Tensor | None = None,
    metric_value_pred: torch.Tensor | None = None,
    pred_scene_scale_raw: torch.Tensor | None = None,
    raw_to_meter_pred: torch.Tensor | None = None,
    raw_to_meter_fused: torch.Tensor | None = None,
    overlap_residual_scale: torch.Tensor | None = None,
    raw_to_meter_effective: torch.Tensor | None = None,
    metric_scale_valid: torch.Tensor | None = None,
) -> dict[str, Any]:
    if world_scope not in {"clip_local_world", "stitched_video_world"}:
        raise ValueError("world_scope must identify the clip-local or stitched-video gauge")
    camera = _first_tensor(
        outputs,
        "camera_pose_metric_full",
        "camera_pose_interpolated_full",
        "camera_pose_refined_full",
    )
    if camera is None or camera.ndim != 3 or camera.shape[-2:] != (4, 4):
        raise ValueError("metric inference requires R-rate camera pose [T_R, 4, 4]")
    camera = camera.to(dtype=_geometry_dtype(camera.dtype))
    full_frames = int(camera.shape[0])
    camera_valid = _first_tensor(
        outputs,
        "camera_pose_metric_valid_full",
        "camera_interpolation_valid_full",
        "camera_refined_valid_full",
    )
    if camera_valid is None:
        camera_valid = torch.isfinite(camera).all(dim=(-2, -1))
    camera_valid = camera_valid.detach().cpu().to(dtype=torch.bool)
    if tuple(camera_valid.shape) != (full_frames,):
        raise ValueError("camera validity must have shape [T_R]")

    depth = depth_metric_global
    if depth is None:
        depth = _first_tensor(outputs, "depth_metric_global", "depth_global")
    intrinsics_global = intrinsics_K_global
    if intrinsics_global is None:
        intrinsics_global = _first_tensor(outputs, "intrinsics_K_global", "intrinsics_global")
    intrinsics_full = intrinsics_K_full
    if intrinsics_full is None:
        intrinsics_full = _first_tensor(outputs, "intrinsics_K_full", "intrinsics_full")
    if depth is None or depth.ndim not in (3, 4):
        raise ValueError("metric inference requires native G-rate depth")
    global_frames = int(depth.shape[0])
    depth_indices = depth_frame_indices.detach().cpu().to(dtype=torch.long).reshape(-1)
    if depth_indices.numel() != global_frames:
        raise ValueError("depth_frame_indices must align with the native G axis")
    if intrinsics_global is None or tuple(intrinsics_global.shape) != (global_frames, 3, 3):
        raise ValueError("K_global must have shape [T_G, 3, 3]")
    if intrinsics_full is None or tuple(intrinsics_full.shape) != (full_frames, 3, 3):
        raise ValueError("K_full must have shape [T_R, 3, 3]")
    depth = depth.to(dtype=_geometry_dtype(depth.dtype))
    intrinsics_global = intrinsics_global.to(dtype=_geometry_dtype(intrinsics_global.dtype))
    intrinsics_full = intrinsics_full.to(dtype=_geometry_dtype(intrinsics_full.dtype))
    intrinsics_global_valid = torch.isfinite(intrinsics_global).all(dim=(-2, -1))
    intrinsics_global_valid &= intrinsics_global[..., 0, 0].abs() > 1e-6
    intrinsics_global_valid &= intrinsics_global[..., 1, 1].abs() > 1e-6
    intrinsics_full_valid = torch.isfinite(intrinsics_full).all(dim=(-2, -1))
    intrinsics_full_valid &= intrinsics_full[..., 0, 0].abs() > 1e-6
    intrinsics_full_valid &= intrinsics_full[..., 1, 1].abs() > 1e-6
    confidence = depth_confidence_global
    if confidence is None:
        confidence = _first_tensor(outputs, "depth_confidence_global", "depth_conf_global")
    if confidence is None:
        confidence = torch.ones_like(depth)
    if tuple(confidence.shape) != tuple(depth.shape):
        raise ValueError("depth confidence must align with metric depth")
    depth_valid = depth_metric_valid_global
    if depth_valid is None:
        depth_valid = _first_tensor(outputs, "depth_metric_valid_global")
    if depth_valid is None:
        depth_values = depth.squeeze(-1) if depth.ndim == 4 and depth.shape[-1] == 1 else depth
        depth_valid = (torch.isfinite(depth_values) & (depth_values > 0)).flatten(1).any(dim=1)
    depth_valid = depth_valid.detach().cpu().to(dtype=torch.bool)
    if tuple(depth_valid.shape) != (global_frames,):
        raise ValueError("metric depth validity must have shape [T_G]")

    joint = _first_tensor(
        outputs,
        "hand_joint_xyz_metric_full",
        "dense_joint_xyz_interpolated_full",
        "dense_joint_xyz_refined_full",
    )
    vertex = _first_tensor(
        outputs,
        "hand_vertex_xyz_metric_full",
        "dense_vertex_xyz_interpolated_full",
        "dense_vertex_xyz_refined_full",
    )
    if joint is None or joint.ndim != 4 or tuple(joint.shape[:2]) != (full_frames, 2):
        raise ValueError("metric inference requires Hand joints [T_R, 2, J, 3]")
    if vertex is None or vertex.ndim != 4 or tuple(vertex.shape[:2]) != (full_frames, 2):
        raise ValueError("metric inference requires Hand vertices [T_R, 2, V, 3]")
    if joint.shape[-1] != 3 or vertex.shape[-1] != 3:
        raise ValueError("Hand XYZ tensors must end in three metric coordinates")
    hand_valid = _first_tensor(outputs, "hand_metric_valid_full", "hand_refined_valid_full")
    if hand_valid is None:
        hand_valid = torch.isfinite(joint).all(dim=(-2, -1)) & torch.isfinite(vertex).all(
            dim=(-2, -1)
        )
    hand_valid = hand_valid.detach().cpu().to(dtype=torch.bool)
    if tuple(hand_valid.shape) != (full_frames, 2):
        raise ValueError("Hand validity must have shape [T_R, 2]")

    scale_sources = (
        raw_to_meter_effective,
        raw_to_meter_fused,
        raw_to_meter_pred,
        metric_value_pred,
        pred_scene_scale_raw,
        metric_scale_valid,
    )
    scale_count = max(
        (
            int(value.numel())
            for value in scale_sources
            if isinstance(value, torch.Tensor)
        ),
        default=1,
    )
    scale_dtype = torch.float32
    metric_value = _scale_vector(
        metric_value_pred,
        count=scale_count,
        default=float("nan"),
        dtype=scale_dtype,
    )
    scene_scale = _scale_vector(
        pred_scene_scale_raw,
        count=scale_count,
        default=float("nan"),
        dtype=scale_dtype,
    )
    predicted_factor = _scale_vector(
        raw_to_meter_pred,
        count=scale_count,
        default=1.0,
        dtype=scale_dtype,
    )
    fused_factor = _scale_vector(raw_to_meter_fused, count=scale_count, default=1.0, dtype=scale_dtype) if raw_to_meter_fused is not None else predicted_factor.clone()
    residual_scale = _scale_vector(
        overlap_residual_scale,
        count=scale_count,
        default=1.0,
        dtype=scale_dtype,
    )
    effective_factor = (_scale_vector(raw_to_meter_effective, count=scale_count, default=1.0, dtype=scale_dtype)
                        if raw_to_meter_effective is not None else fused_factor * residual_scale)
    scale_valid = _scale_vector(
        metric_scale_valid,
        count=scale_count,
        default=0.0,
        dtype=torch.float32,
    ).to(dtype=torch.bool)
    if raw_to_meter_fused is not None and bool(scale_valid.any()):
        expected = fused_factor[scale_valid] * residual_scale[scale_valid]
        if not bool(torch.isfinite(expected).all()) or not bool((expected > 0).all()) or not torch.allclose(effective_factor[scale_valid], expected, atol=1e-6, rtol=1e-5):
            raise ValueError("effective scene factor must equal fused factor times overlap residual scale")

    hand: dict[str, Any] = {
        "root_xyz_metric": joint[..., 0, :],
        "joint_xyz_metric": joint,
        "vertex_xyz_metric": vertex,
        "valid": hand_valid,
        "coordinate_frame": "camera",
        "unit": "meter",
    }
    optional_hand_fields = {
        "joint_visibility_logits": "dense_joint_visibility_logits_refined_full",
        "vertex_visibility_logits": "dense_vertex_visibility_logits_refined_full",
        "joint_contact_logits": "dense_joint_contact_logits_refined_full",
        "vertex_contact_logits": "dense_vertex_contact_logits_refined_full",
        "joint_contact_log_distance": "dense_joint_contact_log_distance_refined_full",
        "vertex_contact_log_distance": "dense_vertex_contact_log_distance_refined_full",
    }
    for public_name, flat_name in optional_hand_fields.items():
        value = outputs.get(flat_name)
        if isinstance(value, torch.Tensor):
            hand[public_name] = value
            if public_name.endswith(("_visibility_logits", "_contact_logits")):
                hand[public_name.removesuffix("_logits") + "_probability"] = torch.sigmoid(
                    value
                )
            elif public_name.endswith("_contact_log_distance"):
                hand[public_name.removesuffix("_log_distance") + "_distance_metric"] = (
                    torch.expm1(value.float()).clamp_min(0.0) / 1000.0
                )

    return {
        "schema_version": INFERENCE_METRIC_SCHEMA_VERSION,
        "camera": {
            "pose_w2c_metric": camera,
            "valid": camera_valid,
            "frame_indices": torch.arange(full_frames, dtype=torch.long),
            "world_scope": world_scope,
            "translation_unit": "meter",
        },
        "depth": {
            "value_metric": depth,
            "confidence": confidence,
            "valid": depth_valid,
            "frame_indices": depth_indices,
            "rate": "G",
            "unit": "meter",
        },
        "intrinsics": {
            "K_global": intrinsics_global,
            "K_full": intrinsics_full,
            "K_global_valid": intrinsics_global_valid,
            "K_full_valid": intrinsics_full_valid,
            "global_frame_indices": depth_indices,
            "pixel_frame": "model_input_resized",
        },
        "metric_scale": {
            "scope": "per_clip",
            "metric_value_pred": metric_value,
            "pred_scene_scale_raw": scene_scale,
            "raw_to_meter_pred": predicted_factor,
            "raw_to_meter_fused": fused_factor,
            "overlap_residual_scale": residual_scale,
            "raw_to_meter_effective": effective_factor,
            "metric_value_effective": scene_scale * effective_factor,
            "valid": scale_valid,
        },
        "hand": hand,
    }


def attach_metric_prediction_contract(
    payload: dict[str, Any],
    metric_predictions: dict[str, Any],
) -> None:
    if metric_predictions.get("schema_version") != INFERENCE_METRIC_SCHEMA_VERSION:
        raise ValueError("unknown metric inference schema")
    payload["inference_schema_version"] = INFERENCE_METRIC_SCHEMA_VERSION
    payload["metric_predictions"] = metric_predictions
    payload["camera_pose"] = metric_predictions["camera"]["pose_w2c_metric"]
    payload["camera_pose_metric_full"] = metric_predictions["camera"][
        "pose_w2c_metric"
    ]
    payload["camera_pose_metric_valid_full"] = metric_predictions["camera"]["valid"]
    payload["depth"] = metric_predictions["depth"]["value_metric"]
    payload["depth_metric_global"] = metric_predictions["depth"]["value_metric"]
    payload["depth_metric_valid_global"] = metric_predictions["depth"]["valid"]
    payload["depth_confidence_global"] = metric_predictions["depth"]["confidence"]
    payload["depth_frame_indices"] = metric_predictions["depth"]["frame_indices"]
    payload["K"] = metric_predictions["intrinsics"]["K_global"]
    payload["K_full"] = metric_predictions["intrinsics"]["K_full"]
    payload["intrinsics_K_global"] = metric_predictions["intrinsics"]["K_global"]
    payload["intrinsics_K_full"] = metric_predictions["intrinsics"]["K_full"]
    payload["intrinsics_K_global_valid"] = metric_predictions["intrinsics"][
        "K_global_valid"
    ]
    payload["intrinsics_K_full_valid"] = metric_predictions["intrinsics"]["K_full_valid"]
    payload["intrinsics_global"] = metric_predictions["intrinsics"]["K_global"]
    payload["intrinsics_full"] = metric_predictions["intrinsics"]["K_full"]
    payload["intrinsics"] = metric_predictions["intrinsics"]["K_full"]
    payload["depth_global"] = metric_predictions["depth"]["value_metric"]
    payload["depth_conf_global"] = metric_predictions["depth"]["confidence"]
    payload["metric_scale"] = metric_predictions["metric_scale"]
    payload["hand"] = metric_predictions["hand"]
    payload["joint_xyz"] = metric_predictions["hand"]["joint_xyz_metric"]
    payload["vertex_xyz"] = metric_predictions["hand"]["vertex_xyz_metric"]
    payload["hand_joint_xyz_metric_full"] = metric_predictions["hand"][
        "joint_xyz_metric"
    ]
    payload["hand_vertex_xyz_metric_full"] = metric_predictions["hand"][
        "vertex_xyz_metric"
    ]
    payload["hand_metric_valid_full"] = metric_predictions["hand"]["valid"]
