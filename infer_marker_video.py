from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Iterator

import cv2
import numpy as np
import torch

from egohandmetric_prompt.inference_video_io import (
    WindowedVideoFrames,
    add_input_resize_arguments,
    resolve_full_frame_input_size,
    resolve_requested_input_size,
)

from egohandmetric_prompt import (
    build_runtime_marker_model,
    load_marker_model_weights,
    load_project_config,
    marker_model_floating_dtype,
)
from egohandmetric_prompt.configs import active_marker_model_config
from egohandmetric_prompt.data.marker_mesh import (
    marker_edges_for_count,
    marker_faces_for_count,
)
from egohandmetric_prompt.inference_contracts import (
    build_inference_provenance,
    prepare_inference_output_dir,
    reconstruct_multirate_metric_scene_outputs,
)
from egohandmetric_prompt.inference_visualization import (
    INFERENCE_COLOR_MODES,
    JOINT_DOT_RADIUS,
    JOINT_EDGE_WIDTH,
    JOINT_OUTLINE_RGB,
    MARKER_DOT_RADIUS,
    MARKER_EDGE_WIDTH,
    OCCLUDED_ALPHA,
    StrokeLayer,
    edge_endpoint_colors,
    face_alpha,
    hand_face_color,
    semantic_point_colors,
    signal_face_colors,
)
from egohandmetric_prompt.inference_root_z import add_root_z_arguments, root_z_options_from_args
from egohandmetric_prompt.inference_hand_anchors import add_hand_anchor_arguments, hand_anchor_options_from_args
from egohandmetric_prompt.inference_hand_depth_scale import add_hand_depth_scale_arguments, hand_depth_scale_options_from_args
from egohandmetric_prompt.inference_intrinsics import (
    add_root_k_arguments, load_root_intrinsics, validate_root_k_arguments,
)
from egohandmetric_prompt.models import MultiRateMarkerOutput
from egohandmetric_prompt.multirate_contracts import MultiRateFrameMap, build_multirate_frame_map
from egohandmetric_prompt.inference_forward import (
    _build_prediction_marker_forward_contract,
    _call_marker_model,
)


def _configure_deterministic_execution(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(int(seed))
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


MANO_JOINT_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
)


_MULTIRATE_HIGH_OUTPUT_KEYS = (
    "camera_pose_refined_high",
    "dense_vertex_xyz",
    "dense_vertex_visibility_logits",
    "dense_vertex_contact_logits",
    "dense_vertex_contact_log_conf",
    "dense_joint_xyz",
    "dense_joint_visibility_logits",
    "dense_joint_contact_logits",
    "dense_joint_contact_log_conf",
)
_MULTIRATE_HIGH_BOOLEAN_OUTPUT_KEYS = (
    "camera_refined_valid_high",
    "camera_anchor_exact_mask_high",
    "root_translation_valid",
)
_MULTIRATE_GLOBAL_OUTPUT_KEYS = (
    "camera_pose_global",
    "camera_pose_encoding_global",
    "intrinsics_global",
    "depth_global",
    "depth_conf_global",
)




































def _marker_model_config(project_config):
    return project_config.marker_model


def _build_video_temporal_batch(
    window_indices: list[int],
    *,
    fps: float,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if not window_indices:
        raise ValueError("video inference window cannot be empty")
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"video FPS must be finite and positive, got {fps!r}")
    frame_indices = torch.as_tensor(window_indices, device=device, dtype=torch.int64).view(1, -1)
    frame_count = int(frame_indices.shape[1])
    return {
        "frame_present_mask": torch.ones(1, frame_count, device=device, dtype=torch.bool),
        "high_temporal_pair_valid": torch.ones(1, max(frame_count - 1, 0), device=device, dtype=torch.bool),
        "temporal_pair_valid": torch.ones(1, max(frame_count - 1, 0), device=device, dtype=torch.bool),
    }


def _build_multirate_video_forward_contract(
    *,
    project_config: Any,
    images: torch.Tensor,
    window_indices: list[int],
    fps: float,
    global_stride: int = 5,
    global_anchor_phase: int | None = None,
) -> tuple[dict[str, Any], MultiRateFrameMap]:
    temporal_batch = _build_video_temporal_batch(window_indices, fps=fps, device=images.device)
    forward_contract = _build_prediction_marker_forward_contract(
        project_config=project_config,
        batch=temporal_batch,
        images=images,
        global_stride=global_stride,
        global_anchor_phase=(
            global_stride // 2
            if global_anchor_phase is None
            else global_anchor_phase
        ),
    )
    frame_map = forward_contract.get("multirate_frame_map")
    if not isinstance(frame_map, MultiRateFrameMap):
        frame_map = build_multirate_frame_map(
            high_frame_present=temporal_batch["frame_present_mask"],
            high_temporal_pair_valid=temporal_batch["temporal_pair_valid"],
            global_stride=global_stride,
            global_anchor_phase=(
                global_stride // 2
                if global_anchor_phase is None
                else global_anchor_phase
            ),
        )
        forward_contract.pop("frame_present_mask", None)
        forward_contract.pop("temporal_pair_valid", None)
        forward_contract["multirate_frame_map"] = frame_map
    frame_map.validate()
    return forward_contract, frame_map


def _reconstruct_multirate_metric_scale_outputs(
    outputs: dict[str, Any],
    *,
    global_frame_present: torch.Tensor | None = None,
) -> dict[str, Any]:
    return reconstruct_multirate_metric_scene_outputs(
        outputs,
        global_frame_present=global_frame_present,
    )




def _project_points(points_xyz: torch.Tensor, intrinsics: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    z = points_xyz[..., 2]
    valid = torch.isfinite(points_xyz).all(dim=-1)
    intrinsics_valid = torch.isfinite(intrinsics).all(dim=(-2, -1))
    while intrinsics_valid.ndim < valid.ndim:
        intrinsics_valid = intrinsics_valid.unsqueeze(-1)
    valid &= intrinsics_valid
    valid &= z > 1e-6
    safe_z = z.clamp_min(1e-6)
    fx = intrinsics[..., 0, 0]
    fy = intrinsics[..., 1, 1]
    cx = intrinsics[..., 0, 2]
    cy = intrinsics[..., 1, 2]
    while fx.ndim < safe_z.ndim:
        fx = fx.unsqueeze(-1)
        fy = fy.unsqueeze(-1)
        cx = cx.unsqueeze(-1)
        cy = cy.unsqueeze(-1)
    u = fx * (points_xyz[..., 0] / safe_z) + cx
    v = fy * (points_xyz[..., 1] / safe_z) + cy
    uv = torch.stack([u, v], dim=-1)
    return uv, valid


def _rgb_to_bgr_tuple(color_rgb: np.ndarray) -> tuple[int, int, int]:
    return tuple(int(value) for value in color_rgb[::-1])


def _paint_faces(
    canvas: np.ndarray,
    faces_uv: np.ndarray,
    face_colors: np.ndarray,
    face_alphas: np.ndarray,
    face_depths: np.ndarray,
) -> None:
    if len(faces_uv) == 0:
        return
    polygons = np.rint(faces_uv).astype(np.int32)
    color_layer = canvas.copy()
    alpha_layer = np.zeros(canvas.shape[:2], dtype=np.float32)
    for index in np.argsort(-np.asarray(face_depths, dtype=np.float32)):
        polygon = polygons[index]
        cv2.fillPoly(
            color_layer,
            [polygon],
            _rgb_to_bgr_tuple(np.asarray(face_colors[index], dtype=np.uint8)),
        )
        cv2.fillPoly(
            alpha_layer,
            [polygon],
            float(face_alphas[index]),
            lineType=cv2.LINE_AA,
        )
    weight = alpha_layer[:, :, None]
    blended = canvas.astype(np.float32) * (1.0 - weight) + color_layer.astype(np.float32) * weight
    np.copyto(canvas, np.rint(blended).astype(np.uint8))


def _paint_wireframe(
    canvas: np.ndarray,
    *,
    uv: np.ndarray,
    valid: np.ndarray,
    visibility: np.ndarray,
    depth: np.ndarray,
    edges: np.ndarray,
    colors: np.ndarray,
    dot_radius: int,
    edge_width: int,
    occluded_alpha: float,
) -> None:
    layer = StrokeLayer(canvas.shape[0], canvas.shape[1])
    stroke = np.asarray(colors, dtype=np.uint8)[..., ::-1]
    visible = np.asarray(visibility, dtype=np.float32) >= 0.5
    starts, ends = edge_endpoint_colors(np.asarray(colors, dtype=np.uint8), edges)
    starts = starts[..., ::-1]
    ends = ends[..., ::-1]
    if len(edges):
        edge_depth = 0.5 * (depth[edges[:, 0]] + depth[edges[:, 1]])
        for index in np.argsort(-edge_depth):
            head, tail = int(edges[index, 0]), int(edges[index, 1])
            if not (valid[head] and valid[tail]):
                continue
            alpha = 1.0 if (visible[head] and visible[tail]) else occluded_alpha
            start, end = uv[head], uv[tail]
            if np.array_equal(starts[index], ends[index]):
                layer.segment(start, end, starts[index], width=edge_width, alpha=alpha)
                continue
            middle = 0.5 * (start + end)
            layer.segment(start, middle, starts[index], width=edge_width, alpha=alpha)
            layer.segment(middle, end, ends[index], width=edge_width, alpha=alpha)
    for index in np.argsort(-np.asarray(depth, dtype=np.float32)):
        if not valid[index]:
            continue
        alpha = 1.0 if visible[index] else occluded_alpha
        layer.dot(uv[index], stroke[index], radius=dot_radius, alpha=alpha)
    layer.composite(canvas)


def _paint_joints(
    canvas: np.ndarray,
    *,
    uv: np.ndarray,
    valid: np.ndarray,
    visibility: np.ndarray,
    colors: np.ndarray,
    occluded_alpha: float,
) -> None:
    layer = StrokeLayer(canvas.shape[0], canvas.shape[1])
    stroke = np.asarray(colors, dtype=np.uint8)[..., ::-1]
    outline = np.asarray(JOINT_OUTLINE_RGB, dtype=np.uint8)[::-1]
    visible = np.asarray(visibility, dtype=np.float32) >= 0.5
    for head, tail in np.asarray(MANO_JOINT_EDGES, dtype=np.int64):
        head, tail = int(head), int(tail)
        if not (valid[head] and valid[tail]):
            continue
        alpha = 1.0 if (visible[head] and visible[tail]) else occluded_alpha
        layer.segment(uv[head], uv[tail], outline, width=JOINT_EDGE_WIDTH + 2, alpha=alpha)
        layer.segment(uv[head], uv[tail], stroke[head], width=JOINT_EDGE_WIDTH, alpha=alpha)
    for index in np.flatnonzero(valid):
        alpha = 1.0 if visible[index] else occluded_alpha
        layer.dot(uv[index], outline, radius=JOINT_DOT_RADIUS + 1, alpha=alpha)
        layer.dot(uv[index], stroke[index], radius=JOINT_DOT_RADIUS, alpha=alpha)
    layer.composite(canvas)


def _draw_overlay_mode_label(image: np.ndarray, mode: str) -> None:
    label = "Contact" if mode == "Contact Class" else mode
    if mode == "Contact Distance":
        label = "Distance 0-50 mm"
    cv2.putText(
        image,
        label,
        (6, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (35, 35, 35),
        thickness=3,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        image,
        label,
        (6, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (245, 245, 245),
        thickness=1,
        lineType=cv2.LINE_AA,
    )


def _render_overlay_frames(
    rgb_frames_uint8: Iterable[np.ndarray],
    marker_uv: torch.Tensor,
    marker_uv_valid: torch.Tensor,
    marker_visibility_logits: torch.Tensor,
    joint_uv: torch.Tensor,
    joint_uv_valid: torch.Tensor,
    joint_visibility_logits: torch.Tensor,
    marker_edges: torch.Tensor,
    marker_faces: torch.Tensor,
    marker_depth: torch.Tensor | None = None,
    *,
    marker_visibility_probability: torch.Tensor | None = None,
    stream: bool = False,
) -> list[np.ndarray] | Iterator[np.ndarray]:
    marker_uv_np = marker_uv.detach().cpu().float().numpy()
    marker_valid_np = marker_uv_valid.detach().cpu().bool().numpy()
    marker_visibility_np = (
        torch.sigmoid(marker_visibility_logits)
        if marker_visibility_probability is None else marker_visibility_probability
    ).detach().cpu().float().numpy()
    joint_uv_np = joint_uv.detach().cpu().float().numpy()
    joint_valid_np = joint_uv_valid.detach().cpu().bool().numpy()
    joint_visibility_np = torch.sigmoid(joint_visibility_logits).detach().cpu().float().numpy()
    marker_edges_np = marker_edges.detach().cpu().long().numpy()
    marker_faces_np = marker_faces.detach().cpu().long().numpy()
    marker_depth_np = (
        None if marker_depth is None else marker_depth.detach().cpu().float().numpy()
    )

    def render():
        count = 0
        for frame_index, frame_rgb in enumerate(rgb_frames_uint8):
            if frame_index >= marker_uv_np.shape[0]:
                raise ValueError("overlay tensors must align with RGB frames")
            canvas = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            for side_index in range(marker_uv_np.shape[1]):
                uv = marker_uv_np[frame_index, side_index]
                valid = marker_valid_np[frame_index, side_index]
                visibility = marker_visibility_np[frame_index, side_index]
                if marker_depth_np is None:
                    depth = np.zeros(uv.shape[0], dtype=np.float32)
                else:
                    depth = marker_depth_np[frame_index, side_index]
                marker_colors = semantic_point_colors(
                    "Geometry",
                    side_index=side_index,
                    point_kind="marker",
                    visibility=visibility,
                    contact_probability=None,
                    contact_distance_m=None,
                )
                if marker_faces_np.size:
                    faces = marker_faces_np[valid[marker_faces_np].all(-1)]
                    _paint_faces(
                        canvas,
                        uv[faces],
                        np.repeat(hand_face_color(side_index)[None], len(faces), axis=0),
                        face_alpha(visibility, faces),
                        depth[faces].mean(axis=1),
                    )
                _paint_wireframe(
                    canvas,
                    uv=uv,
                    valid=valid,
                    visibility=visibility,
                    depth=depth,
                    edges=marker_edges_np,
                    colors=marker_colors,
                    dot_radius=MARKER_DOT_RADIUS,
                    edge_width=MARKER_EDGE_WIDTH,
                    occluded_alpha=OCCLUDED_ALPHA,
                )
                joint_colors = semantic_point_colors(
                    "Geometry",
                    side_index=side_index,
                    point_kind="joint",
                    visibility=joint_visibility_np[frame_index, side_index],
                    contact_probability=None,
                    contact_distance_m=None,
                )
                _paint_joints(
                    canvas,
                    uv=joint_uv_np[frame_index, side_index],
                    valid=joint_valid_np[frame_index, side_index],
                    visibility=joint_visibility_np[frame_index, side_index],
                    colors=joint_colors,
                    occluded_alpha=OCCLUDED_ALPHA,
                )
            count += 1
            yield canvas
        if count != marker_uv_np.shape[0]:
            raise ValueError("overlay tensors must align with RGB frames")

    return render() if stream else list(render())


def _render_semantic_overlay_frame(
    frame_rgb: np.ndarray,
    *,
    mode: str,
    marker_uv: np.ndarray,
    marker_valid: np.ndarray,
    marker_visibility: np.ndarray,
    marker_contact: np.ndarray,
    marker_distance_m: np.ndarray,
    joint_uv: np.ndarray,
    joint_valid: np.ndarray,
    joint_visibility: np.ndarray,
    joint_contact: np.ndarray,
    joint_distance_m: np.ndarray,
    marker_edges: np.ndarray,
    marker_faces: np.ndarray,
    marker_depth: np.ndarray,
    title: str | None = None,
) -> np.ndarray:
    canvas = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    for side_index in range(marker_uv.shape[0]):
        visibility = marker_visibility[side_index]
        marker_colors = semantic_point_colors(
            mode,
            side_index=side_index,
            point_kind="marker",
            visibility=visibility,
            contact_probability=marker_contact[side_index],
            contact_distance_m=marker_distance_m[side_index],
        )
        if marker_faces.size:
            faces = marker_faces[marker_valid[side_index, marker_faces].all(-1)]
            wash = (
                np.repeat(hand_face_color(side_index)[None], len(faces), axis=0)
                if mode == "Geometry"
                else signal_face_colors(marker_colors, faces)
            )
            _paint_faces(
                canvas,
                marker_uv[side_index, faces],
                wash,
                face_alpha(visibility, faces),
                marker_depth[side_index][faces].mean(axis=1),
            )
        _paint_wireframe(
            canvas,
            uv=marker_uv[side_index],
            valid=marker_valid[side_index],
            visibility=visibility,
            depth=marker_depth[side_index],
            edges=marker_edges,
            colors=marker_colors,
            dot_radius=MARKER_DOT_RADIUS,
            edge_width=MARKER_EDGE_WIDTH,
            occluded_alpha=OCCLUDED_ALPHA,
        )
        joint_colors = semantic_point_colors(
            mode,
            side_index=side_index,
            point_kind="joint",
            visibility=joint_visibility[side_index],
            contact_probability=joint_contact[side_index],
            contact_distance_m=joint_distance_m[side_index],
        )
        _paint_joints(
            canvas,
            uv=joint_uv[side_index],
            valid=joint_valid[side_index],
            visibility=joint_visibility[side_index],
            colors=joint_colors,
            occluded_alpha=OCCLUDED_ALPHA,
        )
    _draw_overlay_mode_label(canvas, mode if title is None else title)
    return canvas
def _render_multimode_overlay_frames(
    rgb_frames_uint8: Iterable[np.ndarray],
    marker_uv: torch.Tensor,
    marker_uv_valid: torch.Tensor,
    marker_visibility_logits: torch.Tensor,
    marker_contact_logits: torch.Tensor,
    marker_contact_distance_m: torch.Tensor,
    joint_uv: torch.Tensor,
    joint_uv_valid: torch.Tensor,
    joint_visibility_logits: torch.Tensor,
    joint_contact_logits: torch.Tensor,
    joint_contact_distance_m: torch.Tensor,
    marker_edges: torch.Tensor,
    *,
    marker_faces: torch.Tensor,
    marker_depth: torch.Tensor,
    marker_visibility_probability: torch.Tensor | None = None,
    marker_contact_probability: torch.Tensor | None = None,
    stream: bool = False,
) -> list[np.ndarray] | Iterator[np.ndarray]:
    marker_uv_np = marker_uv.detach().cpu().float().numpy()
    marker_valid_np = marker_uv_valid.detach().cpu().bool().numpy()
    marker_visibility_np = (
        torch.sigmoid(marker_visibility_logits)
        if marker_visibility_probability is None else marker_visibility_probability
    ).detach().cpu().float().numpy()
    marker_contact_np = (
        torch.sigmoid(marker_contact_logits)
        if marker_contact_probability is None else marker_contact_probability
    ).detach().cpu().float().numpy()
    marker_distance_np = marker_contact_distance_m.detach().cpu().float().numpy()
    joint_uv_np = joint_uv.detach().cpu().float().numpy()
    joint_valid_np = joint_uv_valid.detach().cpu().bool().numpy()
    joint_visibility_np = torch.sigmoid(joint_visibility_logits).detach().cpu().float().numpy()
    joint_contact_np = torch.sigmoid(joint_contact_logits).detach().cpu().float().numpy()
    joint_distance_np = joint_contact_distance_m.detach().cpu().float().numpy()
    marker_edges_np = marker_edges.detach().cpu().long().numpy()
    marker_faces_np = marker_faces.detach().cpu().long().numpy()
    marker_depth_np = marker_depth.detach().cpu().float().numpy()
    if joint_uv_np.shape[0] != marker_uv_np.shape[0]:
        raise ValueError("multi-mode overlay tensors must align with RGB frames")

    def render():
        count = 0
        for frame_index, frame_rgb in enumerate(rgb_frames_uint8):
            if frame_index >= marker_uv_np.shape[0]:
                raise ValueError("overlay tensors must align with RGB frames")
            panels = [
                _render_semantic_overlay_frame(
                    frame_rgb,
                    mode=mode,
                    marker_uv=marker_uv_np[frame_index],
                    marker_valid=marker_valid_np[frame_index],
                    marker_visibility=marker_visibility_np[frame_index],
                    marker_contact=marker_contact_np[frame_index],
                    marker_distance_m=marker_distance_np[frame_index],
                    joint_uv=joint_uv_np[frame_index],
                    joint_valid=joint_valid_np[frame_index],
                    joint_visibility=joint_visibility_np[frame_index],
                    joint_contact=joint_contact_np[frame_index],
                    joint_distance_m=joint_distance_np[frame_index],
                    marker_edges=marker_edges_np,
                    marker_faces=marker_faces_np,
                    marker_depth=marker_depth_np[frame_index],
                )
                for mode in INFERENCE_COLOR_MODES
            ]
            top = np.concatenate(panels[:2], axis=1)
            bottom = np.concatenate(panels[2:], axis=1)
            count += 1
            yield np.concatenate((top, bottom), axis=0)
        if count != marker_uv_np.shape[0]:
            raise ValueError("overlay tensors must align with RGB frames")

    return render() if stream else list(render())






def _marker_faces_from_config(project_config, *, vertex_count: int) -> torch.Tensor:
    model_config = _marker_model_config(project_config)
    marker_faces = marker_faces_for_count(model_config.num_markers)
    if marker_faces.numel() > 0 and int(marker_faces.max().item()) >= vertex_count:
        raise ValueError(
            f"marker face count mismatch: max_face_index={int(marker_faces.max().item())} output_vertices={vertex_count}"
        )
    return marker_faces


def _marker_edges_from_config(project_config, *, vertex_count: int) -> torch.Tensor:
    marker_faces = _marker_faces_from_config(project_config, vertex_count=vertex_count)
    if marker_faces.numel() == 0:
        return torch.zeros((0, 2), dtype=torch.long)
    model_config = _marker_model_config(project_config)
    return marker_edges_for_count(model_config.num_markers)


def _as_public_multirate_outputs(
    raw_outputs: Any,
    *,
    global_frame_present: torch.Tensor | None = None,
) -> dict[str, Any]:
    if not isinstance(raw_outputs, MultiRateMarkerOutput):
        raise TypeError("multi-rate video inference requires MultiRateMarkerOutput from MarkerModel")
    outputs = raw_outputs.to_dict()
    required = (
        set(_MULTIRATE_HIGH_OUTPUT_KEYS)
        | set(_MULTIRATE_HIGH_BOOLEAN_OUTPUT_KEYS)
        | set(_MULTIRATE_GLOBAL_OUTPUT_KEYS)
    )
    missing = sorted(required.difference(outputs))
    if missing:
        raise RuntimeError(f"multi-rate model output is missing public fields: {missing}")
    for forbidden in ("camera_pose", "depth", "intrinsics"):
        if forbidden in outputs:
            raise RuntimeError(f"multi-rate public output must not expose ambiguous {forbidden!r}")
    return _reconstruct_multirate_metric_scale_outputs(
        outputs,
        global_frame_present=global_frame_present,
    )




def _parse_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    return device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--video-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--input-fps", type=float, default=None)
    add_input_resize_arguments(parser)
    parser.add_argument("--high-fps", type=float, default=10.0)
    parser.add_argument("--global-stride", type=int, choices=range(1, 6), default=5)
    parser.add_argument("--chunk-seconds", type=float, default=2.0)
    parser.add_argument("--overlap-seconds", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="auto",
                        help="Compute device: auto, cpu, cuda or cuda:N.")
    parser.add_argument("--save-overlay", action=argparse.BooleanOptionalAction, default=True,
                        help="Write the rendered overlay video and per-frame PNGs.")
    parser.add_argument("--overlay-frame-images", action=argparse.BooleanOptionalAction, default=True,
                        help="Also save per-frame overlay PNGs; disable to reduce export I/O.")
    parser.add_argument("--upsample-mano", action="store_true", help="Add independent 778-vertex meter outputs; preserve all 195 predictions.")
    parser.add_argument("--overlay-marker-count", type=int, choices=(195, 778), default=195)
    parser.add_argument(
        "--smooth-hand-root-h",
        action="store_true",
        help="confidence-aware three-frame smoothing of the H-rate Root on the multi-rate path",
    )
    parser.add_argument(
        "--save-debug-tensors",
        action="store_true",
        help="save the full internal debug tensors on the multi-rate path",
    )
    parser.add_argument(
        "--allow-output-overwrite",
        action="store_true",
        help="explicitly clear and overwrite the previous results the inference scripts manage in output-dir",
    )
    add_root_z_arguments(parser)
    add_hand_anchor_arguments(parser)
    add_hand_depth_scale_arguments(parser)
    add_root_k_arguments(parser)
    args = parser.parse_args()
    cli_started_at = time.perf_counter()
    validate_root_k_arguments(args)
    root_z_options = root_z_options_from_args(args)
    hand_anchor_options = hand_anchor_options_from_args(args)
    hand_depth_scale_options = hand_depth_scale_options_from_args(args)

    project_config = load_project_config(args.config)
    print(f"project={project_config.project.default_name} config={args.config or 'default'}")
    _configure_deterministic_execution(project_config.project.seed)
    requested_size_hw = resolve_requested_input_size(
        default_size_hw=(project_config.marker_runtime.image_height, project_config.marker_runtime.image_width),
        input_height=args.input_height,
        input_width=args.input_width,
    )
    target_height, target_width = requested_size_hw
    frames = WindowedVideoFrames(
        args.video_path,
        max_frames=args.max_frames,
        target_height=target_height,
        target_width=target_width,
        resize_mode=args.input_resize_mode,
    )
    decoded_fps = frames.fps
    fps = float(decoded_fps if args.input_fps is None else args.input_fps)
    prepare_inference_output_dir(
        args.output_dir,
        allow_overwrite=bool(args.allow_output_overwrite),
    )

    device = _parse_device(args.device)
    if args.checkpoint is None:
        raise ValueError("video inference requires an explicit --checkpoint")
    actual_input_size_hw = frames.image_size_hw
    root_intrinsics_input = load_root_intrinsics(
        source=args.root_k_source, file_path=args.root_k_file,
        video_path=args.video_path, source_frame_indices=list(range(len(frames))),
        image_size_hw=actual_input_size_hw,
    )
    model = build_runtime_marker_model(project_config).to(device)
    if args.checkpoint is not None:
        load_marker_model_weights(
            model,
            args.checkpoint,
            model_config=active_marker_model_config(project_config),
            align_model_floating_dtype=True,
        )
    model.eval()
    from egohandmetric_prompt.temporal_rate_contract import TemporalRateContract
    from infer_marker_multiclip import run_long_video_inference

    provenance = None
    if args.config is not None and args.checkpoint is not None:
        provenance = build_inference_provenance(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            video_path=args.video_path,
            decoded_video_fps=decoded_fps,
            effective_r_fps=fps,
            model_input_size_hw=frames.image_size_hw,
            source_video_size_hw=frames.source_size_hw,
            temporal_architecture_id=_marker_model_config(
                project_config
            ).temporal_architecture_id,
            command_argv=[sys.executable, *sys.argv],
            repo_root=Path(__file__).resolve().parent,
        )
        provenance["model_compute_dtype"] = str(
            marker_model_floating_dtype(model)
        )
        _, input_affine = resolve_full_frame_input_size(
            frames.source_size_hw,
            requested_size_hw,
            mode=args.input_resize_mode,
        )
        provenance.update({
            "input_resize_mode": args.input_resize_mode,
            "requested_input_size_hw": list(requested_size_hw),
            "input_affine_source_to_model": input_affine.tolist(),
        })
    smoothing_kwargs = (
        {"hand_h_root_smoothing": True}
        if args.smooth_hand_root_h
        else {}
    )
    with frames:
        summary = run_long_video_inference(
            hand_anchor_options=hand_anchor_options,
            hand_depth_scale_options=hand_depth_scale_options,
            project_config=project_config,
            model=model,
            frames=frames,
            video_path=args.video_path,
            output_dir=args.output_dir,
            device=device,
            rate_contract=TemporalRateContract(
                r_fps=fps,
                h_fps=float(args.high_fps),
                global_stride=int(args.global_stride),
            ),
            chunk_seconds=float(args.chunk_seconds),
            overlap_seconds=float(args.overlap_seconds),
            save_overlay=bool(args.save_overlay),
            save_overlay_frame_images=bool(args.overlay_frame_images),
            allow_output_overwrite=bool(args.allow_output_overwrite),
            provenance=provenance,
            include_debug_tensors=bool(args.save_debug_tensors),
            upsample_mano=bool(args.upsample_mano),
            overlay_marker_count=args.overlay_marker_count,
            root_z_options=root_z_options,
            root_intrinsics_input=root_intrinsics_input,
            cli_started_at=cli_started_at,
            **smoothing_kwargs,
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
