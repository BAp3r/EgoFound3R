from __future__ import annotations


from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any
import numpy as np
import torch
from egohandmetric_prompt.data.marker_vertices import marker_vertex_ids_for_count


MANO778_SCHEMA = "egofound3r.hand_778.v1"


@lru_cache(maxsize=1)
def _sampling_tensors() -> tuple[torch.Tensor, ...]:
    path = Path(__file__).parent / "data/mano_upsampling/mano_195_to_778.npz"
    with np.load(path, allow_pickle=False) as asset:
        neighbors = torch.from_numpy(asset["neighbor_indices"].copy()).long()
        weights = torch.from_numpy(asset["geometry_weights"].copy())
        source_ids = torch.from_numpy(asset["source_vertex_ids"].copy()).long()
        faces = torch.from_numpy(asset["faces"].copy()).long()
    expected = torch.as_tensor(marker_vertex_ids_for_count(195), dtype=torch.long)
    if not torch.equal(source_ids, expected):
        raise ValueError("MANO upsampling source ordering differs from canonical 195 markers")
    edges = torch.cat((faces[:, :2], faces[:, 1:], faces[:, (2, 0)]))
    edges = torch.unique(edges.sort(dim=-1).values, dim=0)
    scalar_weights = weights.clamp_min(0)
    scalar_weights = scalar_weights / scalar_weights.sum(dim=-1, keepdim=True)
    return neighbors, weights, scalar_weights, source_ids, faces, edges


@torch.no_grad()
def build_mano778_hand(
    hand: Mapping[str, Any],
    *,
    source_vertex_ids: torch.Tensor | None = None,
) -> dict[str, Any]:
    xyz = hand.get("vertex_xyz_metric")
    if not isinstance(xyz, torch.Tensor) or xyz.shape[-2:] != (195, 3):
        raise ValueError("MANO upsampling requires metric XYZ [..., 195, 3]")
    if hand.get("unit") != "meter" or hand.get("coordinate_frame") != "camera":
        raise ValueError("MANO upsampling requires the camera-space meter Hand contract")
    valid = hand.get("valid")
    if not isinstance(valid, torch.Tensor) or valid.shape != xyz.shape[:-2]:
        raise ValueError("Hand validity must align with XYZ leading dimensions")
    neighbors, weights, scalar_weights, ids, faces, edges = _sampling_tensors()
    if source_vertex_ids is not None and not torch.equal(source_vertex_ids.detach().cpu(), ids):
        raise ValueError("input vertex IDs are not the canonical 195 MANO ordering")
    neighbors = neighbors.to(device=xyz.device)
    ids = ids.to(device=xyz.device)
    dtype = torch.float64 if xyz.dtype == torch.float64 else torch.float32
    source = xyz.detach().to(dtype=dtype)
    center = source.mean(dim=-2, keepdim=True)
    local = source - center
    vertices = center + (
        local[..., neighbors, :] * weights.to(device=xyz.device, dtype=dtype)[..., None]
    ).sum(dim=-2)
    vertices[..., ids, :] = source
    result: dict[str, Any] = {
        "schema_version": MANO778_SCHEMA,
        "vertex_xyz_metric": vertices,
        "valid": valid.detach().clone(),
        "coordinate_frame": "camera",
        "unit": "meter",
        "vertex_ids": torch.arange(778, dtype=torch.long),
        "source_vertex_ids": ids.detach().cpu().clone(),
        "marker_faces": faces.clone(),
        "marker_edges": edges.clone(),
        "interpolation": {
            "source_repository": (
                "https://github.com/microsoft/MeshGraphormer"
                "/blob/main/src/modeling/data/mano_downsampling.npz"
            ),
            "contact_distance_unit": "meter",
        },
    }

    def interpolate(value: torch.Tensor) -> torch.Tensor:
        if value.shape != xyz.shape[:-1]:
            raise ValueError("vertex attribute must align with [..., 195] XYZ")
        attr_dtype = torch.float64 if value.dtype == torch.float64 else torch.float32
        values = value.detach().to(device=xyz.device, dtype=attr_dtype)
        output = (values[..., neighbors] * scalar_weights.to(values)).sum(dim=-1)
        output[..., ids] = values
        return output

    def restore_anchors(output: torch.Tensor, value: torch.Tensor) -> None:
        if value.shape != xyz.shape[:-1]:
            raise ValueError("vertex attribute must align with [..., 195] XYZ")
        output[..., ids] = value.detach().to(output)

    for attribute in ("visibility", "contact"):
        prefix = f"vertex_{attribute}"
        probability = hand.get(f"{prefix}_probability")
        logits = hand.get(f"{prefix}_logits")
        if not isinstance(probability, torch.Tensor):
            if not isinstance(logits, torch.Tensor):
                continue
            probability = torch.sigmoid(logits)
        interpolated = interpolate(probability).clamp(0.0, 1.0)
        restore_anchors(interpolated, probability)
        result[f"{prefix}_probability"] = interpolated
        if isinstance(logits, torch.Tensor):
            logits_dtype = torch.promote_types(interpolated.dtype, logits.dtype)
            new_logits = torch.logit(interpolated.to(dtype=logits_dtype), eps=1e-7)
            restore_anchors(new_logits, logits)
            result[f"{prefix}_logits"] = new_logits

    distance = hand.get("vertex_contact_distance_metric")
    log_distance = hand.get("vertex_contact_log_distance")
    if not isinstance(distance, torch.Tensor) and isinstance(log_distance, torch.Tensor):
        distance = torch.expm1(log_distance.float()).clamp_min(0.0) / 1000.0
    if isinstance(distance, torch.Tensor):
        interpolated_distance = interpolate(distance).clamp_min(0.0)
        restore_anchors(interpolated_distance, distance)
        result["vertex_contact_distance_metric"] = interpolated_distance
        if isinstance(log_distance, torch.Tensor):
            log_dtype = torch.promote_types(interpolated_distance.dtype, log_distance.dtype)
            new_log_distance = torch.log1p(interpolated_distance.to(dtype=log_dtype) * 1000.0)
            restore_anchors(new_log_distance, log_distance)
            result["vertex_contact_log_distance"] = new_log_distance
    return result


def attach_mano778_output(payload: dict[str, Any]) -> dict[str, Any]:
    predictions = payload.get("metric_predictions")
    if not isinstance(predictions, dict) or not isinstance(predictions.get("hand"), dict):
        raise ValueError("MANO upsampling requires canonical metric_predictions.hand")
    hand_778 = build_mano778_hand(
        predictions["hand"], source_vertex_ids=payload.get("vertex_ids"),
    )
    predictions["hand_778"] = hand_778
    payload["hand_778"] = hand_778
    return hand_778
