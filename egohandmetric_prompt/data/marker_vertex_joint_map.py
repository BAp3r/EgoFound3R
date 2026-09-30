from __future__ import annotations


from pathlib import Path
from typing import Sequence
import torch
from egohandmetric_prompt.data.marker_vertices import marker_vertex_ids_for_count


OPENPOSE_HAND_BONES: tuple[tuple[int, int], ...] = (
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


DEFAULT_MARKER_VERTEX_JOINT_WEIGHT_COUNT = 195




def marker_vertex_joint_weight_path(marker_count: int) -> Path:
    if int(marker_count) != DEFAULT_MARKER_VERTEX_JOINT_WEIGHT_COUNT:
        raise ValueError(f"no vertex-joint mapping weights for marker_count={marker_count}")
    return Path(__file__).with_name("marker_vertex_joint_weights_195.pt")


def validate_vertex_joint_weights(
    weights: torch.Tensor,
    *,
    marker_count: int,
    joint_count: int = 21,
    bones: Sequence[tuple[int, int]] = OPENPOSE_HAND_BONES,
) -> None:
    if weights.ndim != 2 or tuple(weights.shape) != (int(marker_count), int(joint_count)):
        raise ValueError(
            f"vertex-joint mapping shape must be [{int(marker_count)}, {int(joint_count)}], "
            f"got {tuple(weights.shape)}"
        )
    if not torch.is_floating_point(weights):
        raise ValueError("vertex-joint mapping must be floating point")
    weights_cpu = weights.detach().to(dtype=torch.float32, device="cpu")
    if not torch.isfinite(weights_cpu).all().item():
        raise ValueError("vertex-joint mapping contains non-finite values")
    if (weights_cpu < -1e-6).any().item():
        raise ValueError("vertex-joint mapping contains negative weights")
    row_sums = weights_cpu.sum(dim=1)
    if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5):
        raise ValueError("vertex-joint mapping rows must sum to 1")
    nonzero = weights_cpu > 1e-6
    nonzero_counts = nonzero.sum(dim=1)
    if (nonzero_counts == 0).any().item() or (nonzero_counts > 2).any().item():
        raise ValueError("vertex-joint mapping rows must contain one or two non-zero joints")
    valid_edges = {tuple(sorted(edge)) for edge in bones}
    valid_joint_ids = {joint for edge in bones for joint in edge}
    for row_index, row_nonzero in enumerate(nonzero):
        joint_ids = torch.nonzero(row_nonzero, as_tuple=False).flatten().tolist()
        if len(joint_ids) == 1:
            if joint_ids[0] not in valid_joint_ids:
                raise ValueError(f"vertex-joint mapping row {row_index} uses invalid joint {joint_ids[0]}")
            continue
        if tuple(sorted(joint_ids)) not in valid_edges:
            raise ValueError(f"vertex-joint mapping row {row_index} does not match one skeleton bone: {joint_ids}")




def marker_vertex_joint_weights_for_count(marker_count: int) -> torch.Tensor:
    marker_count = int(marker_count)
    if marker_count != DEFAULT_MARKER_VERTEX_JOINT_WEIGHT_COUNT:
        raise ValueError(f"missing vertex-joint mapping for marker_count={marker_count}")
    path = marker_vertex_joint_weight_path(marker_count)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    marker_vertex_ids = payload["marker_vertex_ids"].tolist()
    if marker_vertex_ids != marker_vertex_ids_for_count(marker_count):
        raise ValueError("vertex-joint mapping marker_vertex_ids disagree with the marker vertex config")
    bones = [tuple(edge) for edge in payload["bones"].tolist()]
    if bones != list(OPENPOSE_HAND_BONES):
        raise ValueError("vertex-joint mapping bones disagree with the hand skeleton")
    weights = payload["weights"].to(dtype=torch.float32, device="cpu")
    validate_vertex_joint_weights(weights, marker_count=marker_count, joint_count=21)
    return weights
