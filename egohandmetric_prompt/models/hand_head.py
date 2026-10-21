from __future__ import annotations


import math
from collections.abc import Sequence
import torch
from torch import nn
from torch.nn import functional as F
from egohandmetric_prompt.data.marker_mesh import marker_face_incidence_degree
from egohandmetric_prompt.data.marker_vertex_joint_map import validate_vertex_joint_weights
from egohandmetric_prompt.contact_confidence import contact_confidence_from_log
from .transformer_blocks import TransformerDecoderBlock
from .hand_prompt_adapter import KEYPOINT_GEOMETRY_DIM
from .hand_translation_solver import (
    root_subset_reliability,
    solve_partial_observation_translation,
)


def smooth_root_solver_wrist_uv(
    keypoint_uv: torch.Tensor,
    keypoint_confidence: torch.Tensor,
    keypoint_mask: torch.Tensor,
    *,
    confidence_threshold: float = 0.5,
    isolated_residual: float = 0.06,
    neighbor_span: float = 0.12,
    one_sided_jump: float = 0.12,
) -> tuple[torch.Tensor, torch.Tensor]:
    if keypoint_uv.ndim != 5 or keypoint_uv.shape[-1] != 2 or keypoint_uv.shape[-2] < 1:
        raise ValueError("keypoint_uv must have shape [B, T, S, J, 2]")
    expected = keypoint_uv.shape[:-1]
    if keypoint_confidence.shape != expected or keypoint_mask.shape != expected:
        raise ValueError("keypoint confidence and mask must align with keypoint_uv")
    if keypoint_mask.dtype is not torch.bool:
        raise ValueError("keypoint_mask must be bool")
    for name, value in (
        ("confidence_threshold", confidence_threshold),
        ("isolated_residual", isolated_residual),
        ("neighbor_span", neighbor_span),
        ("one_sided_jump", one_sided_jump),
    ):
        if not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"{name} must be finite and nonnegative")

    smoothed = keypoint_uv.detach().clone()
    applied = torch.zeros(
        keypoint_uv.shape[:3],
        device=keypoint_uv.device,
        dtype=torch.bool,
    )
    if keypoint_uv.shape[1] < 3:
        return smoothed, applied
    finite = keypoint_mask & torch.isfinite(keypoint_uv).all(dim=-1)
    confidence = torch.nan_to_num(
        keypoint_confidence.detach(), nan=0.0, posinf=0.0, neginf=0.0
    )
    for batch_index in range(keypoint_uv.shape[0]):
        for side_index in range(keypoint_uv.shape[2]):
            for frame_index in range(1, keypoint_uv.shape[1] - 1):
                current_valid = bool(finite[batch_index, frame_index, side_index, 0].item())
                previous_valid = bool(finite[batch_index, frame_index - 1, side_index, 0].item())
                following_valid = bool(finite[batch_index, frame_index + 1, side_index, 0].item())
                if not (current_valid and previous_valid and following_valid):
                    continue
                if float(confidence[batch_index, frame_index, side_index, 0].item()) >= confidence_threshold:
                    continue
                previous = keypoint_uv[batch_index, frame_index - 1, side_index, 0].detach()
                current = keypoint_uv[batch_index, frame_index, side_index, 0].detach()
                following = keypoint_uv[batch_index, frame_index + 1, side_index, 0].detach()
                prediction = 0.5 * (previous + following)
                residual = torch.linalg.vector_norm(current - prediction)
                span = torch.linalg.vector_norm(previous - following)
                jump_previous = torch.linalg.vector_norm(current - previous)
                jump_following = torch.linalg.vector_norm(current - following)
                isolated = (residual > isolated_residual) and (span < neighbor_span)
                one_sided = max(float(jump_previous), float(jump_following)) > one_sided_jump
                if isolated or (one_sided and span < neighbor_span):
                    smoothed[batch_index, frame_index, side_index, 0] = prediction
                    applied[batch_index, frame_index, side_index] = True
    return smoothed, applied


class GatedFaceIncidenceResidual(nn.Module):
    def __init__(
        self,
        marker_faces: torch.Tensor,
        *,
        marker_count: int,
        gate_init: float = 0.0,
        residual_strength: float = 0.5,
    ) -> None:
        super().__init__()
        if marker_faces.numel() == 0:
            raise ValueError("face-incidence residual requires non-empty marker_faces")
        if not math.isfinite(float(gate_init)):
            raise ValueError("face-incidence residual gate_init must be finite")
        if not math.isfinite(float(residual_strength)) or not 0.0 <= float(residual_strength) <= 1.0:
            raise ValueError("face-incidence residual strength must be finite and in [0, 1]")
        marker_faces = marker_faces.detach().to(dtype=torch.long)
        degree = marker_face_incidence_degree(marker_faces, marker_count=marker_count)
        self.marker_count = int(marker_count)
        self.register_buffer("marker_faces", marker_faces, persistent=False)
        self.register_buffer("incident_face_degree", degree, persistent=False)
        self.gamma = nn.Parameter(torch.tensor(float(gate_init), dtype=torch.float32))
        self.residual_strength = float(residual_strength)

    def forward(self, vertex_features: torch.Tensor) -> torch.Tensor:
        if vertex_features.ndim != 3 or vertex_features.shape[1] != self.marker_count:
            raise ValueError(
                f"vertex_features must have shape [N, {self.marker_count}, C]"
            )
        face_features = vertex_features[:, self.marker_faces, :].mean(dim=2)
        incident_features = face_features.repeat_interleave(3, dim=1)
        neighbor_features = torch.zeros_like(vertex_features)
        neighbor_features.index_add_(1, self.marker_faces.reshape(-1), incident_features)
        degree = self.incident_face_degree.to(dtype=vertex_features.dtype).view(1, -1, 1)
        neighbor_features = neighbor_features / degree
        gate = self.residual_strength * torch.tanh(self.gamma).to(dtype=vertex_features.dtype)
        return vertex_features + gate * (neighbor_features - vertex_features)


class HandHead(nn.Module):
    def __init__(
        self,
        prompt_dim: int,
        num_markers: int,
        num_joints: int = 21,
        num_queries: int | None = None,
        hand_feature_dim: int | None = None,
        root_num_layers: int = 1,
        joint_num_layers: int = 5,
        vertex_num_layers: int = 5,
        num_heads: int = 12,
        mlp_ratio: int = 4,
        vertex_joint_weights: torch.Tensor | None = None,
        marker_faces: torch.Tensor | None = None,
        enable_vertex_face_residual: bool = False,
        vertex_face_residual_layers: Sequence[int] = (),
        vertex_face_residual_gate_init: float = 0.0,
        vertex_face_residual_strength: float = 0.5,
    ) -> None:
        super().__init__()
        if int(num_joints) != 21:
            raise ValueError("HandHead currently requires 21 joints including root.")
        if num_queries is not None and int(num_queries) != int(num_markers):
            raise ValueError("HandHead requires num_queries == num_markers.")
        if vertex_joint_weights is None:
            raise ValueError("HandHead requires fixed vertex_joint_weights.")
        self.prompt_dim = int(prompt_dim)
        self.hand_feature_dim = self.prompt_dim if hand_feature_dim is None else int(hand_feature_dim)
        self.num_vertices = int(num_markers)
        self.num_joints = int(num_joints)
        validate_vertex_joint_weights(
            vertex_joint_weights,
            marker_count=self.num_vertices,
            joint_count=self.num_joints,
        )
        self.register_buffer("vertex_joint_weights", vertex_joint_weights.detach().to(dtype=torch.float32))

        self.root_query = nn.Parameter(torch.randn(1, 1, self.prompt_dim) * 0.02)
        self.joint_queries = nn.Parameter(torch.randn(1, self.num_joints - 1, self.prompt_dim) * 0.02)
        self.vertex_queries = nn.Parameter(torch.randn(1, self.num_vertices, self.prompt_dim) * 0.02)

        self.root_query_proj = nn.Linear(self.prompt_dim, self.prompt_dim)
        self.joint_query_proj = nn.Linear(self.prompt_dim, self.prompt_dim)
        self.vertex_query_proj = nn.Linear(self.prompt_dim, self.prompt_dim)
        self.root_condition_proj = nn.Linear(self.prompt_dim, self.prompt_dim)
        self.joint_keypoint_proj = nn.Linear(self.prompt_dim, self.prompt_dim)
        self.vertex_keypoint_proj = nn.Linear(self.prompt_dim, self.prompt_dim)
        self.joint_2d_proj = nn.Linear(KEYPOINT_GEOMETRY_DIM, self.prompt_dim)
        self.hand_feature_proj = (
            nn.Identity()
            if self.hand_feature_dim == self.prompt_dim
            else nn.Sequential(
                nn.LayerNorm(self.hand_feature_dim),
                nn.Linear(self.hand_feature_dim, self.prompt_dim),
            )
        )
        self.metric_feature_proj = (
            nn.Identity()
            if self.hand_feature_dim == self.prompt_dim
            else nn.Sequential(
                nn.LayerNorm(self.hand_feature_dim),
                nn.Linear(self.hand_feature_dim, self.prompt_dim),
            )
        )

        self.view_feature_proj = (
            nn.Identity()
            if self.hand_feature_dim == self.prompt_dim
            else nn.Sequential(
                nn.LayerNorm(self.hand_feature_dim),
                nn.Linear(self.hand_feature_dim, self.prompt_dim),
            )
        )

        self.root_blocks = nn.ModuleList(
            [TransformerDecoderBlock(self.prompt_dim, num_heads, mlp_ratio) for _ in range(root_num_layers)]
        )
        self.joint_blocks = nn.ModuleList(
            [TransformerDecoderBlock(self.prompt_dim, num_heads, mlp_ratio) for _ in range(joint_num_layers)]
        )
        self.vertex_blocks = nn.ModuleList(
            [TransformerDecoderBlock(self.prompt_dim, num_heads, mlp_ratio) for _ in range(vertex_num_layers)]
        )
        residual_layers = tuple(int(layer) for layer in vertex_face_residual_layers)
        if len(set(residual_layers)) != len(residual_layers):
            raise ValueError("vertex_face_residual_layers must not contain duplicates")
        if any(layer < 0 or layer >= len(self.vertex_blocks) for layer in residual_layers):
            raise ValueError("vertex_face_residual_layers must index vertex decoder blocks")
        if not math.isfinite(float(vertex_face_residual_strength)) or not 0.0 <= float(
            vertex_face_residual_strength
        ) <= 1.0:
            raise ValueError("vertex_face_residual_strength must be finite and in [0, 1]")
        self.vertex_face_residuals = nn.ModuleDict()
        if enable_vertex_face_residual:
            if marker_faces is None:
                raise ValueError("vertex face residual requires marker_faces")
            self.vertex_face_residuals = nn.ModuleDict(
                {
                    str(layer): GatedFaceIncidenceResidual(
                        marker_faces,
                        marker_count=self.num_vertices,
                        gate_init=vertex_face_residual_gate_init,
                        residual_strength=vertex_face_residual_strength,
                    )
                    for layer in residual_layers
                }
            )
        self.root_output_norm = nn.LayerNorm(self.prompt_dim)
        self.joint_output_norm = nn.LayerNorm(self.prompt_dim)
        self.vertex_output_norm = nn.LayerNorm(self.prompt_dim)

        self.root_depth_prior_metric_head = nn.Linear(self.prompt_dim, 1)
        nn.init.normal_(self.root_depth_prior_metric_head.weight, mean=0.0, std=0.001)
        nn.init.constant_(self.root_depth_prior_metric_head.bias, 0.6)
        self.root_visibility_head = nn.Linear(self.prompt_dim, 1)
        self.root_contact_head = nn.Linear(self.prompt_dim, 1)
        self.root_contact_distance_head = nn.Linear(self.prompt_dim, 1)
        self.root_contact_log_conf_head = nn.Linear(self.prompt_dim, 1)
        self.root_log_conf_head = nn.Linear(self.prompt_dim, 1)
        self.joint_offset_metric_head = nn.Linear(self.prompt_dim, 3)
        self.joint_visibility_head = nn.Linear(self.prompt_dim, 1)
        self.joint_contact_head = nn.Linear(self.prompt_dim, 1)
        self.joint_contact_distance_head = nn.Linear(self.prompt_dim, 1)
        self.joint_contact_log_conf_head = nn.Linear(self.prompt_dim, 1)
        self.joint_log_conf_head = nn.Linear(self.prompt_dim, 1)
        self.vertex_offset_metric_head = nn.Linear(self.prompt_dim, 3)
        self.vertex_visibility_head = nn.Linear(self.prompt_dim, 1)
        self.vertex_contact_head = nn.Linear(self.prompt_dim, 1)
        self.vertex_contact_distance_head = nn.Linear(self.prompt_dim, 1)
        self.vertex_contact_log_conf_head = nn.Linear(self.prompt_dim, 1)
        self.vertex_log_conf_head = nn.Linear(self.prompt_dim, 1)
        confidence_heads = [
            self.root_log_conf_head,
            self.joint_log_conf_head,
            self.vertex_log_conf_head,
            self.root_contact_log_conf_head,
            self.joint_contact_log_conf_head,
            self.vertex_contact_log_conf_head,
        ]
        for conf_head in confidence_heads:
            nn.init.zeros_(conf_head.weight)
            nn.init.zeros_(conf_head.bias)
        for metric_head in (
            self.joint_offset_metric_head,
            self.vertex_offset_metric_head,
        ):
            nn.init.normal_(metric_head.weight, mean=0.0, std=0.001)
            nn.init.zeros_(metric_head.bias)

    def _checked_keypoint_inputs(
        self,
        *,
        final_prompt_set: torch.Tensor,
        keypoint_prompt_tokens: torch.Tensor,
        keypoint_geometry: torch.Tensor,
        keypoint_geometry_valid: torch.Tensor,
        keypoint_uv: torch.Tensor,
        keypoint_confidence: torch.Tensor,
        normalized_intrinsics: torch.Tensor,
        normalized_intrinsics_valid: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        batch_size = final_prompt_set.shape[0]
        expected_token_shape = (batch_size, self.num_joints, self.prompt_dim)
        if tuple(keypoint_prompt_tokens.shape) != expected_token_shape:
            raise ValueError(f"keypoint_prompt_tokens must have shape {expected_token_shape}.")
        if tuple(keypoint_geometry.shape) != (batch_size, self.num_joints, KEYPOINT_GEOMETRY_DIM):
            raise ValueError(f"keypoint_geometry must have shape [N, 21, {KEYPOINT_GEOMETRY_DIM}].")
        if tuple(keypoint_geometry_valid.shape) != (batch_size, self.num_joints):
            raise ValueError("keypoint_geometry_valid must have shape [N, 21].")
        if tuple(keypoint_uv.shape) != (batch_size, self.num_joints, 2):
            raise ValueError("keypoint_uv must have shape [N, 21, 2].")
        if tuple(keypoint_confidence.shape) != (batch_size, self.num_joints):
            raise ValueError("keypoint_confidence must have shape [N, 21].")
        if tuple(normalized_intrinsics.shape) != (batch_size, 4):
            raise ValueError("normalized_intrinsics must have shape [N, 4].")
        if tuple(normalized_intrinsics_valid.shape) != (batch_size,):
            raise ValueError("normalized_intrinsics_valid must have shape [N].")
        dtype = final_prompt_set.dtype
        device = final_prompt_set.device
        keypoint_prompt_tokens = keypoint_prompt_tokens.to(device=device, dtype=dtype)
        keypoint_geometry_valid = keypoint_geometry_valid.to(device=device, dtype=torch.bool)
        keypoint_geometry = torch.where(
            keypoint_geometry_valid.unsqueeze(-1),
            torch.nan_to_num(keypoint_geometry.to(device=device, dtype=dtype), nan=0.0, posinf=0.0, neginf=0.0),
            torch.zeros_like(keypoint_geometry, device=device, dtype=dtype),
        )
        geometry_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        keypoint_uv = keypoint_uv.detach().to(device=device, dtype=geometry_dtype)
        keypoint_confidence = keypoint_confidence.detach().to(device=device, dtype=geometry_dtype)
        normalized_intrinsics = normalized_intrinsics.detach().to(device=device, dtype=geometry_dtype)
        normalized_intrinsics_valid = normalized_intrinsics_valid.detach().to(
            device=device,
            dtype=torch.bool,
        )
        solver_keypoint_valid = (
            keypoint_geometry_valid
            & normalized_intrinsics_valid.unsqueeze(-1)
            & torch.isfinite(keypoint_uv).all(dim=-1)
            & torch.isfinite(keypoint_confidence)
        )
        keypoint_uv = torch.nan_to_num(keypoint_uv, nan=0.0, posinf=0.0, neginf=0.0)
        keypoint_confidence = torch.where(
            solver_keypoint_valid,
            torch.nan_to_num(keypoint_confidence, nan=0.0, posinf=0.0, neginf=0.0),
            torch.zeros_like(keypoint_confidence),
        )
        normalized_intrinsics = torch.nan_to_num(
            normalized_intrinsics,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        return (
            keypoint_prompt_tokens,
            keypoint_geometry,
            keypoint_geometry_valid,
            keypoint_uv,
            keypoint_confidence,
            solver_keypoint_valid,
            normalized_intrinsics,
        )

    def _project_hand_features(self, final_prompt_set: torch.Tensor, hand_features: torch.Tensor | None) -> list[torch.Tensor]:
        root_prompt, joint_prompt, vertex_prompt = final_prompt_set.unbind(dim=1)
        if hand_features is None:
            return [root_prompt[:, None], joint_prompt[:, None], vertex_prompt[:, None]]
        projected_features = self.hand_feature_proj(hand_features)
        if projected_features.ndim != 3 or tuple(projected_features.shape[:2]) != (final_prompt_set.shape[0], 3):
            raise ValueError("hand_features must have shape [N, 3, C].")
        return [projected_features[:, 0:1], projected_features[:, 1:2], projected_features[:, 2:3]]

    def _metric_memory(
        self,
        *,
        final_prompt_set: torch.Tensor,
        metric_prompt: torch.Tensor | None,
        metric_feature: torch.Tensor | None,
    ) -> list[torch.Tensor]:
        dtype = final_prompt_set.dtype
        device = final_prompt_set.device
        memory = []
        if metric_prompt is not None:
            if tuple(metric_prompt.shape) != (final_prompt_set.shape[0], 1, self.prompt_dim):
                raise ValueError("metric_prompt must have shape [N, 1, C].")
            memory.append(metric_prompt.to(device=device, dtype=dtype))
        if metric_feature is not None:
            if metric_feature.ndim != 2 or metric_feature.shape[0] != final_prompt_set.shape[0]:
                raise ValueError("metric_feature must have shape [N, C].")
            projected_metric = self.metric_feature_proj(metric_feature.to(device=device, dtype=dtype))
            memory.append(projected_metric[:, None])
        return memory

    def _view_memory(
        self,
        *,
        final_prompt_set: torch.Tensor,
        view_memory: torch.Tensor,
    ) -> list[torch.Tensor]:
        expected = (final_prompt_set.shape[0], 3, self.hand_feature_dim)
        if tuple(view_memory.shape) != expected:
            raise ValueError(f"view_memory must have shape {expected}.")
        dtype = final_prompt_set.dtype
        device = final_prompt_set.device
        view = self.view_feature_proj(view_memory.to(device=device, dtype=dtype))
        return [view[:, 0:1], view[:, 1:2], view[:, 2:3]]

    def forward(
        self,
        final_prompt_set: torch.Tensor,
        *,
        hand_features: torch.Tensor,
        view_memory: torch.Tensor,
        keypoint_prompt_tokens: torch.Tensor,
        root_condition_token_final: torch.Tensor,
        keypoint_geometry: torch.Tensor,
        keypoint_geometry_valid: torch.Tensor,
        keypoint_uv: torch.Tensor,
        keypoint_confidence: torch.Tensor,
        normalized_intrinsics: torch.Tensor,
        normalized_intrinsics_valid: torch.Tensor,
        image_size_hw: tuple[int, int] | None = None,
        metric_prompt: torch.Tensor | None = None,
        metric_feature: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if final_prompt_set.ndim != 3 or tuple(final_prompt_set.shape[1:]) != (3, self.prompt_dim):
            raise ValueError("final_prompt_set must have shape [N, 3, C].")
        root_prompt = final_prompt_set[:, 0]
        joint_prompt = final_prompt_set[:, 1]
        vertex_prompt = final_prompt_set[:, 2]
        if tuple(root_condition_token_final.shape) != (final_prompt_set.shape[0], self.prompt_dim):
            raise ValueError("root_condition_token_final must have shape [N, C].")
        root_condition_token_final = root_condition_token_final.to(
            device=final_prompt_set.device,
            dtype=final_prompt_set.dtype,
        )
        (
            keypoint_prompt_tokens,
            keypoint_geometry,
            keypoint_geometry_valid,
            keypoint_uv,
            keypoint_confidence,
            solver_keypoint_valid,
            normalized_intrinsics,
        ) = self._checked_keypoint_inputs(
            final_prompt_set=final_prompt_set,
            keypoint_prompt_tokens=keypoint_prompt_tokens,
            keypoint_geometry=keypoint_geometry,
            keypoint_geometry_valid=keypoint_geometry_valid,
            keypoint_uv=keypoint_uv,
            keypoint_confidence=keypoint_confidence,
            normalized_intrinsics=normalized_intrinsics,
            normalized_intrinsics_valid=normalized_intrinsics_valid,
        )
        root_feature, joint_feature, vertex_feature = self._project_hand_features(final_prompt_set, hand_features)
        metric_memory = self._metric_memory(
            final_prompt_set=final_prompt_set,
            metric_prompt=metric_prompt,
            metric_feature=metric_feature,
        )
        view_memory_by_type = self._view_memory(
            final_prompt_set=final_prompt_set,
            view_memory=view_memory,
        )

        root_query = (
            self.root_query.to(device=final_prompt_set.device, dtype=final_prompt_set.dtype).expand(final_prompt_set.shape[0], -1, -1)
            + self.root_query_proj(root_prompt)[:, None]
            + self.root_condition_proj(root_condition_token_final)[:, None]
        )
        root_memory = torch.cat(
            [
                root_prompt[:, None],
                root_feature,
                keypoint_prompt_tokens[:, 0:1],
                *metric_memory,
                view_memory_by_type[0],
            ],
            dim=1,
        )
        for block in self.root_blocks:
            root_query = block(root_query, root_memory)
        root_summary = self.root_output_norm(root_query[:, 0])
        root_depth_prior_metric_raw = self.root_depth_prior_metric_head(root_summary).squeeze(-1)
        geometry_dtype = torch.float64 if final_prompt_set.dtype == torch.float64 else torch.float32
        root_depth_prior_metric = 1e-4 + F.softplus(
            root_depth_prior_metric_raw.to(dtype=geometry_dtype) - 1e-4,
            beta=10.0,
            threshold=20.0,
        )
        root_visibility_logits = self.root_visibility_head(root_summary).squeeze(-1)
        root_log_conf = self.root_log_conf_head(root_summary).squeeze(-1)

        joint_geometry_feature = torch.where(
            keypoint_geometry_valid[:, 1:].unsqueeze(-1),
            self.joint_2d_proj(keypoint_geometry[:, 1:]),
            torch.zeros_like(keypoint_prompt_tokens[:, 1:]),
        )
        joint_queries = (
            self.joint_queries.to(device=final_prompt_set.device, dtype=final_prompt_set.dtype).expand(final_prompt_set.shape[0], -1, -1)
            + self.joint_query_proj(joint_prompt).unsqueeze(1)
            + self.joint_keypoint_proj(keypoint_prompt_tokens[:, 1:])
            + joint_geometry_feature
        )
        joint_memory = torch.cat(
            [
                root_summary[:, None],
                root_prompt[:, None],
                joint_prompt[:, None],
                root_feature,
                joint_feature,
                view_memory_by_type[1],
                keypoint_prompt_tokens,
            ],
            dim=1,
        )
        for block in self.joint_blocks:
            joint_queries = block(joint_queries, joint_memory)
        joint_summary = self.joint_output_norm(joint_queries)
        joint_offset_nonroot = self.joint_offset_metric_head(joint_summary).to(dtype=geometry_dtype)
        joint_offset = joint_offset_nonroot.new_zeros(
            joint_offset_nonroot.shape[0],
            self.num_joints,
            3,
        )
        joint_offset[:, 1:] = joint_offset_nonroot
        root_solve = solve_partial_observation_translation(
            joint_offset,
            keypoint_uv,
            keypoint_confidence,
            solver_keypoint_valid,
            normalized_intrinsics,
            root_depth_prior_metric,
            image_size_hw=image_size_hw,
        )
        root_translation_valid = root_solve.valid
        root_xyz = root_solve.translation
        residual_quality = torch.exp(
            -10.0
            * torch.nan_to_num(
                root_solve.weighted_residual,
                nan=float("inf"),
                posinf=float("inf"),
                neginf=float("inf"),
            )
        )
        observed_confidence = torch.where(
            solver_keypoint_valid,
            keypoint_confidence.clamp(0.0, 1.0),
            torch.zeros_like(keypoint_confidence),
        )
        observed_count = solver_keypoint_valid.sum(dim=-1)
        mean_observed_confidence = observed_confidence.sum(dim=-1) / observed_count.clamp_min(
            1
        ).to(dtype=observed_confidence.dtype)
        anchor_confidence = torch.where(
            root_solve.wrist_used,
            keypoint_confidence[:, 0].clamp(0.0, 1.0),
            mean_observed_confidence,
        )
        mode_reliability = root_xyz.new_tensor((0.0, 0.35, 0.70, 0.80, 1.0, 0.95)).index_select(
            0,
            root_solve.mode.reshape(-1),
        ).reshape_as(root_solve.mode)
        root_translation_anchor_quality = (
            anchor_confidence
            * residual_quality
            * mode_reliability
            * root_translation_valid.to(dtype=keypoint_confidence.dtype)
        ).clamp(0.0, 1.0)
        observation_reliability = root_subset_reliability(
            joint_offset, keypoint_uv, keypoint_confidence, solver_keypoint_valid,
            normalized_intrinsics, root_solve,
        )
        root_translation_reported_confidence = (
            root_translation_anchor_quality * observation_reliability.score
        ).clamp(0.0, 1.0)
        joint_xyz = root_xyz[:, None, :] + joint_offset
        joint_visibility_logits = root_visibility_logits.new_empty(root_xyz.shape[0], self.num_joints)
        joint_visibility_logits[:, 0] = root_visibility_logits
        joint_visibility_logits[:, 1:] = self.joint_visibility_head(joint_summary).squeeze(-1)
        joint_contact_logits = root_visibility_logits.new_empty(root_xyz.shape[0], self.num_joints)
        joint_contact_logits[:, 0] = self.root_contact_head(root_summary).squeeze(-1)
        joint_contact_logits[:, 1:] = self.joint_contact_head(joint_summary).squeeze(-1)
        joint_contact_log_distance = root_visibility_logits.new_empty(root_xyz.shape[0], self.num_joints)
        joint_contact_log_distance[:, 0] = self.root_contact_distance_head(root_summary).squeeze(-1)
        joint_contact_log_distance[:, 1:] = self.joint_contact_distance_head(joint_summary).squeeze(-1)
        joint_contact_log_conf = root_visibility_logits.new_empty(root_xyz.shape[0], self.num_joints)
        joint_contact_log_conf[:, 0] = self.root_contact_log_conf_head(root_summary).squeeze(-1)
        joint_contact_log_conf[:, 1:] = self.joint_contact_log_conf_head(joint_summary).squeeze(-1)
        joint_log_conf = root_log_conf.new_empty(root_xyz.shape[0], self.num_joints)
        joint_log_conf[:, 0] = root_log_conf
        joint_log_conf[:, 1:] = self.joint_log_conf_head(joint_summary).squeeze(-1)

        vertex_weights = self.vertex_joint_weights.to(device=final_prompt_set.device, dtype=final_prompt_set.dtype)
        vertex_keypoint_context = torch.einsum("vj,njc->nvc", vertex_weights, keypoint_prompt_tokens)
        vertex_queries = (
            self.vertex_queries.to(device=final_prompt_set.device, dtype=final_prompt_set.dtype).expand(final_prompt_set.shape[0], -1, -1)
            + self.vertex_query_proj(vertex_prompt).unsqueeze(1)
            + self.vertex_keypoint_proj(vertex_keypoint_context)
        )
        vertex_memory = torch.cat(
            [
                root_summary[:, None],
                root_prompt[:, None],
                vertex_prompt[:, None],
                root_feature,
                vertex_feature,
                view_memory_by_type[2],
                keypoint_prompt_tokens,
            ],
            dim=1,
        )
        for block_index, block in enumerate(self.vertex_blocks):
            vertex_queries = block(vertex_queries, vertex_memory)
            if str(block_index) in self.vertex_face_residuals:
                vertex_queries = self.vertex_face_residuals[str(block_index)](vertex_queries)
        vertex_summary = self.vertex_output_norm(vertex_queries)
        vertex_offset = self.vertex_offset_metric_head(vertex_summary).to(dtype=geometry_dtype)
        vertex_xyz = root_xyz.detach()[:, None, :] + vertex_offset
        vertex_contact_logits = self.vertex_contact_head(vertex_summary).squeeze(-1)
        vertex_contact_log_distance = self.vertex_contact_distance_head(vertex_summary).squeeze(-1)
        vertex_contact_log_conf = self.vertex_contact_log_conf_head(vertex_summary).squeeze(-1)
        vertex_log_conf = self.vertex_log_conf_head(vertex_summary).squeeze(-1)
        return {
            "root_depth_prior_metric_raw": root_depth_prior_metric_raw,
            "root_depth_prior_metric": root_depth_prior_metric,
            "root_translation_analytic": root_solve.analytic_translation,
            "root_translation_analytic_valid": root_solve.analytic_valid,
            "root_translation_valid": root_translation_valid,
            "root_translation_anchor_quality": root_translation_anchor_quality,
            "root_translation_reported_confidence": root_translation_reported_confidence,
            "root_translation_observation_reliability": observation_reliability.score,
            "root_depth_subset_spread_m": observation_reliability.spread_m,
            "root_depth_subset_relative_spread": observation_reliability.relative_spread,
            "root_depth_subset_valid_count": observation_reliability.valid_count,
            "root_depth_subset_valid": observation_reliability.valid,
            "root_translation_valid_joint_count": root_solve.valid_joint_count,
            "root_translation_weighted_residual": root_solve.weighted_residual,
            "root_translation_condition_number": root_solve.condition_number,
            "root_translation_depth_blend_weight": root_solve.depth_blend_weight,
            "root_translation_depth_condition_quality": root_solve.depth_condition_quality,
            "root_depth_geometric_quality": root_solve.depth_geometric_quality,
            "root_depth_prior_agreement": root_solve.depth_prior_agreement,
            "root_translation_analytic_weighted_residual": (
                root_solve.analytic_weighted_residual
            ),
            "root_translation_analytic_support_count": root_solve.analytic_support_count,
            "root_translation_mode": root_solve.mode,
            "root_translation_wrist_used": root_solve.wrist_used,
            "root_xy_consensus_valid_count": root_solve.xy_consensus_valid_count,
            "root_xy_consensus_gate": root_solve.xy_consensus_gate,
            "root_xy_consensus_correction_px": root_solve.xy_consensus_correction_px,
            "root_xy_consensus_residual": root_solve.xy_consensus_residual,
            "root_xy_consensus_corrected": root_solve.xy_consensus_corrected,
            "vertex_xyz": vertex_xyz,
            "vertex_offset": vertex_offset,
            "vertex_visibility_logits": self.vertex_visibility_head(vertex_summary).squeeze(-1),
            "vertex_contact_logits": vertex_contact_logits,
            "vertex_contact_log_distance": vertex_contact_log_distance,
            "vertex_contact_probability": torch.sigmoid(vertex_contact_logits),
            "vertex_contact_log_conf": vertex_contact_log_conf,
            "vertex_contact_conf": contact_confidence_from_log(vertex_contact_log_conf),
            "joint_xyz": joint_xyz,
            "joint_offset": joint_offset,
            "joint_visibility_logits": joint_visibility_logits,
            "joint_contact_logits": joint_contact_logits,
            "joint_contact_log_distance": joint_contact_log_distance,
            "joint_contact_probability": torch.sigmoid(joint_contact_logits),
            "joint_contact_log_conf": joint_contact_log_conf,
            "joint_contact_conf": contact_confidence_from_log(joint_contact_log_conf),
            "joint_log_conf": joint_log_conf,
            "joint_conf": torch.exp(joint_log_conf),
            "vertex_log_conf": vertex_log_conf,
            "vertex_conf": torch.exp(vertex_log_conf),
        }
