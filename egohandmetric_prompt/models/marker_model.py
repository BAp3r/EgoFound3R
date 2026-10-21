from __future__ import annotations


from dataclasses import dataclass
from typing import Any
import torch
from torch import nn
from torch.nn import functional as F
from egohandmetric_prompt.metric_geometry import (
    scene_average_distance_from_depth,
)
from egohandmetric_prompt.multirate_contracts import MultiRateFrameMap, gather_time_axis
from .camera_temporal_refiner import (
    FullRateRefinerOutput,
    TemporalInterpolationQueryMap,
    _geometry_dtype,
    clip_local_metric_w2c,
    smooth_high_rate_hand_root,
)
from .hand_head import HandHead, smooth_root_solver_wrist_uv
from .hand_in_view_head import HandInViewHead
from .hand_keypoint_localization_head import (
    HandKeypointLocalizationHead,
    decode_local_keypoint_cells,
)
from .hand_prompt_adapter import HandPromptAdapter
from .hand_view_fusion import HandViewFusionMLP, ViewMemoryProjector
from .metric_head import MetricHead
from .prompt_injected_aggregator_runner import PromptInjectedAggregatorRunner
from .temporal_hand_prompt_adapter import TemporalHandPromptAdapter


@dataclass(slots=True)
class MarkerBackboneFeatures:
    dino_feature_map: torch.Tensor
    dino_tokens: torch.Tensor
    mid_tokens: torch.Tensor
    camera_pose: torch.Tensor
    depth: torch.Tensor
    intrinsics: torch.Tensor
    camera_pose_encoding: torch.Tensor | None = None
    depth_conf: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class KeypointConditioning:
    keypoint_features: torch.Tensor
    keypoint_uv: torch.Tensor
    keypoint_conf: torch.Tensor
    keypoint_mask: torch.Tensor

    def validate(self) -> None:
        if self.keypoint_uv.ndim != 5 or self.keypoint_uv.shape[2] != 2 or self.keypoint_uv.shape[-1] != 2:
            raise ValueError("keypoint_uv must have shape [B, T, 2, J, 2]")
        expected = self.keypoint_uv.shape[:-1]
        if self.keypoint_features.shape[:-1] != expected:
            raise ValueError("keypoint_features must align with keypoint_uv")
        if self.keypoint_conf.shape != expected or self.keypoint_mask.shape != expected:
            raise ValueError("keypoint confidence/mask must align with keypoint_uv")
        if self.keypoint_mask.dtype is not torch.bool:
            raise ValueError("keypoint_mask must be bool")
        for name, value in (
            ("keypoint_features", self.keypoint_features),
            ("keypoint_uv", self.keypoint_uv),
            ("keypoint_conf", self.keypoint_conf),
        ):
            if value.requires_grad:
                raise ValueError(f"{name} must be detached before hand-side consumption")


@dataclass(frozen=True, slots=True)
class HandStageGatePlan:
    stage_index: int = 4
    view_prompt_delta: float = 1.0
    view_memory: float = 1.0

    def validate(self) -> None:
        if self.stage_index not in (1, 2, 3, 4):
            raise ValueError("hand stage index must be in [1, 4]")
        for name, value in (
            ("view_prompt_delta", self.view_prompt_delta),
            ("view_memory", self.view_memory),
        ):
            if float(value) not in (0.0, 1.0):
                raise ValueError(f"{name} must be exactly 0.0 or 1.0")


@dataclass(slots=True)
class _FrontHalf:
    heatmap_logits: torch.Tensor
    keypoint_offset: torch.Tensor
    coarse_heatmap_logits: torch.Tensor
    center_logits: torch.Tensor
    keypoint_peak_logits: torch.Tensor
    keypoint_peak_conf: torch.Tensor
    predicted_keypoints: KeypointConditioning
    selected_keypoints: KeypointConditioning
    hand_in_view_logits: torch.Tensor
    in_view_probability: torch.Tensor
    adapter_outputs: dict[str, torch.Tensor]
    hand_prompt_set: torch.Tensor
    metric_feature: torch.Tensor


@dataclass(frozen=True, slots=True)
class Global3ROutput:
    aggregated_tokens_list: tuple[torch.Tensor | None, ...]
    camera_pose_encoding: torch.Tensor
    camera_pose: torch.Tensor
    intrinsics: torch.Tensor
    depth: torch.Tensor
    depth_conf: torch.Tensor


@dataclass(frozen=True, slots=True)
class MetricOutput:
    log_metric_value: torch.Tensor
    metric_value: torch.Tensor


@dataclass(frozen=True, slots=True)
class HighRateHandOutput:
    values: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return dict(self.values)


@dataclass(frozen=True, slots=True)
class MultiRateMarkerOutput:
    high_rate: HighRateHandOutput
    global_3r: Global3ROutput
    high_rate_camera: Any
    full_rate_interpolation: FullRateRefinerOutput
    metric: MetricOutput

    def to_dict(self) -> dict[str, Any]:
        result = self.high_rate.to_dict()
        result.update(
            {
                "camera_pose_refined_high": self.high_rate_camera.camera_pose,
                "camera_extrinsic_vector_base_high": self.high_rate_camera.extrinsic_vector_base,
                "camera_extrinsic_vector_refined_high": self.high_rate_camera.extrinsic_vector_refined,
                "camera_residual_vector_high": self.high_rate_camera.residual_vector,
                "camera_anchor_exact_mask_high": self.high_rate_camera.anchor_exact_mask,
                "camera_refined_valid_high": self.high_rate_camera.camera_refined_valid,
                "camera_pose_global": self.global_3r.camera_pose,
                "camera_pose_encoding_global": self.global_3r.camera_pose_encoding,
                "intrinsics_global": self.global_3r.intrinsics,
                "depth_global": self.global_3r.depth,
                "depth_conf_global": self.global_3r.depth_conf,
                "log_metric_value": self.metric.log_metric_value,
                "metric_value": self.metric.metric_value,
            }
        )
        result.update(self.full_rate_interpolation.to_public_dict())
        return result










class MultiRateMarkerComponents(nn.Module):
    def __init__(
        self,
        *,
        hand_global_token_bridge: nn.Module,
        hand_prompt_pooler: nn.Module,
        metric_prompt_pooler: nn.Module,
        metric_feature_pooler: nn.Module,
        final_hand_prompt_bridge: nn.Module,
        final_metric_prompt_bridge: nn.Module,
        temporal_interpolator: nn.Module,
    ) -> None:
        super().__init__()
        components = {
            "hand_global_token_bridge": hand_global_token_bridge,
            "hand_prompt_pooler": hand_prompt_pooler,
            "metric_prompt_pooler": metric_prompt_pooler,
            "metric_feature_pooler": metric_feature_pooler,
            "final_hand_prompt_bridge": final_hand_prompt_bridge,
            "final_metric_prompt_bridge": final_metric_prompt_bridge,
            "temporal_interpolator": temporal_interpolator,
        }
        for name, module in components.items():
            if not isinstance(module, nn.Module):
                raise TypeError(f"{name} must be an nn.Module")
            setattr(self, name, module)


class MarkerModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        keypoint_head: HandKeypointLocalizationHead,
        hand_adapter: HandPromptAdapter,
        temporal_adapter: TemporalHandPromptAdapter,
        hand_head: HandHead,
        *,
        hand_in_view_head: HandInViewHead,
        hand_view_fusion: HandViewFusionMLP,
        view_memory_projector: ViewMemoryProjector,
        metric_head: MetricHead | None = None,
        keypoint_peak_threshold: float = 0.5,
        multirate_components: MultiRateMarkerComponents,
    ) -> None:
        super().__init__()
        if not 0.0 <= float(keypoint_peak_threshold) <= 1.0:
            raise ValueError("keypoint_peak_threshold must lie in [0, 1]")
        self.backbone = backbone
        self.keypoint_head = keypoint_head
        self.hand_adapter = hand_adapter
        self.temporal_adapter = temporal_adapter
        self.hand_head = hand_head
        self.hand_in_view_head = hand_in_view_head
        self.hand_view_fusion = hand_view_fusion
        self.view_memory_projector = view_memory_projector
        self.keypoint_peak_threshold = float(keypoint_peak_threshold)
        self.hand_feature_dim = hand_adapter.hand_feature_dim
        self.metric_head = metric_head if metric_head is not None else MetricHead(
            prompt_dim=hand_head.prompt_dim,
            hand_feature_dim=self.hand_feature_dim,
        )
        self.local_dino_sample_gate = nn.Parameter(torch.tensor(0.25))
        self.local_dino_sample_score = nn.Linear(hand_adapter.dino_proj.in_features, 1)
        self.intrinsics_delta_gate = nn.Parameter(torch.tensor(0.1))
        self.inject_layer_idx = getattr(backbone, "mid_layer_index", 11)
        self.prompt_to_backbone = nn.Linear(hand_head.prompt_dim, keypoint_head.in_channels)
        self.backbone_to_prompt = nn.Linear(keypoint_head.in_channels, hand_head.prompt_dim)
        self.multirate_components = multirate_components


    @staticmethod
    def _semantic_grid(uv: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
        uv = uv.to(dtype=torch.float32)
        pixels = torch.stack([uv[..., 0] * float(width - 1), uv[..., 1] * float(height - 1)], dim=-1)
        size = pixels.new_tensor([float(width), float(height)])
        return 2.0 * (pixels + 0.5) / size - 1.0

    def _sample_dino_at_uv(self, dino_feature_map: torch.Tensor, uv: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, frames, channels, height, width = dino_feature_map.shape
        if uv.shape[:3] != (batch, frames, 2) or uv.shape[-1] != 2:
            raise ValueError("UV must align with DINO feature maps")
        joints = uv.shape[3]
        if mask.shape != uv.shape[:-1]:
            raise ValueError("keypoint mask must align with UV")
        flat_map = dino_feature_map.reshape(batch * frames, channels, height, width)
        flat_uv = torch.nan_to_num(uv.float(), nan=0.0, posinf=0.0, neginf=0.0).reshape(
            batch * frames, 2 * joints, 2
        )
        center_grid = self._semantic_grid(flat_uv, height=height, width=width)
        offsets = center_grid.new_tensor(
            (
                (0.0, 0.0),
                (-1.0 / float(width), 0.0),
                (1.0 / float(width), 0.0),
                (0.0, -1.0 / float(height)),
                (0.0, 1.0 / float(height)),
            )
        )
        grid = center_grid.unsqueeze(2) + offsets.view(1, 1, 5, 2)
        with torch.autocast(device_type=flat_map.device.type, enabled=False):
            sampled = F.grid_sample(
                flat_map.float(),
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )
            samples = sampled.permute(0, 2, 3, 1)
            scorer_dtype = self.local_dino_sample_score.weight.dtype
            local_scores = self.local_dino_sample_score(samples.to(dtype=scorer_dtype)).squeeze(-1)
            local_weights = torch.softmax(local_scores.float(), dim=-1)
            center = samples[:, :, 0]
            neighborhood = (samples * local_weights.to(dtype=samples.dtype).unsqueeze(-1)).sum(dim=2)
            gate = torch.tanh(self.local_dino_sample_gate.float())
            sampled = center + gate * (neighborhood - center)
        sampled = sampled.reshape(batch, frames, 2, joints, channels).to(dtype=flat_map.dtype)
        valid = mask.to(device=flat_map.device, dtype=torch.bool).unsqueeze(-1)
        return torch.where(valid, sampled, torch.zeros_like(sampled))

    @staticmethod
    def _decode_semantic_uv(
        *,
        heatmap_logits: torch.Tensor,
        keypoint_offset: torch.Tensor,
        image_height: int,
        image_width: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        continuous_xy, peak_logits, peak_conf, cells = decode_local_keypoint_cells(
            heatmap_logits,
            keypoint_offset,
        )
        heatmap_height, heatmap_width = heatmap_logits.shape[-2:]
        continuous_x = continuous_xy[..., 0]
        continuous_y = continuous_xy[..., 1]
        pixel_x = (continuous_x + 0.5) * float(image_width) / float(heatmap_width) - 0.5
        pixel_y = (continuous_y + 0.5) * float(image_height) / float(heatmap_height) - 0.5
        uv = torch.stack(
            [pixel_x / float(max(image_width - 1, 1)), pixel_y / float(max(image_height - 1, 1))],
            dim=-1,
        ).clamp(0.0, 1.0)
        return uv, peak_logits, peak_conf, cells

    def _supports_multirate_prompt_injection(self) -> bool:
        return all(
            hasattr(self.backbone, name)
            for name in (
                "encode_dino_high_rate",
                "run_global_prefix_from_dino",
                "run_suffix",
                "decode_camera_pose_encoding",
                "decode_camera_from_pose_encoding",
                "decode_depth_with_conf",
                "build_intrinsics",
            )
        )

    def _require_multirate_components(self) -> MultiRateMarkerComponents:
        if not self._supports_multirate_prompt_injection():
            raise RuntimeError(
                "the configured multi-rate topology requires a split frozen backbone "
                "(encode_dino_high_rate/run_global_prefix_from_dino/run_suffix/decode API)"
            )
        return self.multirate_components


    def _prediction_keypoint_conditioning(
        self,
        *,
        dino_feature_map: torch.Tensor,
        heatmap_logits: torch.Tensor,
        keypoint_offset: torch.Tensor,
        image_height: int,
        image_width: int,
        side_in_view_probability: torch.Tensor | None = None,
        sample_features: bool = True,
    ) -> tuple[KeypointConditioning, torch.Tensor, torch.Tensor, torch.Tensor]:
        uv, peak_logits, peak_conf, cells = self._decode_semantic_uv(
            heatmap_logits=heatmap_logits,
            keypoint_offset=keypoint_offset,
            image_height=image_height,
            image_width=image_width,
        )
        if side_in_view_probability is None:
            side_probability = torch.ones_like(peak_conf[..., 0])
        else:
            side_probability = side_in_view_probability.detach().to(
                device=peak_conf.device,
                dtype=peak_conf.dtype,
            )
            if side_probability.shape != peak_conf.shape[:3]:
                raise ValueError("side_in_view_probability must have shape [B, T, 2]")
            side_probability = torch.nan_to_num(
                side_probability,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).clamp(0.0, 1.0)
        effective_conf = peak_conf * side_probability.unsqueeze(-1)
        finite = torch.isfinite(uv).all(dim=-1) & torch.isfinite(peak_conf)
        side_present = side_probability >= 0.5
        joint_visible = peak_conf >= float(self.keypoint_peak_threshold)
        mask = side_present.unsqueeze(-1) & joint_visible & finite
        features = (
            self._sample_dino_at_uv(dino_feature_map, uv, mask)
            if sample_features
            else dino_feature_map.new_zeros(*mask.shape, dino_feature_map.shape[2])
        )
        conditioning = KeypointConditioning(
            keypoint_features=features.detach(),
            keypoint_uv=torch.where(mask.unsqueeze(-1), uv, torch.zeros_like(uv)).detach(),
            keypoint_conf=torch.where(mask, effective_conf, torch.zeros_like(effective_conf)).detach(),
            keypoint_mask=mask.detach().to(dtype=torch.bool),
        )
        conditioning.validate()
        return conditioning, peak_logits, peak_conf, cells

    def _sample_selected_keypoint_conditioning(
        self,
        *,
        dino_feature_map: torch.Tensor,
        selected: KeypointConditioning,
    ) -> KeypointConditioning:
        selected.validate()
        sampled = KeypointConditioning(
            keypoint_features=self._sample_dino_at_uv(
                dino_feature_map,
                selected.keypoint_uv,
                selected.keypoint_mask,
            ).detach(),
            keypoint_uv=selected.keypoint_uv.detach(),
            keypoint_conf=selected.keypoint_conf.detach(),
            keypoint_mask=selected.keypoint_mask.detach(),
        )
        sampled.validate()
        return sampled

    def _front_half(
        self,
        *,
        images: torch.Tensor,
        dino_feature_map: torch.Tensor,
        dino_tokens: torch.Tensor,
        mid_tokens: torch.Tensor,
        frame_present_mask: torch.Tensor,
        stage_plan: HandStageGatePlan,
    ) -> _FrontHalf:
        keypoint_outputs = self.keypoint_head(
            dino_feature_map,
            frame_present_mask=frame_present_mask,
        )
        heatmap_logits = keypoint_outputs["keypoint_heatmap_logits"]
        keypoint_offset = keypoint_outputs["keypoint_offset"]
        coarse_heatmap_logits = keypoint_outputs["keypoint_coarse_heatmap_logits"]
        center_logits = keypoint_outputs["keypoint_center_logits"]
        hand_in_view_logits = self.hand_in_view_head(dino_feature_map.detach())
        in_view_probability = hand_in_view_logits.detach().sigmoid()
        prediction, peak_logits, peak_conf, _ = self._prediction_keypoint_conditioning(
            dino_feature_map=dino_feature_map,
            heatmap_logits=heatmap_logits,
            keypoint_offset=keypoint_offset,
            image_height=images.shape[-2],
            image_width=images.shape[-1],
            side_in_view_probability=in_view_probability,
            sample_features=False,
        )
        selected = self._sample_selected_keypoint_conditioning(
            dino_feature_map=dino_feature_map,
            selected=prediction,
        )
        adapter_outputs = self.hand_adapter(
            dino_tokens=dino_tokens,
            agg_tokens=mid_tokens,
            keypoint_features=selected.keypoint_features,
            keypoint_uv=selected.keypoint_uv,
            keypoint_conf=selected.keypoint_conf,
            keypoint_mask=selected.keypoint_mask,
        )
        base_keypoint_prompt_tokens = adapter_outputs["keypoint_prompt_tokens"]
        adapter_outputs["keypoint_prompt_tokens_base"] = base_keypoint_prompt_tokens
        adapter_outputs["root_condition_token_final"] = adapter_outputs[
            "semantic_condition_token_visual"
        ][..., 0, :]
        view_fused, _ = self.hand_view_fusion(
            base_hand_prompt=adapter_outputs["hand_prompt_set"],
            keypoint_context=adapter_outputs["semantic_keypoint_context"],
            in_view_probability=in_view_probability,
            stage_multiplier=stage_plan.view_prompt_delta,
        )
        hand_prompt_set = self.temporal_adapter(view_fused)
        metric_feature = self.metric_head.fuse_metric_feature(
            scene_metric_feature=adapter_outputs["scene_metric_feature"],
            hand_feature_set=adapter_outputs["hand_feature_set"],
            in_view_probability=in_view_probability,
        )
        return _FrontHalf(
            heatmap_logits=heatmap_logits,
            keypoint_offset=keypoint_offset,
            coarse_heatmap_logits=coarse_heatmap_logits,
            center_logits=center_logits,
            keypoint_peak_logits=peak_logits,
            keypoint_peak_conf=peak_conf,
            predicted_keypoints=prediction,
            selected_keypoints=selected,
            hand_in_view_logits=hand_in_view_logits,
            in_view_probability=in_view_probability,
            adapter_outputs=adapter_outputs,
            hand_prompt_set=hand_prompt_set,
            metric_feature=metric_feature,
        )

    @staticmethod
    def _normalized_intrinsics(
        intrinsics: torch.Tensor,
        *,
        image_height: int,
        image_width: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if intrinsics.ndim != 4 or intrinsics.shape[-2:] != (3, 3):
            raise ValueError("intrinsics must have shape [B, T, 3, 3]")
        scale_x = float(max(image_width - 1, 1))
        scale_y = float(max(image_height - 1, 1))
        normalized = torch.stack(
            (
                intrinsics[..., 0, 0] / scale_x,
                intrinsics[..., 1, 1] / scale_y,
                intrinsics[..., 0, 2] / scale_x,
                intrinsics[..., 1, 2] / scale_y,
            ),
            dim=-1,
        )
        valid = (
            torch.isfinite(normalized).all(dim=-1)
            & (normalized[..., 0] > 1e-6)
            & (normalized[..., 1] > 1e-6)
        )
        return torch.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0), valid

    @staticmethod
    def _masked_component_median(value: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        masked = torch.where(valid.unsqueeze(-1), value, torch.full_like(value, float("nan"))).float()
        ordered = torch.sort(masked, dim=1, stable=True).values
        count = torch.isfinite(ordered).sum(dim=1, keepdim=True)
        index = (count - 1).clamp_min(0) // 2
        median = ordered.gather(1, index.expand(-1, -1, ordered.shape[-1])).squeeze(1)
        median = median.to(dtype=value.dtype)
        consensus_valid = valid.any(dim=1)
        median = torch.where(consensus_valid.unsqueeze(-1), median, torch.zeros_like(median))
        return median, consensus_valid

    def _multirate_intrinsics_condition(
        self,
        intrinsics_global: torch.Tensor,
        frame_map: MultiRateFrameMap,
        *,
        image_height: int,
        image_width: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized, valid = self._normalized_intrinsics(
            intrinsics_global.detach(),
            image_height=image_height,
            image_width=image_width,
        )
        valid &= frame_map.global_frame_present
        consensus, consensus_valid = self._masked_component_median(normalized, valid)
        residual = torch.where(valid.unsqueeze(-1), normalized - consensus.unsqueeze(1), torch.zeros_like(normalized))
        max_index = max(intrinsics_global.shape[1] - 1, 0)
        left_index = frame_map.left_anchor.clamp(0, max_index)
        right_index = frame_map.right_anchor.clamp(0, max_index)
        gather_index = left_index.unsqueeze(-1).expand(-1, -1, 4)
        left_residual = residual.gather(1, gather_index)
        right_residual = residual.gather(1, right_index.unsqueeze(-1).expand(-1, -1, 4))
        left_valid = frame_map.left_anchor_valid & valid.gather(1, left_index)
        right_valid = frame_map.right_anchor_valid & valid.gather(1, right_index)
        alpha = frame_map.alpha_right.to(device=normalized.device, dtype=normalized.dtype).unsqueeze(-1)
        interpolated = (1.0 - alpha) * left_residual + alpha * right_residual
        interpolated = torch.where(left_valid.unsqueeze(-1), interpolated, right_residual)
        interpolated = torch.where(right_valid.unsqueeze(-1), interpolated, left_residual)
        endpoint_valid = left_valid | right_valid
        gate = torch.tanh(self.intrinsics_delta_gate).to(dtype=normalized.dtype)
        condition = consensus.unsqueeze(1) + gate * interpolated
        condition_valid = frame_map.high_frame_present & consensus_valid.unsqueeze(1) & endpoint_valid
        condition = torch.where(condition_valid.unsqueeze(-1), condition, torch.zeros_like(condition))
        return condition, condition_valid

    @staticmethod
    def _provided_root_intrinsics_condition(
        intrinsics_high: torch.Tensor,
        intrinsics_valid_high: torch.Tensor | None,
        frame_present: torch.Tensor,
        *,
        image_height: int,
        image_width: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expected_intrinsics_shape = (*frame_present.shape, 3, 3)
        if tuple(intrinsics_high.shape) != expected_intrinsics_shape:
            raise ValueError(
                "root_solver_intrinsics must have shape "
                f"{expected_intrinsics_shape}, got {tuple(intrinsics_high.shape)}"
            )
        if intrinsics_high.device != frame_present.device:
            raise ValueError("root_solver_intrinsics and frame masks must share the model device")
        if intrinsics_valid_high is None:
            provided_valid = torch.ones_like(frame_present, dtype=torch.bool)
        else:
            if tuple(intrinsics_valid_high.shape) != tuple(frame_present.shape):
                raise ValueError("root_solver_intrinsics_valid must have shape [B, T_H]")
            if intrinsics_valid_high.dtype is not torch.bool:
                raise ValueError("root_solver_intrinsics_valid must be bool")
            if intrinsics_valid_high.device != frame_present.device:
                raise ValueError(
                    "root_solver_intrinsics_valid and frame masks must share the model device"
                )
            provided_valid = intrinsics_valid_high.detach()
        normalized, finite_valid = MarkerModel._normalized_intrinsics(
            intrinsics_high.detach(),
            image_height=image_height,
            image_width=image_width,
        )
        condition_valid = frame_present & provided_valid & finite_valid
        return (
            torch.where(
                condition_valid.unsqueeze(-1),
                normalized,
                torch.zeros_like(normalized),
            ),
            condition_valid,
        )

    def _view_memory(
        self,
        *,
        front: _FrontHalf,
        stage_plan: HandStageGatePlan,
    ) -> torch.Tensor:
        return self.view_memory_projector(
            front.in_view_probability,
            stage_multiplier=stage_plan.view_memory,
        )

    def _decode_fixed_hand_outputs(
        self,
        *,
        final_hand_prompts: torch.Tensor,
        final_metric_prompt: torch.Tensor,
        front: _FrontHalf,
        view_memory: torch.Tensor,
        intrinsics_condition_high: torch.Tensor | None = None,
        intrinsics_condition_valid_high: torch.Tensor | None = None,
        image_size_hw: tuple[int, int] | None = None,
        root_uv_smoothing: bool = False,
    ) -> dict[str, torch.Tensor]:
        batch, frames, sides = final_hand_prompts.shape[:3]
        flat = batch * frames * sides
        def flatten(value: torch.Tensor, trailing_dims: int) -> torch.Tensor:
            return value.reshape(flat, *value.shape[-trailing_dims:])
        if intrinsics_condition_high is None or intrinsics_condition_valid_high is None:
            intrinsics_condition_high = final_hand_prompts.new_zeros(
                batch,
                frames,
                4,
            )
            intrinsics_condition_valid_high = torch.zeros(
                batch,
                frames,
                device=final_hand_prompts.device,
                dtype=torch.bool,
            )
        metric_prompt = final_metric_prompt[:, :, None].expand(batch, frames, sides, 1, final_metric_prompt.shape[-1])
        metric_feature = front.metric_feature[:, :, None].expand(batch, frames, sides, front.metric_feature.shape[-1])
        selected_uv = front.selected_keypoints.keypoint_uv
        selected_confidence = front.selected_keypoints.keypoint_conf
        selected_mask = front.selected_keypoints.keypoint_mask
        if root_uv_smoothing:
            solver_uv, root_uv_smoothing_applied = smooth_root_solver_wrist_uv(
                selected_uv,
                selected_confidence,
                selected_mask,
            )
        else:
            solver_uv = selected_uv
            root_uv_smoothing_applied = torch.zeros(
                selected_uv.shape[:3],
                device=selected_uv.device,
                dtype=torch.bool,
            )
        hand_outputs = self.hand_head(
            flatten(final_hand_prompts, 2),
            hand_features=flatten(front.adapter_outputs["hand_feature_set"], 2),
            view_memory=flatten(view_memory, 2),
            keypoint_prompt_tokens=flatten(front.adapter_outputs["keypoint_prompt_tokens"], 2),
            root_condition_token_final=flatten(
                front.adapter_outputs["root_condition_token_final"],
                1,
            ),
            keypoint_geometry=flatten(front.adapter_outputs["keypoint_geometry"], 2),
            keypoint_geometry_valid=front.adapter_outputs["keypoint_geometry_valid"].reshape(flat, -1),
            keypoint_uv=flatten(
                solver_uv,
                2,
            ),
            keypoint_confidence=selected_confidence.reshape(flat, -1),
            normalized_intrinsics=intrinsics_condition_high[:, :, None].expand(
                batch,
                frames,
                sides,
                4,
            ).reshape(flat, 4),
            normalized_intrinsics_valid=intrinsics_condition_valid_high[:, :, None].expand(
                batch,
                frames,
                sides,
            ).reshape(flat),
            image_size_hw=image_size_hw,
            metric_prompt=metric_prompt.reshape(flat, 1, metric_prompt.shape[-1]),
            metric_feature=metric_feature.reshape(flat, -1),
        )
        result = {name: value.reshape(batch, frames, sides, *value.shape[1:]) for name, value in hand_outputs.items()}
        result["_root_solver_keypoint_uv"] = solver_uv
        result["_root_solver_keypoint_mask"] = selected_mask
        result["_root_uv_smoothing_applied"] = root_uv_smoothing_applied
        return result

    def _format_outputs(
        self,
        *,
        camera_pose: torch.Tensor,
        camera_pose_encoding: torch.Tensor | None,
        depth: torch.Tensor,
        depth_conf: torch.Tensor | None,
        intrinsics: torch.Tensor,
        front: _FrontHalf,
        final_hand_prompts: torch.Tensor,
        final_metric_prompt: torch.Tensor,
        hand_outputs: dict[str, torch.Tensor],
        metric_outputs: dict[str, torch.Tensor],
        view_memory: torch.Tensor,
        root_solver_intrinsics_condition: torch.Tensor,
        root_solver_intrinsics_condition_valid: torch.Tensor,
        root_solver_intrinsics_is_external: bool,
    ) -> dict[str, Any]:
        batch, frames, sides = final_hand_prompts.shape[:3]
        flat = batch * frames * sides
        if tuple(root_solver_intrinsics_condition.shape) != (batch, frames, 4):
            raise ValueError(
                "root_solver_intrinsics_condition must have shape [B, T_H, 4]"
            )
        if tuple(root_solver_intrinsics_condition_valid.shape) != (batch, frames):
            raise ValueError(
                "root_solver_intrinsics_condition_valid must have shape [B, T_H]"
            )
        root_xyz = hand_outputs["joint_xyz"][..., 0, :]
        root_log_conf = hand_outputs["joint_log_conf"][..., :1]
        root_conf = hand_outputs["joint_conf"][..., :1]
        outputs: dict[str, Any] = {
            "camera_pose": camera_pose,
            "camera_pose_encoding": camera_pose_encoding,
            "depth": depth,
            "depth_conf": depth_conf,
            "intrinsics": intrinsics,
            "heatmap_logits": front.heatmap_logits,
            "keypoint_heatmap_logits": front.heatmap_logits,
            "keypoint_offset": front.keypoint_offset,
            "keypoint_coarse_heatmap_logits": front.coarse_heatmap_logits,
            "keypoint_center_logits": front.center_logits,
            "keypoint_peak_logits": front.keypoint_peak_logits,
            "keypoint_peak_conf": front.keypoint_peak_conf,
            "keypoint_uv": front.selected_keypoints.keypoint_uv,
            "keypoint_conf": front.selected_keypoints.keypoint_conf,
            "keypoint_mask": front.selected_keypoints.keypoint_mask,
            "predicted_keypoint_uv": front.predicted_keypoints.keypoint_uv,
            "predicted_keypoint_conf": front.predicted_keypoints.keypoint_conf,
            "predicted_keypoint_mask": front.predicted_keypoints.keypoint_mask,
            "root_depth_prior_metric": hand_outputs["root_depth_prior_metric"],
            "root_translation_analytic": hand_outputs["root_translation_analytic"],
            "root_translation_analytic_valid": hand_outputs["root_translation_analytic_valid"],
            "root_translation_valid": hand_outputs["root_translation_valid"],
            "root_translation_anchor_quality": hand_outputs[
                "root_translation_anchor_quality"
            ],
            "root_translation_reported_confidence": hand_outputs[
                "root_translation_reported_confidence"
            ],
            "root_translation_observation_reliability": hand_outputs["root_translation_observation_reliability"],
            "root_depth_subset_spread_m": hand_outputs["root_depth_subset_spread_m"],
            "root_depth_subset_relative_spread": hand_outputs["root_depth_subset_relative_spread"],
            "root_depth_subset_valid_count": hand_outputs["root_depth_subset_valid_count"],
            "root_depth_subset_valid": hand_outputs["root_depth_subset_valid"],
            "root_translation_valid_joint_count": hand_outputs[
                "root_translation_valid_joint_count"
            ],
            "root_translation_weighted_residual": hand_outputs[
                "root_translation_weighted_residual"
            ],
            "root_translation_condition_number": hand_outputs[
                "root_translation_condition_number"
            ],
            "root_translation_depth_blend_weight": hand_outputs[
                "root_translation_depth_blend_weight"
            ],
            "root_translation_depth_condition_quality": hand_outputs[
                "root_translation_depth_condition_quality"
            ],
            "root_depth_geometric_quality": hand_outputs[
                "root_depth_geometric_quality"
            ],
            "root_depth_prior_agreement": hand_outputs[
                "root_depth_prior_agreement"
            ],
            "root_translation_analytic_weighted_residual": hand_outputs[
                "root_translation_analytic_weighted_residual"
            ],
            "root_translation_analytic_support_count": hand_outputs[
                "root_translation_analytic_support_count"
            ],
            "root_translation_mode": hand_outputs["root_translation_mode"],
            "root_translation_wrist_used": hand_outputs[
                "root_translation_wrist_used"
            ],
            "root_xy_consensus_valid_count": hand_outputs[
                "root_xy_consensus_valid_count"
            ],
            "root_xy_consensus_gate": hand_outputs["root_xy_consensus_gate"],
            "root_xy_consensus_correction_px": hand_outputs[
                "root_xy_consensus_correction_px"
            ],
            "root_xy_consensus_residual": hand_outputs[
                "root_xy_consensus_residual"
            ],
            "root_xy_consensus_corrected": hand_outputs[
                "root_xy_consensus_corrected"
            ],
            "root_solver_normalized_intrinsics": root_solver_intrinsics_condition.detach(),
            "root_solver_intrinsics_valid": root_solver_intrinsics_condition_valid.detach(),
            "root_solver_intrinsics_is_external": torch.full(
                (batch, frames),
                bool(root_solver_intrinsics_is_external),
                device=root_solver_intrinsics_condition.device,
                dtype=torch.bool,
            ),
            "keypoint_prompt_tokens": front.adapter_outputs["keypoint_prompt_tokens"],
            "keypoint_prompt_tokens_base": front.adapter_outputs["keypoint_prompt_tokens_base"],
            "hand_in_view_logits": front.hand_in_view_logits,
            "in_view_probability": front.in_view_probability,
            "hand_feature_set": front.adapter_outputs["hand_feature_set"],
            "hand_prompt_set": front.hand_prompt_set,
            "metric_feature": front.metric_feature,
            "scene_metric_feature": front.adapter_outputs["scene_metric_feature"],
            "metric_prompt": front.adapter_outputs["scene_metric_prompt"],
            "final_hand_prompts": final_hand_prompts,
            "final_metric_prompt": final_metric_prompt,
            "view_memory": view_memory,
            "log_metric_value": metric_outputs["log_metric_value"],
            "metric_value": metric_outputs["metric_value"],
            "dense_vertex_xyz": hand_outputs["vertex_xyz"],
            "dense_vertex_offset": hand_outputs["vertex_offset"],
            "dense_vertex_visibility_logits": hand_outputs["vertex_visibility_logits"],
            "dense_vertex_contact_logits": hand_outputs["vertex_contact_logits"],
            "dense_vertex_contact_log_distance": hand_outputs["vertex_contact_log_distance"],
            "dense_vertex_contact_probability": hand_outputs["vertex_contact_probability"],
            "dense_vertex_contact_log_conf": hand_outputs["vertex_contact_log_conf"],
            "dense_vertex_contact_conf": hand_outputs["vertex_contact_conf"],
            "dense_vertex_log_conf": hand_outputs["vertex_log_conf"],
            "dense_vertex_conf": hand_outputs["vertex_conf"],
            "dense_hand_root_xyz": root_xyz,
            "dense_hand_root_log_conf": root_log_conf,
            "dense_hand_root_conf": root_conf,
            "dense_joint_xyz": hand_outputs["joint_xyz"],
            "dense_joint_offset": hand_outputs["joint_offset"],
            "dense_joint_visibility_logits": hand_outputs["joint_visibility_logits"],
            "dense_joint_contact_logits": hand_outputs["joint_contact_logits"],
            "dense_joint_contact_log_distance": hand_outputs["joint_contact_log_distance"],
            "dense_joint_contact_probability": hand_outputs["joint_contact_probability"],
            "dense_joint_contact_log_conf": hand_outputs["joint_contact_log_conf"],
            "dense_joint_contact_conf": hand_outputs["joint_contact_conf"],
            "dense_joint_log_conf": hand_outputs["joint_log_conf"],
            "dense_joint_conf": hand_outputs["joint_conf"],
        }
        outputs.update(
            {
                "prompt_vertex_xyz": hand_outputs["vertex_xyz"].reshape(flat, hand_outputs["vertex_xyz"].shape[-2], 3),
                "prompt_vertex_offset": hand_outputs["vertex_offset"].reshape(flat, hand_outputs["vertex_offset"].shape[-2], 3),
                "prompt_vertex_visibility_logits": hand_outputs["vertex_visibility_logits"].reshape(flat, -1),
                "prompt_vertex_contact_logits": hand_outputs["vertex_contact_logits"].reshape(flat, -1),
                "prompt_vertex_contact_log_conf": hand_outputs["vertex_contact_log_conf"].reshape(flat, -1),
                "prompt_vertex_log_conf": hand_outputs["vertex_log_conf"].reshape(flat, -1),
                "prompt_joint_xyz": hand_outputs["joint_xyz"].reshape(flat, 21, 3),
                "prompt_joint_offset": hand_outputs["joint_offset"].reshape(flat, 21, 3),
                "prompt_joint_visibility_logits": hand_outputs["joint_visibility_logits"].reshape(flat, 21),
                "prompt_joint_contact_logits": hand_outputs["joint_contact_logits"].reshape(flat, 21),
                "prompt_joint_contact_log_conf": hand_outputs["joint_contact_log_conf"].reshape(flat, 21),
                "prompt_joint_log_conf": hand_outputs["joint_log_conf"].reshape(flat, 21),
                "prompt_hand_root_xyz": root_xyz.reshape(flat, 3),
                "prompt_hand_root_log_conf": root_log_conf.reshape(flat, 1),
            }
        )
        for source_name, public_name in (
            ("_root_solver_keypoint_uv", "root_solver_keypoint_uv"),
            ("_root_solver_keypoint_mask", "root_solver_keypoint_mask"),
            ("_root_uv_smoothing_applied", "root_uv_smoothing_applied"),
        ):
            if source_name in hand_outputs:
                outputs[public_name] = hand_outputs[source_name]
        return outputs

    def forward(
        self,
        images: torch.Tensor,
        *,
        multirate_frame_map: MultiRateFrameMap,
        temporal_interpolation_query_map: TemporalInterpolationQueryMap | None = None,
        hand_stage_gate_plan: HandStageGatePlan | None = None,
        root_solver_intrinsics: torch.Tensor | None = None,
        root_solver_intrinsics_valid: torch.Tensor | None = None,
        hand_h_root_smoothing: bool = False,
        root_uv_smoothing: bool = False,
    ) -> MultiRateMarkerOutput:
        if images.ndim != 5 or images.shape[2] != 3:
            raise ValueError("images must have shape [B, T, 3, H, W]")
        if type(hand_h_root_smoothing) is not bool:
            raise TypeError("hand_h_root_smoothing must be bool")
        if type(root_uv_smoothing) is not bool:
            raise TypeError("root_uv_smoothing must be bool")
        if hand_h_root_smoothing and self.training:
            raise RuntimeError("H-rate Root smoothing is inference-only; call model.eval() first")
        if root_uv_smoothing and self.training:
            raise RuntimeError("Root UV smoothing is inference-only; call model.eval() first")
        if root_solver_intrinsics is None and root_solver_intrinsics_valid is not None:
            raise ValueError(
                "root_solver_intrinsics_valid requires root_solver_intrinsics"
            )
        stage_plan = hand_stage_gate_plan or HandStageGatePlan()
        stage_plan.validate()
        if not isinstance(multirate_frame_map, MultiRateFrameMap):
            raise TypeError("multirate_frame_map must be a MultiRateFrameMap")
        multirate_frame_map.validate()
        if multirate_frame_map.high_frame_present.shape != images.shape[:2]:
            raise ValueError("multirate_frame_map high timeline must align with images")
        if multirate_frame_map.high_frame_present.device != images.device:
            raise ValueError("multirate_frame_map and images must be on the same device")
        if temporal_interpolation_query_map is not None:
            temporal_interpolation_query_map.validate(multirate_frame_map)
        return self._forward_prompt_injected_multirate(
            images,
            frame_map=multirate_frame_map,
            interpolation_query_map=temporal_interpolation_query_map,
            stage_plan=stage_plan,
            root_solver_intrinsics=root_solver_intrinsics,
            root_solver_intrinsics_valid=root_solver_intrinsics_valid,
            hand_h_root_smoothing=hand_h_root_smoothing,
            root_uv_smoothing=root_uv_smoothing,
        )


    def _decode_global_3r(
        self,
        *,
        images_high: torch.Tensor,
        frame_map: MultiRateFrameMap,
        prefix: Any,
        suffix: Any,
    ) -> Global3ROutput:
        batch_size = images_high.shape[0]
        global_frames = frame_map.global_frame_present.shape[1]
        if global_frames == 0:
            raise ValueError("a multi-rate forward requires at least one global anchor slot")
        images_global = gather_time_axis(
            images_high,
            frame_map.global_anchor_indices,
            frame_map.global_frame_present,
        )
        pose_encoding = self.backbone.decode_camera_pose_encoding(
            suffix.aggregated_tokens_list,
            patch_token_start=prefix.patch_token_start,
        )
        expected_pose_shape = (batch_size, global_frames, 9)
        if tuple(pose_encoding.shape) != expected_pose_shape:
            raise RuntimeError(
                "global CameraHead must return [B, T_G, 9], got "
                f"{tuple(pose_encoding.shape)} instead of {expected_pose_shape}"
            )
        camera_pose = self.backbone.decode_camera_from_pose_encoding(
            pose_encoding,
            image_size_hw=images_high.shape[-2:],
        )
        if tuple(camera_pose.shape) != (batch_size, global_frames, 4, 4):
            raise RuntimeError("global CameraHead must decode [B, T_G, 4, 4] W2C poses")
        intrinsics = self.backbone.build_intrinsics(
            image_size_hw=images_high.shape[-2:],
            device=images_high.device,
            dtype=images_high.dtype,
            pose_encoding=pose_encoding,
        )
        if tuple(intrinsics.shape) != (batch_size, global_frames, 3, 3):
            raise RuntimeError("global intrinsics must have shape [B, T_G, 3, 3]")
        depth, depth_conf = self.backbone.decode_depth_with_conf(
            suffix.aggregated_tokens_list,
            images=images_global,
            patch_token_start=prefix.patch_token_start,
        )
        if depth.shape[:2] != (batch_size, global_frames) or depth_conf.shape[:2] != (batch_size, global_frames):
            raise RuntimeError("global DenseHead must preserve the T_G axis")
        return Global3ROutput(
            aggregated_tokens_list=tuple(suffix.aggregated_tokens_list),
            camera_pose_encoding=pose_encoding.to(device=images_high.device, dtype=_geometry_dtype(images_high.dtype)),
            camera_pose=camera_pose.to(device=images_high.device, dtype=_geometry_dtype(images_high.dtype)),
            intrinsics=intrinsics.to(device=images_high.device, dtype=_geometry_dtype(images_high.dtype)),
            depth=depth.to(device=images_high.device, dtype=_geometry_dtype(images_high.dtype)),
            depth_conf=depth_conf.to(device=images_high.device, dtype=_geometry_dtype(images_high.dtype)),
        )

    def _format_multirate_high_rate_output(
        self,
        *,
        global_3r: Global3ROutput,
        high_rate_camera: Any,
        front: _FrontHalf,
        final_hand_prompts: torch.Tensor,
        final_metric_prompt: torch.Tensor,
        hand_outputs: dict[str, torch.Tensor],
        metric_outputs: dict[str, torch.Tensor],
        view_memory: torch.Tensor,
        root_solver_intrinsics_condition: torch.Tensor,
        root_solver_intrinsics_condition_valid: torch.Tensor,
        root_solver_intrinsics_is_external: bool,
    ) -> HighRateHandOutput:
        values = self._format_outputs(
            camera_pose=high_rate_camera.camera_pose,
            camera_pose_encoding=None,
            depth=global_3r.depth,
            depth_conf=global_3r.depth_conf,
            intrinsics=global_3r.intrinsics,
            front=front,
            final_hand_prompts=final_hand_prompts,
            final_metric_prompt=final_metric_prompt,
            hand_outputs=hand_outputs,
            metric_outputs=metric_outputs,
            view_memory=view_memory,
            root_solver_intrinsics_condition=root_solver_intrinsics_condition,
            root_solver_intrinsics_condition_valid=root_solver_intrinsics_condition_valid,
            root_solver_intrinsics_is_external=root_solver_intrinsics_is_external,
        )
        for key in ("camera_pose", "camera_pose_encoding", "depth", "depth_conf", "intrinsics"):
            values.pop(key, None)
        return HighRateHandOutput(values=values)

    def _canonical_interpolation_inputs(
        self,
        *,
        global_3r: Global3ROutput,
        hand_outputs: dict[str, torch.Tensor],
        metric_outputs: dict[str, torch.Tensor],
        frame_map: MultiRateFrameMap,
    ) -> tuple[
        torch.Tensor,
        dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        batch_size, _ = frame_map.high_frame_present.shape
        global_present = frame_map.global_frame_present.to(
            device=global_3r.depth.device,
            dtype=torch.bool,
        )

        pred_depth = global_3r.depth
        if pred_depth.ndim == 5 and pred_depth.shape[-1] == 1:
            pred_depth = pred_depth.squeeze(-1)
        if pred_depth.ndim != 4:
            raise ValueError("global depth must have shape [B, T_G, H, W]")
        scene_support = global_present.unsqueeze(-1).unsqueeze(-1)
        pred_scene_scale = scene_average_distance_from_depth(
            pred_depth.detach(),
            global_3r.intrinsics.detach(),
            global_3r.camera_pose.detach(),
            scene_support,
        )
        metric_value = metric_outputs["metric_value"].detach().to(
            device=pred_scene_scale.device,
            dtype=pred_scene_scale.dtype,
        )
        depth_support_valid = (
            scene_support
            & torch.isfinite(pred_depth)
            & (pred_depth > 0)
            & torch.isfinite(global_3r.intrinsics).all(dim=(-2, -1)).unsqueeze(-1).unsqueeze(-1)
            & torch.isfinite(global_3r.camera_pose).all(dim=(-2, -1)).unsqueeze(-1).unsqueeze(-1)
        ).flatten(1).any(dim=1)
        raw_factor = metric_value / pred_scene_scale.clamp_min(1e-6)
        scale_valid = (
            depth_support_valid
            & torch.isfinite(pred_scene_scale)
            & (pred_scene_scale > 0)
            & torch.isfinite(metric_value)
            & (metric_value > 0)
            & torch.isfinite(raw_factor)
            & (raw_factor > 0)
        )
        metric_factor = torch.where(scale_valid, raw_factor, torch.ones_like(raw_factor)).detach()

        predicted_camera_anchor_valid = global_present & scale_valid.unsqueeze(-1)
        camera_metric = clip_local_metric_w2c(
            global_3r.camera_pose,
            predicted_camera_anchor_valid,
            translation_scale=metric_factor,
        )
        predicted_anchor_valid = hand_outputs["root_translation_valid"].detach().to(
            device=global_3r.depth.device,
            dtype=torch.bool,
        ) & frame_map.high_frame_present.unsqueeze(-1)
        predicted_anchor_quality = hand_outputs["root_translation_anchor_quality"].detach().to(
            device=global_3r.depth.device,
            dtype=torch.float32,
        ).clamp(0.0, 1.0)
        hand_anchor_valid = predicted_anchor_valid
        hand_anchor_quality = torch.where(
            predicted_anchor_valid,
            predicted_anchor_quality,
            torch.zeros_like(predicted_anchor_quality),
        )

        diagnostics = {
            "interpolation_pred_scene_scale_global": pred_scene_scale.detach(),
            "interpolation_scene_metric_scale_factor": metric_factor,
            "interpolation_scene_metric_scale_valid": scale_valid.detach(),
            "interpolation_camera_anchor_valid_global": predicted_camera_anchor_valid,
            "interpolation_hand_anchor_valid_high": hand_anchor_valid,
            "interpolation_hand_anchor_quality_high": hand_anchor_quality,
        }
        return (
            camera_metric,
            dict(hand_outputs),
            predicted_camera_anchor_valid,
            hand_anchor_valid,
            hand_anchor_quality,
            diagnostics,
        )

    def _forward_prompt_injected_multirate(
        self,
        images_high: torch.Tensor,
        *,
        frame_map: MultiRateFrameMap,
        interpolation_query_map: TemporalInterpolationQueryMap | None,
        stage_plan: HandStageGatePlan,
        root_solver_intrinsics: torch.Tensor | None,
        root_solver_intrinsics_valid: torch.Tensor | None,
        hand_h_root_smoothing: bool,
        root_uv_smoothing: bool,
    ) -> MultiRateMarkerOutput:
        components = self._require_multirate_components()
        batch_size, high_frames = images_high.shape[:2]
        global_frames = frame_map.global_frame_present.shape[1]
        if global_frames == 0:
            raise ValueError("all-high-padding clips cannot run the frozen global aggregator")
        frame_present_high = frame_map.high_frame_present.to(device=images_high.device, dtype=torch.bool)
        global_present = frame_map.global_frame_present.to(device=images_high.device, dtype=torch.bool)

        dino_high = self.backbone.encode_dino_high_rate(images_high)
        if dino_high.feature_map.shape[:2] != (batch_size, high_frames) or dino_high.tokens.shape[:2] != (
            batch_size,
            high_frames,
        ):
            raise RuntimeError("high-rate frozen DINO output must preserve [B, T_H]")
        prefix = self.backbone.run_global_prefix_from_dino(
            dino_high,
            frame_map,
            inject_layer_idx=self.inject_layer_idx,
        )
        if prefix.public_tokens.shape[:2] != (batch_size, global_frames):
            raise RuntimeError("global frozen prefix must run on T_G anchors only")
        if prefix.mid_tokens.shape[:3] != (batch_size, global_frames, dino_high.tokens.shape[2]):
            raise RuntimeError("global prefix patch tokens must align with the frozen DINO patch grid")

        hand_global_tokens_high = components.hand_global_token_bridge(
            dino_high.tokens,
            prefix.mid_tokens,
            frame_map,
            global_token_valid=global_present,
            query_valid_high=frame_present_high,
        )
        expected_hand_context = (
            batch_size,
            high_frames,
            dino_high.tokens.shape[2],
            self.hand_adapter.agg_proj.in_features,
        )
        if tuple(hand_global_tokens_high.shape) != expected_hand_context:
            raise RuntimeError(
                "hand global-token bridge must return [B, T_H, N_patch, C_agg], got "
                f"{tuple(hand_global_tokens_high.shape)} instead of {expected_hand_context}"
            )

        front = self._front_half(
            images=images_high,
            dino_feature_map=dino_high.feature_map,
            dino_tokens=dino_high.tokens,
            mid_tokens=hand_global_tokens_high,
            frame_present_mask=frame_present_high,
            stage_plan=stage_plan,
        )

        hand_prompt_valid_high = frame_present_high[:, :, None, None].expand(
            -1,
            -1,
            front.hand_prompt_set.shape[2],
            front.hand_prompt_set.shape[3],
        )
        hand_prompt_global = components.hand_prompt_pooler(
            front.hand_prompt_set,
            frame_map,
            hand_prompt_valid_high=hand_prompt_valid_high,
        )
        metric_feature_global = components.metric_feature_pooler(
            front.metric_feature,
            frame_map,
            metric_feature_valid_high=frame_present_high,
        )
        metric_prompt_global = components.metric_prompt_pooler(
            front.adapter_outputs["scene_metric_prompt"].squeeze(2),
            frame_map,
            metric_feature_valid_high=frame_present_high,
        ).unsqueeze(2)

        prompt_run = PromptInjectedAggregatorRunner.append_prompts(
            prefix.public_tokens,
            hand_prompts=self.prompt_to_backbone(
                hand_prompt_global.reshape(batch_size, global_frames, 6, hand_prompt_global.shape[-1])
            ),
            metric_prompts=self.prompt_to_backbone(metric_prompt_global),
            patch_token_start=prefix.patch_token_start,
            patch_grid_hw=getattr(getattr(prefix, "layout", None), "patch_grid_hw", None),
        )
        suffix = self.backbone.run_suffix(
            prompt_run.internal_tokens,
            layout=prompt_run.layout,
            start_layer_idx=self.inject_layer_idx + 1,
            prefix_aggregated_tokens_list=prefix.aggregated_tokens_list,
        )
        final_hand_prompt_global = self.backbone_to_prompt(suffix.final_hand_prompts).reshape(
            batch_size,
            global_frames,
            2,
            3,
            -1,
        )
        final_metric_prompt_global = self.backbone_to_prompt(suffix.final_metric_prompts)
        global_3r = self._decode_global_3r(
            images_high=images_high,
            frame_map=frame_map,
            prefix=prefix,
            suffix=suffix,
        )
        if root_solver_intrinsics is None:
            intrinsics_condition_high, intrinsics_condition_valid_high = (
                self._multirate_intrinsics_condition(
                    global_3r.intrinsics,
                    frame_map,
                    image_height=images_high.shape[-2],
                    image_width=images_high.shape[-1],
                )
            )
            root_solver_intrinsics_is_external = False
        else:
            intrinsics_condition_high, intrinsics_condition_valid_high = (
                self._provided_root_intrinsics_condition(
                    root_solver_intrinsics,
                    root_solver_intrinsics_valid,
                    frame_present_high,
                    image_height=images_high.shape[-2],
                    image_width=images_high.shape[-1],
                )
            )
            root_solver_intrinsics_is_external = True

        final_hand_prompt_high = components.final_hand_prompt_bridge(
            front.hand_prompt_set,
            final_hand_prompt_global,
            frame_map,
            global_prompt_valid=global_present,
            query_valid_high=frame_present_high,
        )
        final_metric_prompt_high = components.final_metric_prompt_bridge(
            front.adapter_outputs["scene_metric_prompt"],
            final_metric_prompt_global,
            frame_map,
            global_prompt_valid=global_present,
            query_valid_high=frame_present_high,
        )
        view_memory = self._view_memory(
            front=front,
            stage_plan=stage_plan,
        )
        hand_outputs = self._decode_fixed_hand_outputs(
            final_hand_prompts=final_hand_prompt_high,
            final_metric_prompt=final_metric_prompt_high,
            front=front,
            view_memory=view_memory,
            intrinsics_condition_high=intrinsics_condition_high,
            intrinsics_condition_valid_high=intrinsics_condition_valid_high,
            image_size_hw=(int(images_high.shape[-2]), int(images_high.shape[-1])),
            root_uv_smoothing=root_uv_smoothing,
        )
        metric_outputs = self.metric_head.decode(
            final_metric_prompt_global,
            metric_feature=metric_feature_global,
            frame_present_mask=global_present,
        )
        (
            interpolation_camera_global,
            interpolation_hand_high,
            interpolation_camera_anchor_valid_global,
            interpolation_hand_anchor_valid_high,
            interpolation_hand_anchor_quality_high,
            interpolation_diagnostics,
        ) = self._canonical_interpolation_inputs(
            global_3r=global_3r,
            hand_outputs=hand_outputs,
            metric_outputs=metric_outputs,
            frame_map=frame_map,
        )
        root_smoothing = None
        if hand_h_root_smoothing:
            root_smoothing = smooth_high_rate_hand_root(
                interpolation_hand_high["joint_xyz"],
                interpolation_hand_high["vertex_xyz"],
                interpolation_hand_anchor_valid_high,
                interpolation_hand_anchor_quality_high,
            )
            interpolation_hand_high = dict(interpolation_hand_high)
            interpolation_hand_high["joint_xyz"] = root_smoothing.joint_xyz
            interpolation_hand_high["vertex_xyz"] = root_smoothing.vertex_xyz
        full_rate_interpolation = components.temporal_interpolator(
            camera_pose_global=interpolation_camera_global,
            camera_anchor_valid_global=interpolation_camera_anchor_valid_global,
            hand_outputs_high=interpolation_hand_high,
            hand_anchor_valid_high=interpolation_hand_anchor_valid_high,
            hand_anchor_quality_high=interpolation_hand_anchor_quality_high,
            frame_map=frame_map,
            query_map=interpolation_query_map,
        )
        high_rate_camera = full_rate_interpolation.high_rate_camera()
        high_rate = self._format_multirate_high_rate_output(
            global_3r=global_3r,
            high_rate_camera=high_rate_camera,
            front=front,
            final_hand_prompts=final_hand_prompt_high,
            final_metric_prompt=final_metric_prompt_high,
            hand_outputs=hand_outputs,
            metric_outputs=metric_outputs,
            view_memory=view_memory,
            root_solver_intrinsics_condition=intrinsics_condition_high,
            root_solver_intrinsics_condition_valid=intrinsics_condition_valid_high,
            root_solver_intrinsics_is_external=root_solver_intrinsics_is_external,
        )
        high_rate.values.update(interpolation_diagnostics)
        if root_smoothing is not None:
            high_rate.values.update(
                {
                    "hand_root_xyz_raw_high": root_smoothing.root_xyz_raw,
                    "hand_root_xyz_smoothed_high": root_smoothing.root_xyz_smoothed,
                    "hand_root_smoothing_delta_high": root_smoothing.root_delta,
                    "hand_root_smoothing_applied_high": root_smoothing.applied_mask,
                }
            )
        return MultiRateMarkerOutput(
            high_rate=high_rate,
            global_3r=global_3r,
            high_rate_camera=high_rate_camera,
            full_rate_interpolation=full_rate_interpolation,
            metric=MetricOutput(
                log_metric_value=metric_outputs["log_metric_value"],
                metric_value=metric_outputs["metric_value"],
            ),
        )
