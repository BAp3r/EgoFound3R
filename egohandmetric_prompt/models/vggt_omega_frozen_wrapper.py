from __future__ import annotations


from dataclasses import dataclass
import math
from pathlib import Path
import torch
from torch import nn
from egohandmetric_prompt.models.token_layout import TokenLayout
from egohandmetric_prompt.models.camera_temporal_refiner import _geometry_dtype
from egohandmetric_prompt.vendor import ensure_vendor_paths


_MAX_DEPTH_CONFIDENCE = 1e30


def _validate_depth_confidence_max_value(value: float) -> float:
    maximum = float(value)
    if not math.isfinite(maximum) or not 1.0 < maximum <= _MAX_DEPTH_CONFIDENCE:
        raise ValueError("depth_confidence_max_value must be finite and within (1, 1e30]")
    return maximum


def bounded_depth_confidence_logits(
    confidence_logits: torch.Tensor,
    *,
    max_confidence: float,
) -> torch.Tensor:
    if not torch.is_floating_point(confidence_logits):
        raise TypeError("confidence logits must be floating point")
    maximum = _validate_depth_confidence_max_value(max_confidence)
    max_logit = math.log(maximum - 1.0)
    dtype_max_logit = math.log(float(torch.finfo(confidence_logits.dtype).max) - 1.0)
    safe_max_logit = min(max_logit, dtype_max_logit)
    fallback_logit = min(0.0, safe_max_logit)
    return torch.nan_to_num(
        confidence_logits,
        nan=fallback_logit,
        posinf=safe_max_logit,
        neginf=torch.finfo(confidence_logits.dtype).min,
    ).clamp(max=safe_max_logit)


@dataclass(frozen=True, slots=True)
class _BoundDepthConfidenceProjectionHook:
    max_confidence: float

    def __call__(self, _module: nn.Module, _inputs: tuple[object, ...], output: object) -> torch.Tensor:
        if not isinstance(output, torch.Tensor):
            raise TypeError("DenseHead.proj_conf must return a tensor")
        return bounded_depth_confidence_logits(output, max_confidence=self.max_confidence)


def freeze_module(module: nn.Module) -> nn.Module:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad = False
    return module


@dataclass(slots=True)
class VggtOmegaModules:
    aggregator: nn.Module
    camera_head: nn.Module
    dense_head: nn.Module


@dataclass(frozen=True, slots=True)
class _VggtDinoVendorTokens:
    patch_tokens: torch.Tensor


@dataclass(frozen=True, slots=True)
class VggtDinoHighRateOutput:
    feature_map: torch.Tensor
    tokens: torch.Tensor
    vendor_tokens: object
    patch_grid_hw: tuple[int, int]


@dataclass(slots=True)
class VggtPrefixOutput:
    dino_feature_map: torch.Tensor
    dino_tokens: torch.Tensor
    mid_tokens: torch.Tensor
    public_tokens: torch.Tensor
    internal_tokens: torch.Tensor
    layout: TokenLayout
    patch_token_start: int
    aggregated_tokens_list: list[torch.Tensor | None]


@dataclass(slots=True)
class VggtSuffixOutput:
    aggregated_tokens_list: list[torch.Tensor | None]
    final_hand_prompts: torch.Tensor
    final_metric_prompts: torch.Tensor


class VggtOmegaFrozenWrapper(nn.Module):
    def __init__(
        self,
        modules: VggtOmegaModules,
        *,
        image_height: int,
        image_width: int,
        mid_layer_index: int = 11,
        depth_confidence_max_value: float = 64.0,
    ) -> None:
        super().__init__()
        self.aggregator = freeze_module(modules.aggregator)
        self.camera_head = freeze_module(modules.camera_head)
        self.dense_head = freeze_module(modules.dense_head)
        self.depth_confidence_max_value = _validate_depth_confidence_max_value(depth_confidence_max_value)
        self._depth_confidence_projection_hook: _BoundDepthConfidenceProjectionHook | None = None
        self._depth_confidence_projection_hook_handle = None
        projection = getattr(self.dense_head, "proj_conf", None)
        if projection is not None:
            if not isinstance(projection, nn.Module):
                raise TypeError("DenseHead.proj_conf must be an nn.Module")
            self._depth_confidence_projection_hook = _BoundDepthConfidenceProjectionHook(
                self.depth_confidence_max_value
            )
            self._depth_confidence_projection_hook_handle = projection.register_forward_hook(
                self._depth_confidence_projection_hook
            )
        self.image_height = image_height
        self.image_width = image_width
        aggregator_depth = getattr(self.aggregator, "depth", None)
        if aggregator_depth is not None and (mid_layer_index < 0 or mid_layer_index >= aggregator_depth):
            raise ValueError(
                f"mid_layer_index must lie in [0, {aggregator_depth - 1}], got {mid_layer_index}"
            )
        self.mid_layer_index = mid_layer_index

    @classmethod
    def from_vendor(
        cls,
        *,
        checkpoint_path: str | Path,
        image_height: int,
        image_width: int,
        mid_layer_index: int = 11,
        depth_confidence_max_value: float = 64.0,
    ) -> "VggtOmegaFrozenWrapper":
        ensure_vendor_paths()
        from vggt_omega.models.heads.camera_head import CameraHead
        from vggt_omega.models.heads.dense_head import DenseHead
        from vggt_omega.models.aggregator import Aggregator

        modules = VggtOmegaModules(
            aggregator=Aggregator(),
            camera_head=CameraHead(),
            dense_head=DenseHead(),
        )
        wrapper = cls(
            modules,
            image_height=image_height,
            image_width=image_width,
            mid_layer_index=mid_layer_index,
            depth_confidence_max_value=depth_confidence_max_value,
        )
        state_dict = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=True)
        missing, unexpected = wrapper.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"VGGT-Omega checkpoint load is incomplete: missing={missing[:10]} unexpected={unexpected[:10]}"
            )
        return wrapper

    def train(self, mode: bool = True):
        super().train(mode)
        self.aggregator.eval()
        self.camera_head.eval()
        self.dense_head.eval()
        return self

    def _select_mid_layer_tokens(self, aggregated_tokens_list: list[torch.Tensor | None]) -> torch.Tensor:
        if 0 <= self.mid_layer_index < len(aggregated_tokens_list):
            selected = aggregated_tokens_list[self.mid_layer_index]
            if selected is not None:
                return selected
        for tokens in reversed(aggregated_tokens_list):
            if tokens is not None:
                return tokens
        raise ValueError("the aggregator returned no cached layer tokens")

    def _cached_tokens_device_type(self, aggregated_tokens_list: list[torch.Tensor | None]) -> str:
        for tokens in aggregated_tokens_list:
            if tokens is not None:
                return tokens.device.type
        raise ValueError("head decoding needs at least one cached layer token")

    @staticmethod
    def _restore_frozen_head_fp32(module: nn.Module) -> None:
        parameter = next(module.parameters(), None)
        if parameter is not None and parameter.dtype != torch.float32:
            module.float()

    def _prepare_patch_tokens(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_frames, channels, height, width = images.shape
        normalized = (images - self.aggregator._resnet_mean) / self.aggregator._resnet_std
        patch_tokens = self.aggregator.patch_embed(normalized.view(batch_size * num_frames, channels, height, width))
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]
        patch_count, dino_dim = patch_tokens.shape[1:]
        patch_height = height // self.aggregator.patch_size
        patch_width = width // self.aggregator.patch_size
        dino_tokens = patch_tokens.view(batch_size, num_frames, patch_count, dino_dim)
        dino_feature_map = dino_tokens.transpose(-1, -2).reshape(
            batch_size,
            num_frames,
            dino_dim,
            patch_height,
            patch_width,
        )
        return patch_tokens, dino_tokens, dino_feature_map

    def _initial_public_tokens(self, patch_tokens: torch.Tensor, *, batch_size: int, num_frames: int) -> torch.Tensor:
        embed_dim = patch_tokens.shape[-1]
        if hasattr(self.aggregator, "camera_token") and hasattr(self.aggregator, "register_token"):
            from vggt_omega.models.aggregator import slice_expand_and_flatten

            camera_token = slice_expand_and_flatten(self.aggregator.camera_token, batch_size, num_frames)
            register_token = slice_expand_and_flatten(self.aggregator.register_token, batch_size, num_frames)
        else:
            camera_token = patch_tokens.new_zeros(batch_size * num_frames, 1, embed_dim)
            register_token = patch_tokens.new_zeros(
                batch_size * num_frames,
                self.aggregator.patch_token_start - 1,
                embed_dim,
            )
        public_tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)
        return public_tokens.view(batch_size, num_frames, public_tokens.shape[1], embed_dim)

    def _rope(self, *, patch_height: int, patch_width: int):
        if not hasattr(self.aggregator, "rope_embed"):
            return None
        with torch.no_grad():
            rope_sin, rope_cos = self.aggregator.rope_embed(H=patch_height, W=patch_width)
        return (
            rope_sin.to(dtype=torch.float32),
            rope_cos.to(dtype=torch.float32),
        )

    def _cache_layer_output(
        self,
        outputs: list[torch.Tensor | None],
        *,
        block_idx: int,
        frame_tokens: torch.Tensor,
        tokens: torch.Tensor,
        layout: TokenLayout,
    ) -> None:
        if block_idx not in self.aggregator.cached_layer_indices:
            return
        public_frame_tokens = layout.internal_to_public(frame_tokens)
        public_tokens = layout.internal_to_public(tokens)
        stripped_frame = public_frame_tokens[:, :, : layout.patch_token_end]
        stripped_tokens = public_tokens[:, :, : layout.patch_token_end]
        outputs[block_idx] = torch.cat([stripped_frame, stripped_tokens], dim=-1)

    def run_prefix(self, images: torch.Tensor, *, inject_layer_idx: int) -> VggtPrefixOutput:
        with torch.no_grad():
            return self._run_prefix(images, inject_layer_idx=inject_layer_idx)

    def encode_dino_high_rate(self, images_high: torch.Tensor) -> VggtDinoHighRateOutput:
        if images_high.ndim != 5:
            raise ValueError(
                "High-rate DINO input must have shape [batch, frames, channels, height, width], "
                f"got {tuple(images_high.shape)}."
            )
        batch_size, num_frames, channels, height, width = images_high.shape
        if channels != 3:
            raise ValueError(f"Expected RGB high-rate input, got {channels} channels.")
        if batch_size <= 0 or num_frames <= 0:
            raise ValueError("High-rate DINO input must contain at least one batch item and one frame.")
        with torch.no_grad():
            _, dino_tokens, dino_feature_map = self._prepare_patch_tokens(images_high)
        return VggtDinoHighRateOutput(
            feature_map=dino_feature_map,
            tokens=dino_tokens,
            vendor_tokens=_VggtDinoVendorTokens(patch_tokens=dino_tokens),
            patch_grid_hw=(int(dino_feature_map.shape[-2]), int(dino_feature_map.shape[-1])),
        )

    def _global_anchor_selection(
        self,
        frame_map: object,
        *,
        batch_size: int,
        high_frame_count: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        indices = getattr(frame_map, "global_anchor_indices", None)
        valid = getattr(frame_map, "global_frame_present", None)
        if not isinstance(indices, torch.Tensor) or not isinstance(valid, torch.Tensor):
            raise TypeError(
                "frame_map must expose Tensor global_anchor_indices and global_frame_present fields."
            )
        if indices.dtype != torch.long:
            raise TypeError(f"global_anchor_indices must be int64, got {indices.dtype}.")
        if valid.dtype != torch.bool:
            raise TypeError(f"global_frame_present must be bool, got {valid.dtype}.")
        if indices.ndim != 2 or valid.ndim != 2 or indices.shape != valid.shape:
            raise ValueError(
                "global_anchor_indices and global_frame_present must both have shape [batch, global_frames]."
            )
        if indices.shape[0] != batch_size:
            raise ValueError(
                f"frame_map batch size {indices.shape[0]} does not match DINO batch size {batch_size}."
            )
        if high_frame_count <= 0:
            raise ValueError("Cannot gather global anchors from an empty high-rate sequence.")
        if indices.shape[1] == 0:
            raise ValueError("Frozen global prefix requires at least one padded global frame slot.")
        return indices.detach().to(device=device), valid.detach().to(device=device)

    def _gather_global_time_axis(self, value: torch.Tensor, frame_map: object) -> torch.Tensor:
        if value.ndim < 2:
            raise ValueError(f"Expected a [batch, time, ...] tensor, got {tuple(value.shape)}.")
        indices, valid = self._global_anchor_selection(
            frame_map,
            batch_size=value.shape[0],
            high_frame_count=value.shape[1],
            device=value.device,
        )
        from egohandmetric_prompt.multirate_contracts import gather_time_axis

        return gather_time_axis(value, indices, valid)

    def _run_prefix_from_dino_tokens(
        self,
        dino_tokens: torch.Tensor,
        *,
        patch_grid_hw: tuple[int, int],
        inject_layer_idx: int,
    ) -> VggtPrefixOutput:
        if dino_tokens.ndim != 4:
            raise ValueError(
                "DINO tokens must have shape [batch, frames, patches, channels], "
                f"got {tuple(dino_tokens.shape)}."
            )
        batch_size, num_frames, patch_count, dino_dim = dino_tokens.shape
        patch_height, patch_width = int(patch_grid_hw[0]), int(patch_grid_hw[1])
        if patch_height <= 0 or patch_width <= 0 or patch_count != patch_height * patch_width:
            raise ValueError(
                f"DINO patch count/grid mismatch: {patch_count} versus {patch_height}x{patch_width}."
            )
        patch_tokens = dino_tokens.reshape(batch_size * num_frames, patch_count, dino_dim)
        dino_feature_map = dino_tokens.transpose(-1, -2).reshape(
            batch_size,
            num_frames,
            dino_dim,
            patch_height,
            patch_width,
        )
        layout = TokenLayout(
            camera_token_count=1,
            register_token_count=self.aggregator.patch_token_start - 1,
            patch_token_count=patch_count,
            hand_prompt_count=0,
            patch_grid_hw=(patch_height, patch_width),
        )
        public_tokens = self._initial_public_tokens(patch_tokens, batch_size=batch_size, num_frames=num_frames)
        internal_tokens = public_tokens
        num_tokens = internal_tokens.shape[2]
        embed_dim = internal_tokens.shape[3]
        rope = self._rope(patch_height=patch_height, patch_width=patch_width)
        outputs: list[torch.Tensor | None] = [None] * self.aggregator.depth
        tokens = internal_tokens
        for block_idx in range(inject_layer_idx + 1):
            tokens_flat = tokens.view(batch_size * num_frames, num_tokens, embed_dim)
            tokens_flat, frame_tokens = self.aggregator._run_frame_block(
                tokens_flat,
                batch_size,
                num_frames,
                num_tokens,
                embed_dim,
                block_idx,
                rope,
            )
            tokens = self.aggregator._run_inter_frame_attention_block(
                tokens_flat,
                batch_size,
                num_frames,
                num_tokens,
                embed_dim,
                block_idx,
                self.aggregator.inter_frame_attention_types[block_idx]
                if hasattr(self.aggregator, "inter_frame_attention_types")
                else "global",
            )
            if tokens.ndim == 3:
                tokens = tokens.view(batch_size, num_frames, num_tokens, embed_dim)
            self._cache_layer_output(
                outputs,
                block_idx=block_idx,
                frame_tokens=frame_tokens,
                tokens=tokens,
                layout=layout,
            )
        current_public_tokens = layout.internal_to_public(tokens)
        mid_layer_tokens = self._select_mid_layer_tokens(outputs)
        mid_tokens = mid_layer_tokens[:, :, self.aggregator.patch_token_start :]
        if mid_tokens.shape[2] != patch_count:
            raise RuntimeError(
                "cached aggregator patch tokens do not match the prefix patch grid: "
                f"{mid_tokens.shape[2]} versus {patch_height}x{patch_width}"
            )
        return VggtPrefixOutput(
            dino_feature_map=dino_feature_map,
            dino_tokens=dino_tokens,
            mid_tokens=mid_tokens,
            public_tokens=current_public_tokens,
            internal_tokens=tokens,
            layout=layout,
            patch_token_start=self.aggregator.patch_token_start,
            aggregated_tokens_list=outputs,
        )

    def run_global_prefix_from_dino(
        self,
        dino_high: VggtDinoHighRateOutput,
        frame_map: object,
        *,
        inject_layer_idx: int,
    ) -> VggtPrefixOutput:
        if not isinstance(dino_high, VggtDinoHighRateOutput):
            raise TypeError("dino_high must be a VggtDinoHighRateOutput.")
        if not isinstance(dino_high.vendor_tokens, _VggtDinoVendorTokens):
            raise TypeError("dino_high.vendor_tokens was not created by this wrapper.")
        if dino_high.vendor_tokens.patch_tokens is not dino_high.tokens:
            raise ValueError(
                "DINO public tokens and wrapper-private token state must refer to the same tensor."
            )
        if dino_high.vendor_tokens.patch_tokens.shape != dino_high.tokens.shape:
            raise ValueError("DINO public tokens and wrapper-private token state have different shapes.")
        with torch.no_grad():
            global_patch_tokens = self._gather_global_time_axis(dino_high.tokens, frame_map)
            return self._run_prefix_from_dino_tokens(
                global_patch_tokens,
                patch_grid_hw=dino_high.patch_grid_hw,
                inject_layer_idx=inject_layer_idx,
            )


    def _run_prefix(self, images: torch.Tensor, *, inject_layer_idx: int) -> VggtPrefixOutput:
        batch_size, num_frames, channels, height, width = images.shape
        if channels != 3:
            raise ValueError(f"expected 3-channel RGB input, got {channels}")
        patch_tokens, dino_tokens, dino_feature_map = self._prepare_patch_tokens(images)
        patch_height = dino_feature_map.shape[-2]
        patch_width = dino_feature_map.shape[-1]
        layout = TokenLayout(
            camera_token_count=1,
            register_token_count=self.aggregator.patch_token_start - 1,
            patch_token_count=patch_tokens.shape[1],
            hand_prompt_count=0,
            patch_grid_hw=(int(patch_height), int(patch_width)),
        )
        public_tokens = self._initial_public_tokens(patch_tokens, batch_size=batch_size, num_frames=num_frames)
        internal_tokens = public_tokens
        num_tokens = internal_tokens.shape[2]
        embed_dim = internal_tokens.shape[3]
        rope = self._rope(patch_height=patch_height, patch_width=patch_width)
        outputs: list[torch.Tensor | None] = [None] * self.aggregator.depth
        tokens = internal_tokens
        for block_idx in range(inject_layer_idx + 1):
            tokens_flat = tokens.view(batch_size * num_frames, num_tokens, embed_dim)
            tokens_flat, frame_tokens = self.aggregator._run_frame_block(
                tokens_flat,
                batch_size,
                num_frames,
                num_tokens,
                embed_dim,
                block_idx,
                rope,
            )
            tokens = self.aggregator._run_inter_frame_attention_block(
                tokens_flat,
                batch_size,
                num_frames,
                num_tokens,
                embed_dim,
                block_idx,
                self.aggregator.inter_frame_attention_types[block_idx] if hasattr(self.aggregator, "inter_frame_attention_types") else "global",
            )
            if tokens.ndim == 3:
                tokens = tokens.view(batch_size, num_frames, num_tokens, embed_dim)
            self._cache_layer_output(outputs, block_idx=block_idx, frame_tokens=frame_tokens, tokens=tokens, layout=layout)
        current_public_tokens = layout.internal_to_public(tokens)
        mid_layer_tokens = self._select_mid_layer_tokens(outputs)
        mid_tokens = mid_layer_tokens[:, :, self.aggregator.patch_token_start :]
        patch_count = patch_height * patch_width
        if mid_tokens.shape[2] != patch_count:
            raise RuntimeError(
                "cached aggregator patch tokens do not match the prefix patch grid: "
                f"{mid_tokens.shape[2]} versus {patch_height}x{patch_width}"
            )
        return VggtPrefixOutput(
            dino_feature_map=dino_feature_map,
            dino_tokens=dino_tokens,
            mid_tokens=mid_tokens,
            public_tokens=current_public_tokens,
            internal_tokens=tokens,
            layout=layout,
            patch_token_start=self.aggregator.patch_token_start,
            aggregated_tokens_list=outputs,
        )

    def _run_suffix_block(
        self,
        internal_tokens: torch.Tensor,
        *,
        layout: TokenLayout,
        block_idx: int,
        rope,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_frames, num_tokens, embed_dim = internal_tokens.shape
        original_patch_token_start = getattr(self.aggregator, "patch_token_start", layout.camera_register_end)
        setattr(self.aggregator, "patch_token_start", layout.internal_patch_start)
        try:
            tokens_flat = internal_tokens.view(batch_size * num_frames, num_tokens, embed_dim)
            tokens_flat, frame_tokens = self.aggregator._run_frame_block(
                tokens_flat,
                batch_size,
                num_frames,
                num_tokens,
                embed_dim,
                block_idx,
                rope,
            )
            tokens = self.aggregator._run_inter_frame_attention_block(
                tokens_flat,
                batch_size,
                num_frames,
                num_tokens,
                embed_dim,
                block_idx,
                self.aggregator.inter_frame_attention_types[block_idx] if hasattr(self.aggregator, "inter_frame_attention_types") else "global",
            )
            if tokens.ndim == 3:
                tokens = tokens.view(batch_size, num_frames, num_tokens, embed_dim)
            return tokens, frame_tokens
        finally:
            setattr(self.aggregator, "patch_token_start", original_patch_token_start)

    def run_suffix(
        self,
        internal_tokens: torch.Tensor,
        *,
        layout: TokenLayout,
        start_layer_idx: int,
        prefix_aggregated_tokens_list: list[torch.Tensor | None],
    ) -> VggtSuffixOutput:
        if layout.patch_grid_hw is not None:
            patch_height, patch_width = layout.patch_grid_hw
        else:
            patch_height = int(layout.patch_token_count**0.5)
            patch_width = layout.patch_token_count // max(patch_height, 1)
        rope = self._rope(patch_height=patch_height, patch_width=patch_width)
        outputs = list(prefix_aggregated_tokens_list)
        tokens = internal_tokens
        for block_idx in range(start_layer_idx, self.aggregator.depth):
            tokens, frame_tokens = self._run_suffix_block(
                tokens,
                layout=layout,
                block_idx=block_idx,
                rope=rope,
            )
            self._cache_layer_output(outputs, block_idx=block_idx, frame_tokens=frame_tokens, tokens=tokens, layout=layout)
        final_hand_prompts, final_metric_prompts = layout.split_internal_prompt_types(tokens)
        return VggtSuffixOutput(
            aggregated_tokens_list=outputs,
            final_hand_prompts=final_hand_prompts,
            final_metric_prompts=final_metric_prompts,
        )

    def decode_camera_pose_encoding(
        self,
        aggregated_tokens_list: list[torch.Tensor | None],
        *,
        patch_token_start: int,
    ) -> torch.Tensor:
        device_type = self._cached_tokens_device_type(aggregated_tokens_list)
        self._restore_frozen_head_fp32(self.camera_head)
        with torch.autocast(device_type=device_type, enabled=False):
            return self.camera_head(aggregated_tokens_list, patch_token_start=patch_token_start)

    def decode_camera_from_pose_encoding(
        self,
        pose_encoding: torch.Tensor,
        *,
        image_size_hw: tuple[int, int],
    ) -> torch.Tensor:
        ensure_vendor_paths()
        from vggt_omega.utils.pose_enc import encoding_to_camera

        pose_encoding = pose_encoding.to(dtype=_geometry_dtype(pose_encoding.dtype))
        with torch.autocast(device_type=pose_encoding.device.type, enabled=False):
            extrinsics, _ = encoding_to_camera(pose_encoding, image_size_hw, build_intrinsics=True)
        batch_size, num_frames = pose_encoding.shape[:2]
        camera_pose = torch.zeros(batch_size, num_frames, 4, 4, device=pose_encoding.device, dtype=pose_encoding.dtype)
        camera_pose[..., :3, :4] = extrinsics
        camera_pose[..., 3, 3] = 1.0
        return camera_pose

    def build_intrinsics(
        self,
        *,
        image_size_hw: tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
        pose_encoding: torch.Tensor | None = None,
        aggregated_tokens_list: list[torch.Tensor | None] | None = None,
        patch_token_start: int | None = None,
    ) -> torch.Tensor:
        ensure_vendor_paths()
        from vggt_omega.utils.pose_enc import encoding_to_camera

        if pose_encoding is None:
            if aggregated_tokens_list is None or patch_token_start is None:
                raise ValueError("building intrinsics needs either pose_encoding or aggregated_tokens_list plus patch_token_start")
            pose_encoding = self.decode_camera_pose_encoding(
                aggregated_tokens_list,
                patch_token_start=patch_token_start,
            )
        pose_encoding = pose_encoding.to(dtype=_geometry_dtype(pose_encoding.dtype))
        with torch.autocast(device_type=pose_encoding.device.type, enabled=False):
            _, intrinsics = encoding_to_camera(pose_encoding, image_size_hw, build_intrinsics=True)
        return intrinsics.to(device=device, dtype=_geometry_dtype(dtype))

    def decode_depth_with_conf(
        self,
        aggregated_tokens_list: list[torch.Tensor | None],
        *,
        images: torch.Tensor,
        patch_token_start: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._restore_frozen_head_fp32(self.dense_head)
        with torch.autocast(device_type=images.device.type, enabled=False):
            depth, depth_conf = self.dense_head(aggregated_tokens_list, images=images, patch_token_start=patch_token_start)
        if depth.ndim == 5 and depth.shape[-1] == 1:
            depth = depth.squeeze(-1)
        if depth_conf.ndim == 5 and depth_conf.shape[-1] == 1:
            depth_conf = depth_conf.squeeze(-1)
        return depth, depth_conf


    def forward(self, images: torch.Tensor):
        from egohandmetric_prompt.models.marker_model import MarkerBackboneFeatures

        prefix = self.run_prefix(images, inject_layer_idx=self.mid_layer_index)
        pose_encoding = self.decode_camera_pose_encoding(
            prefix.aggregated_tokens_list,
            patch_token_start=prefix.patch_token_start,
        )
        camera_pose = self.decode_camera_from_pose_encoding(
            pose_encoding,
            image_size_hw=images.shape[-2:],
        ).to(device=images.device, dtype=_geometry_dtype(images.dtype))
        intrinsics = self.build_intrinsics(
            image_size_hw=images.shape[-2:],
            device=images.device,
            dtype=images.dtype,
            pose_encoding=pose_encoding,
        )
        depth, depth_conf = self.decode_depth_with_conf(
            prefix.aggregated_tokens_list,
            images=images,
            patch_token_start=prefix.patch_token_start,
        )
        depth = depth.to(device=images.device, dtype=_geometry_dtype(images.dtype))
        depth_conf = depth_conf.to(device=images.device, dtype=_geometry_dtype(images.dtype))
        return MarkerBackboneFeatures(
            dino_feature_map=prefix.dino_feature_map.to(device=images.device, dtype=images.dtype),
            dino_tokens=prefix.dino_tokens.to(device=images.device, dtype=images.dtype),
            mid_tokens=prefix.mid_tokens.to(device=images.device, dtype=images.dtype),
            camera_pose=camera_pose,
            depth=depth,
            intrinsics=intrinsics,
            camera_pose_encoding=pose_encoding.to(device=images.device, dtype=_geometry_dtype(images.dtype)),
            depth_conf=depth_conf,
        )
