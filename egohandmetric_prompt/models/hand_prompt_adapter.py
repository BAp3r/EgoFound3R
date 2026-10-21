from __future__ import annotations


import torch
from torch import nn
from torch.nn import functional as F
from .transformer_blocks import TransformerDecoderBlock


KEYPOINT_GEOMETRY_DIM = 19


def normalized_keypoint_geometry(
    keypoint_uv: torch.Tensor,
    keypoint_conf: torch.Tensor,
    keypoint_mask: torch.Tensor,
) -> torch.Tensor:
    if keypoint_uv.shape[:-1] != keypoint_conf.shape or keypoint_uv.shape[-1] != 2:
        raise ValueError("keypoint UV and confidence shapes are incompatible")
    if keypoint_mask.shape != keypoint_conf.shape:
        raise ValueError("keypoint mask must align with confidence")
    mask = keypoint_mask.to(device=keypoint_uv.device, dtype=torch.bool)
    safe_uv = torch.where(mask.unsqueeze(-1), keypoint_uv, torch.zeros_like(keypoint_uv))
    safe_conf = torch.where(mask, keypoint_conf, torch.zeros_like(keypoint_conf))
    safe_uv = torch.nan_to_num(safe_uv, nan=0.0, posinf=0.0, neginf=0.0)
    safe_conf = torch.nan_to_num(safe_conf, nan=0.0, posinf=0.0, neginf=0.0)
    xy = 2.0 * safe_uv - 1.0
    parts = [xy[..., 0:1], xy[..., 1:2], safe_conf.unsqueeze(-1)]
    for frequency in (1.0, 2.0, 4.0, 8.0):
        scaled = torch.pi * frequency * xy
        parts.extend(
            (
                scaled[..., 0:1].sin(),
                scaled[..., 0:1].cos(),
                scaled[..., 1:2].sin(),
                scaled[..., 1:2].cos(),
            )
        )
    geometry = torch.cat(parts, dim=-1)
    return torch.where(mask.unsqueeze(-1), geometry, torch.zeros_like(geometry))


class KeypointContextProjector(nn.Module):
    def __init__(self, *, hand_feature_dim: int) -> None:
        super().__init__()
        self.hand_feature_dim = int(hand_feature_dim)
        self.root_attention_heads = next(
            heads
            for heads in range(min(8, self.hand_feature_dim), 0, -1)
            if self.hand_feature_dim % heads == 0
        )
        self.project = nn.Sequential(
            nn.LayerNorm(hand_feature_dim),
            nn.Linear(hand_feature_dim, hand_feature_dim),
        )
        self.missing_keypoint_context = nn.Parameter(torch.randn(1, 1, 2, hand_feature_dim) * 0.02)
        self.missing_joint_context = nn.Parameter(torch.randn(1, 1, 2, hand_feature_dim) * 0.02)
        self.missing_marker_context = nn.Parameter(torch.randn(1, 1, 2, hand_feature_dim) * 0.02)


    def _semantic_attention_pool(
        self,
        keypoint_tokens: torch.Tensor,
        keypoint_mask: torch.Tensor,
        keypoint_conf: torch.Tensor,
        semantic_queries: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expected_query_shape = (*keypoint_tokens.shape[:3], 3, self.hand_feature_dim)
        if semantic_queries.shape != expected_query_shape:
            raise ValueError(f"semantic_queries must have shape {expected_query_shape}")
        if keypoint_conf.shape != keypoint_mask.shape:
            raise ValueError("keypoint_conf must align with keypoint_mask")

        head_count = self.root_attention_heads
        head_dim = self.hand_feature_dim // head_count
        safe_tokens = torch.where(
            keypoint_mask.unsqueeze(-1),
            keypoint_tokens,
            torch.zeros_like(keypoint_tokens),
        )
        query_heads = F.layer_norm(semantic_queries, (self.hand_feature_dim,)).reshape(
            *semantic_queries.shape[:-1], head_count, head_dim
        )
        token_heads = F.layer_norm(safe_tokens, (self.hand_feature_dim,)).reshape(
            *safe_tokens.shape[:-1], head_count, head_dim
        )
        value_heads = safe_tokens.reshape(
            *safe_tokens.shape[:-1], head_count, head_dim
        )
        logits = torch.einsum("...thd,...jhd->...thj", query_heads, token_heads)
        logits = logits * (head_dim ** -0.5)
        confidence_prior = torch.log(
            keypoint_conf.to(dtype=logits.dtype).clamp_min(torch.finfo(logits.dtype).eps)
        )
        logits = logits + confidence_prior.unsqueeze(-2).unsqueeze(-2)
        valid = keypoint_mask.unsqueeze(-2).unsqueeze(-2)
        masked_logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        weights = torch.softmax(masked_logits.float(), dim=-1)
        weights = weights * valid.to(dtype=weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(weights.dtype).eps
        )
        pooled_heads = torch.einsum(
            "...thj,...jhd->...thd",
            weights.to(dtype=value_heads.dtype),
            value_heads,
        )
        pooled = pooled_heads.reshape(*semantic_queries.shape[:-1], self.hand_feature_dim)
        return pooled, weights.mean(dim=-2).to(dtype=keypoint_tokens.dtype)

    def forward(
        self,
        keypoint_tokens: torch.Tensor,
        keypoint_mask: torch.Tensor,
        *,
        keypoint_conf: torch.Tensor,
        semantic_queries: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if keypoint_tokens.ndim != 5 or keypoint_tokens.shape[-1] != self.hand_feature_dim:
            raise ValueError("keypoint_tokens must have shape [B, T, 2, J, C_hand]")
        if keypoint_mask.shape != keypoint_tokens.shape[:-1]:
            raise ValueError("keypoint_mask must align with keypoint_tokens")
        mask = keypoint_mask.to(device=keypoint_tokens.device, dtype=torch.bool)
        has_observed = mask.any(dim=-1, keepdim=True)
        pooled_context, semantic_weights = self._semantic_attention_pool(
            keypoint_tokens,
            mask,
            keypoint_conf,
            semantic_queries,
        )
        missing_context = torch.stack(
            (
                self.missing_keypoint_context,
                self.missing_joint_context,
                self.missing_marker_context,
            ),
            dim=3,
        ).to(device=keypoint_tokens.device, dtype=keypoint_tokens.dtype)
        missing_context = missing_context.expand(
            keypoint_tokens.shape[0], keypoint_tokens.shape[1], -1, -1, -1
        )

        wrist_valid = mask[..., 0].unsqueeze(-1)
        root_pooled = pooled_context[..., 0, :]
        pooled_context = pooled_context.clone()
        pooled_context[..., 0, :] = torch.where(
            wrist_valid,
            keypoint_tokens[..., 0, :],
            root_pooled,
        )
        semantic_context = torch.where(has_observed.unsqueeze(-1), pooled_context, missing_context)
        projected_context = self.project(semantic_context)
        root_weights = semantic_weights[..., 0, :]
        return (
            projected_context[..., 1, :],
            projected_context,
            root_weights,
            wrist_valid.squeeze(-1),
            has_observed.squeeze(-1),
        )


class HandPromptAdapter(nn.Module):
    prompt_type_count = 3

    def __init__(
        self,
        dino_dim: int,
        agg_dim: int,
        prompt_dim: int,
        hand_feature_dim: int | None = None,
        num_layers: int = 6,
        num_heads: int = 12,
        mlp_ratio: int = 4,
        joint_count: int = 21,
    ) -> None:
        super().__init__()
        self.prompt_dim = int(prompt_dim)
        self.joint_count = int(joint_count)
        self.hand_feature_dim = int(hand_feature_dim) if hand_feature_dim is not None else self.prompt_dim * 2
        self.dino_proj = nn.Linear(dino_dim, self.hand_feature_dim)
        self.agg_proj = nn.Linear(agg_dim, self.hand_feature_dim)
        self.token_fuse = nn.Linear(self.hand_feature_dim * 2, self.hand_feature_dim)
        self.side_type_queries = nn.Parameter(
            torch.randn(1, 1, 2, self.prompt_type_count, self.hand_feature_dim) * 0.02
        )
        self.query_norm = nn.LayerNorm(self.hand_feature_dim)
        self.blocks = nn.ModuleList(
            [TransformerDecoderBlock(self.hand_feature_dim, num_heads, mlp_ratio) for _ in range(num_layers)]
        )
        self.output_norm = nn.LayerNorm(self.hand_feature_dim)
        self.keypoint_feature_proj = nn.Linear(dino_dim, self.hand_feature_dim)
        self.keypoint_coord_proj = nn.Linear(KEYPOINT_GEOMETRY_DIM, self.hand_feature_dim)
        self.keypoint_type_embed = nn.Parameter(
            torch.randn(1, 1, 1, self.joint_count, self.hand_feature_dim) * 0.02
        )
        self.missing_keypoint_token = nn.Parameter(
            torch.randn(1, 1, 1, self.joint_count, self.hand_feature_dim) * 0.02
        )
        self.keypoint_context_projector = KeypointContextProjector(hand_feature_dim=self.hand_feature_dim)
        self.early_loc_gate = nn.Parameter(
            torch.full((1, 1, 2, self.prompt_type_count, 1), 0.5)
        )
        self.loc_context_proj = nn.Sequential(
            nn.LayerNorm(self.hand_feature_dim),
            nn.Linear(self.hand_feature_dim, self.hand_feature_dim),
        )
        self.prompt_projector = nn.Sequential(
            nn.LayerNorm(self.hand_feature_dim),
            nn.Linear(self.hand_feature_dim, self.prompt_dim),
        )
        self.keypoint_prompt_projector = nn.Sequential(
            nn.LayerNorm(self.hand_feature_dim),
            nn.Linear(self.hand_feature_dim, self.prompt_dim),
        )
        self.scene_metric_feature_projector = nn.Sequential(
            nn.LayerNorm(agg_dim),
            nn.Linear(agg_dim, self.hand_feature_dim),
        )
        self.scene_metric_prompt_projector = nn.Sequential(
            nn.LayerNorm(self.hand_feature_dim),
            nn.Linear(self.hand_feature_dim, self.prompt_dim),
        )

    def _keypoint_tokens(
        self,
        *,
        keypoint_features: torch.Tensor,
        keypoint_uv: torch.Tensor,
        keypoint_conf: torch.Tensor,
        keypoint_mask: torch.Tensor,
    ) -> torch.Tensor:
        if keypoint_features.shape[-2:] != (self.joint_count, keypoint_features.shape[-1]):
            raise ValueError("keypoint_features must have a 21-joint axis")
        coord_input = normalized_keypoint_geometry(keypoint_uv, keypoint_conf, keypoint_mask)
        safe_features = torch.where(
            keypoint_mask.unsqueeze(-1),
            torch.nan_to_num(keypoint_features, nan=0.0, posinf=0.0, neginf=0.0),
            torch.zeros_like(keypoint_features),
        )
        tokens = (
            self.keypoint_feature_proj(safe_features)
            + self.keypoint_coord_proj(coord_input)
            + self.keypoint_type_embed.to(device=keypoint_features.device, dtype=keypoint_features.dtype)
        )
        missing_tokens = (
            self.missing_keypoint_token.to(device=keypoint_features.device, dtype=keypoint_features.dtype)
            + self.keypoint_type_embed.to(device=keypoint_features.device, dtype=keypoint_features.dtype)
        ).expand_as(tokens)
        return torch.where(keypoint_mask.unsqueeze(-1), tokens, missing_tokens)

    def forward(
        self,
        dino_tokens: torch.Tensor,
        agg_tokens: torch.Tensor,
        *,
        keypoint_features: torch.Tensor,
        keypoint_uv: torch.Tensor,
        keypoint_conf: torch.Tensor,
        keypoint_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if dino_tokens.ndim != 4 or agg_tokens.ndim != 4:
            raise ValueError("HandPromptAdapter expects [B, T, N, C] DINO and aggregator tokens")
        if dino_tokens.shape[:3] != agg_tokens.shape[:3]:
            raise ValueError("DINO and aggregator tokens must share [B, T, N]")
        batch_size, num_frames, token_count = dino_tokens.shape[:3]
        expected_keypoint_shape = (batch_size, num_frames, 2, self.joint_count)
        if keypoint_features.shape[:4] != expected_keypoint_shape:
            raise ValueError("keypoint_features must have shape [B, T, 2, 21, C]")
        if keypoint_uv.shape != (*expected_keypoint_shape, 2) or keypoint_conf.shape != expected_keypoint_shape:
            raise ValueError("keypoint UV/confidence shapes are incompatible")
        if keypoint_mask.shape != expected_keypoint_shape:
            raise ValueError("keypoint_mask must have shape [B, T, 2, 21]")
        flat_dino_tokens = dino_tokens.reshape(batch_size * num_frames, token_count, dino_tokens.shape[-1])
        flat_agg_tokens = agg_tokens.reshape(batch_size * num_frames, token_count, agg_tokens.shape[-1])
        dino_features = self.dino_proj(flat_dino_tokens)
        agg_features = self.agg_proj(flat_agg_tokens)
        memory = self.token_fuse(torch.cat([dino_features, agg_features], dim=-1))
        query = self.side_type_queries.to(device=memory.device, dtype=memory.dtype).expand(
            batch_size, num_frames, -1, -1, -1
        )
        query = self.query_norm(query.reshape(batch_size * num_frames, 2 * self.prompt_type_count, self.hand_feature_dim))
        mask = keypoint_mask.to(device=memory.device, dtype=torch.bool)
        mask = (
            mask
            & torch.isfinite(keypoint_uv).all(dim=-1).to(device=memory.device)
            & torch.isfinite(keypoint_conf).to(device=memory.device)
            & torch.isfinite(keypoint_features).all(dim=-1).to(device=memory.device)
        )
        keypoint_conf = keypoint_conf.to(device=memory.device, dtype=memory.dtype)
        keypoint_tokens = self._keypoint_tokens(
            keypoint_features=keypoint_features.to(device=memory.device, dtype=memory.dtype),
            keypoint_uv=keypoint_uv.to(device=memory.device, dtype=memory.dtype),
            keypoint_conf=keypoint_conf,
            keypoint_mask=mask,
        )

        early_block_count = min(2, len(self.blocks))
        for block in self.blocks[:early_block_count]:
            query = block(query, memory)
        coarse_queries = query.reshape(
            batch_size, num_frames, 2, self.prompt_type_count, self.hand_feature_dim
        )
        _, coarse_context, _, _, _ = self.keypoint_context_projector(
            keypoint_tokens,
            mask,
            keypoint_conf=keypoint_conf,
            semantic_queries=coarse_queries,
        )
        early_gate = torch.tanh(self.early_loc_gate.to(device=query.device, dtype=query.dtype))
        query = (
            coarse_queries + early_gate * self.loc_context_proj(coarse_context)
        ).reshape(batch_size * num_frames, 2 * self.prompt_type_count, self.hand_feature_dim)
        for block in self.blocks[early_block_count:]:
            query = block(query, memory)

        visual_hand_features = self.output_norm(query).reshape(
            batch_size, num_frames, 2, self.prompt_type_count, self.hand_feature_dim
        )
        (
            keypoint_context,
            semantic_keypoint_context,
            root_attention_weights,
            root_wrist_observed,
            root_any_keypoint_observed,
        ) = self.keypoint_context_projector(
            keypoint_tokens,
            mask,
            keypoint_conf=keypoint_conf,
            semantic_queries=visual_hand_features,
        )
        hand_features = visual_hand_features + self.loc_context_proj(semantic_keypoint_context)
        hand_prompt_set = self.prompt_projector(hand_features)
        keypoint_prompt_tokens = self.keypoint_prompt_projector(keypoint_tokens)
        semantic_condition_token_visual = self.keypoint_prompt_projector(semantic_keypoint_context)
        keypoint_geometry = normalized_keypoint_geometry(
            keypoint_uv.to(device=hand_features.device, dtype=hand_features.dtype),
            keypoint_conf,
            mask,
        )
        scene_summary = flat_agg_tokens.mean(dim=1)
        scene_metric_feature = self.scene_metric_feature_projector(scene_summary).reshape(
            batch_size, num_frames, self.hand_feature_dim
        )
        scene_metric_prompt = self.scene_metric_prompt_projector(scene_metric_feature).unsqueeze(2)
        return {
            "hand_feature_set": hand_features,
            "hand_prompt_set": hand_prompt_set,
            "keypoint_prompt_tokens": keypoint_prompt_tokens,
            "keypoint_context": keypoint_context,
            "semantic_keypoint_context": semantic_keypoint_context,
            "semantic_condition_feature_visual": semantic_keypoint_context,
            "semantic_condition_token_visual": semantic_condition_token_visual,
            "root_attention_weights": root_attention_weights,
            "root_wrist_observed": root_wrist_observed,
            "root_any_keypoint_observed": root_any_keypoint_observed,
            "keypoint_geometry": keypoint_geometry,
            "keypoint_geometry_valid": mask,
            "scene_metric_feature": scene_metric_feature,
            "scene_metric_prompt": scene_metric_prompt,
        }
