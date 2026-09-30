from __future__ import annotations


from dataclasses import dataclass
import torch
from .token_layout import TokenLayout


@dataclass(slots=True)
class PromptRunOutput:
    public_tokens: torch.Tensor
    internal_tokens: torch.Tensor
    layout: TokenLayout


class PromptInjectedAggregatorRunner:
    @staticmethod
    def append_prompts(
        public_tokens: torch.Tensor,
        *,
        hand_prompts: torch.Tensor,
        metric_prompts: torch.Tensor | None,
        patch_token_start: int,
        patch_grid_hw: tuple[int, int] | None = None,
    ) -> PromptRunOutput:
        if public_tokens.ndim != 4:
            raise ValueError("public_tokens must have shape [B, T, N, C]")
        if hand_prompts.ndim != 4 or hand_prompts.shape[:2] != public_tokens.shape[:2]:
            raise ValueError("hand_prompts must have shape [B, T, H, C]")
        if hand_prompts.shape[-1] != public_tokens.shape[-1]:
            raise ValueError("hand prompt dimension must match public token dimension")
        prompt_types = [hand_prompts]
        metric_count = 0
        if metric_prompts is not None:
            if metric_prompts.ndim != 4 or metric_prompts.shape[:2] != public_tokens.shape[:2]:
                raise ValueError("metric_prompts must have shape [B, T, M, C]")
            if metric_prompts.shape[-1] != public_tokens.shape[-1]:
                raise ValueError("metric prompt dimension must match public token dimension")
            prompt_types.append(metric_prompts)
            metric_count = int(metric_prompts.shape[2])
        packed_prompts = torch.cat(prompt_types, dim=2)
        public_tokens_with_prompts = torch.cat([public_tokens, packed_prompts], dim=2)
        layout = TokenLayout(
            camera_token_count=1,
            register_token_count=patch_token_start - 1,
            patch_token_count=public_tokens.shape[2] - patch_token_start,
            hand_prompt_count=int(hand_prompts.shape[2]),
            metric_prompt_count=metric_count,
            patch_grid_hw=patch_grid_hw,
        )
        return PromptRunOutput(
            public_tokens=public_tokens_with_prompts,
            internal_tokens=layout.public_to_internal(public_tokens_with_prompts),
            layout=layout,
        )
