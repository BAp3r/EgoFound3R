from __future__ import annotations


from dataclasses import dataclass
import torch


@dataclass(frozen=True, slots=True)
class TokenLayout:
    camera_token_count: int
    register_token_count: int
    patch_token_count: int
    hand_prompt_count: int
    metric_prompt_count: int = 0
    patch_grid_hw: tuple[int, int] | None = None

    @property
    def camera_register_end(self) -> int:
        return self.camera_token_count + self.register_token_count

    @property
    def hand_prompt_start(self) -> int:
        return self.camera_register_end + self.patch_token_count

    @property
    def patch_token_end(self) -> int:
        return self.hand_prompt_start

    @property
    def prompt_token_count(self) -> int:
        return self.hand_prompt_count + self.metric_prompt_count

    @property
    def total_tokens(self) -> int:
        return self.camera_register_end + self.patch_token_count + self.prompt_token_count

    @property
    def internal_patch_start(self) -> int:
        return self.camera_register_end + self.prompt_token_count

    def split_public(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        camera_register = tokens[..., : self.camera_register_end, :]
        patch_tokens = tokens[..., self.camera_register_end : self.patch_token_end, :]
        prompt_tokens = tokens[..., self.hand_prompt_start :, :]
        return camera_register, patch_tokens, prompt_tokens

    def split_internal(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        camera_register = tokens[..., : self.camera_register_end, :]
        hand_prompts = tokens[..., self.camera_register_end : self.internal_patch_start, :]
        patch_tokens = tokens[..., self.internal_patch_start :, :]
        return camera_register, hand_prompts, patch_tokens

    def split_internal_prompt_types(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, prompt_tokens, _ = self.split_internal(tokens)
        hand_end = self.hand_prompt_count
        hand_prompts = prompt_tokens[..., :hand_end, :]
        metric_prompts = prompt_tokens[..., hand_end : hand_end + self.metric_prompt_count, :]
        return hand_prompts, metric_prompts

    def public_to_internal(self, tokens: torch.Tensor) -> torch.Tensor:
        camera_register, patch_tokens, prompt_tokens = self.split_public(tokens)
        return torch.cat([camera_register, prompt_tokens, patch_tokens], dim=-2)

    def internal_to_public(self, tokens: torch.Tensor) -> torch.Tensor:
        camera_register, prompt_tokens, patch_tokens = self.split_internal(tokens)
        return torch.cat([camera_register, patch_tokens, prompt_tokens], dim=-2)

    def with_hand_prompt_count(self, hand_prompt_count: int) -> "TokenLayout":
        return TokenLayout(
            camera_token_count=self.camera_token_count,
            register_token_count=self.register_token_count,
            patch_token_count=self.patch_token_count,
            hand_prompt_count=hand_prompt_count,
            metric_prompt_count=self.metric_prompt_count,
            patch_grid_hw=self.patch_grid_hw,
        )
