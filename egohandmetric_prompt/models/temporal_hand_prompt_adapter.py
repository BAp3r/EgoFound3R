from __future__ import annotations


import torch
from torch import nn
from .transformer_blocks import paired_directional_alibi_mask


class TemporalHandPromptAdapter(nn.Module):
    def __init__(
        self,
        prompt_dim: int,
        num_layers: int = 2,
        num_heads: int = 12,
        mlp_ratio: int = 4,
    ) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=prompt_dim,
            nhead=num_heads,
            dim_feedforward=prompt_dim * mlp_ratio,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(prompt_dim)
        self.gate_logit = nn.Parameter(torch.tensor(-2.0))

    def forward(self, hand_prompt_set: torch.Tensor) -> torch.Tensor:
        if hand_prompt_set.ndim != 5:
            raise ValueError("hand_prompt_set must have shape [B, T, 2, K, C]")
        batch_size, num_frames, side_count, prompt_type_count, prompt_dim = hand_prompt_set.shape
        flat = hand_prompt_set.permute(0, 2, 3, 1, 4).reshape(
            batch_size * side_count * prompt_type_count,
            num_frames,
            prompt_dim,
        )
        temporal_bias = paired_directional_alibi_mask(
            batch_size=flat.shape[0],
            num_heads=self.num_heads,
            sequence_length=num_frames,
            device=flat.device,
            dtype=flat.dtype,
        )
        delta = self.output_norm(self.encoder(flat, mask=temporal_bias))
        delta = delta.reshape(batch_size, side_count, prompt_type_count, num_frames, prompt_dim).permute(0, 3, 1, 2, 4)
        gate = torch.sigmoid(self.gate_logit).to(device=hand_prompt_set.device, dtype=hand_prompt_set.dtype)
        return hand_prompt_set + gate * delta.to(dtype=hand_prompt_set.dtype)
