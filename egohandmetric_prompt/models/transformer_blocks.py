from __future__ import annotations


import math
import torch
from torch import nn


def _alibi_slopes(num_heads: int, *, device: torch.device) -> torch.Tensor:
    if num_heads <= 0:
        raise ValueError("num_heads must be positive")

    def _power_of_two_slopes(count: int) -> list[float]:
        start = 2.0 ** (-2.0 ** -(math.log2(count) - 3.0))
        return [start ** (index + 1) for index in range(count)]

    closest_power = 2 ** int(math.floor(math.log2(num_heads)))
    slopes = _power_of_two_slopes(closest_power)
    if closest_power != num_heads:
        slopes.extend(_power_of_two_slopes(2 * closest_power)[0::2][: num_heads - closest_power])
    return torch.tensor(slopes, device=device, dtype=torch.float32)


def paired_directional_alibi_mask(
    *,
    batch_size: int,
    num_heads: int,
    sequence_length: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if not dtype.is_floating_point:
        raise ValueError("dtype must be floating point")

    slopes = _alibi_slopes(num_heads, device=device)
    positions = torch.arange(sequence_length, device=device, dtype=torch.float32)
    signed_distance = positions[:, None] - positions[None, :]
    past_distance = signed_distance.clamp_min(0.0)
    future_distance = (-signed_distance).clamp_min(0.0)

    bias = torch.empty(
        num_heads,
        sequence_length,
        sequence_length,
        device=device,
        dtype=torch.float32,
    )
    paired_heads = num_heads - num_heads % 2
    if paired_heads:
        first = slopes[:paired_heads:2, None, None]
        second = slopes[1:paired_heads:2, None, None]
        bias[0:paired_heads:2] = -first * past_distance - second * future_distance
        bias[1:paired_heads:2] = -second * past_distance - first * future_distance
    if paired_heads != num_heads:
        bias[-1] = -slopes[-1] * signed_distance.abs()

    return (
        bias.to(dtype=dtype)
        .unsqueeze(0)
        .expand(batch_size, -1, -1, -1)
        .reshape(batch_size * num_heads, sequence_length, sequence_length)
        .contiguous()
    )


class TransformerDecoderBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.cross_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x, memory, self_key_padding_mask=None, memory_key_padding_mask=None):
        residual = self.self_norm(x)
        self_out, _ = self.self_attn(
            residual,
            residual,
            residual,
            key_padding_mask=self_key_padding_mask,
            need_weights=False,
        )
        x = x + self_out

        cross_query = self.cross_norm(x)
        cross_out, _ = self.cross_attn(
            cross_query,
            memory,
            memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )
        x = x + cross_out
        return x + self.mlp(self.mlp_norm(x))
