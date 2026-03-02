
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

from .rng import torch_uniform01


# FP4-like codebook (16 entries); one duplicate 0 fills the 4-bit space.
_FP4_CODEBOOK = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0, 0.0],
    dtype=torch.float32,
)

_MAX_CODE = 6.0


@dataclass(frozen=True)
class FP4QuantResult:
    codes: torch.Tensor  # int8 codes in [0, 15]
    scale: torch.Tensor  # float32, broadcastable to input
    dequant: torch.Tensor  # float16 reconstructed values


def fp4_codebook(device: torch.device | str | None = None) -> torch.Tensor:
    cb = _FP4_CODEBOOK
    if device is not None:
        cb = cb.to(device)
    return cb


def _group_reshape_last(x: torch.Tensor, group_size: int) -> Tuple[torch.Tensor, int]:
    # Reshape last dim into groups; pad if needed.
    n = x.shape[-1]
    g = (n + group_size - 1) // group_size
    pad = g * group_size - n
    if pad:
        x = torch.nn.functional.pad(x, (0, pad))
    xg = x.reshape(*x.shape[:-1], g, group_size)
    return xg, pad


def _fp4_quantize_scaled(y: torch.Tensor) -> torch.Tensor:
    """Quantize y (already scaled by group scale) to FP4 codes.

    y: float32 tensor of arbitrary shape.
    Returns int8 codes in [0,15].

    This is a *vectorized threshold mapper* (no [...,16] diff tensor),
    much faster than argmin-over-codebook.
    """
    # Use magnitude thresholds at midpoints between codebook values.
    # Positive levels: 0, 0.5, 1, 1.5, 2, 3, 4, 6
    # Midpoints: 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0
    a = y.abs()
    sgn = y < 0

    # Compute magnitude bin 0..7
    # bin 0 => 0
    # bin 1 => 0.5
    # bin 2 => 1
    # bin 3 => 1.5
    # bin 4 => 2
    # bin 5 => 3
    # bin 6 => 4
    # bin 7 => 6
    b = torch.zeros_like(a, dtype=torch.int64)

    b = b + (a >= 0.25).to(torch.int64)
    b = b + (a >= 0.75).to(torch.int64)
    b = b + (a >= 1.25).to(torch.int64)
    b = b + (a >= 1.75).to(torch.int64)
    b = b + (a >= 2.50).to(torch.int64)
    b = b + (a >= 3.50).to(torch.int64)
    b = b + (a >= 5.00).to(torch.int64)

    # Map to code indices.
    # For b==0 => code 0 regardless of sign.
    # For b in 1..7:
    #   positive => code=b
    #   negative => code=7+b  (because -0.5 starts at index 8)
    pos_code = b  # 0..7
    neg_code = 7 + b  # 7..14 (but b==0 would give 7; we fix below)

    codes = torch.where(sgn, neg_code, pos_code)
    codes = torch.where(b == 0, torch.zeros_like(codes), codes)

    return codes.to(torch.int8)


def fp4_quantize_groupwise(
    x: torch.Tensor,
    *,
    group_size: int = 32,
    axis: int = -1,
    sigma: int | bytes | str,
    domain: str,
    scale_jitter: float = 0.0,
) -> FP4QuantResult:
    """Group-wise FP4 quantization with optional **seeded scale jitter**.

    Faster implementation:
      - vectorized threshold mapper (no giant diff tensor)
      - keeps the same API as the earlier reference code
    """
    if group_size <= 0:
        raise ValueError("group_size must be positive")

    cb = fp4_codebook(x.device)  # [16]
    # Move axis to last to reuse code.
    x_last = x.movedim(axis, -1).contiguous()
    xg, pad = _group_reshape_last(x_last, group_size)  # [..., G, group]

    # Scale per group
    max_abs = xg.detach().abs().amax(dim=-1, keepdim=True).to(torch.float32)
    eps = torch.finfo(torch.float32).eps
    max_abs = torch.maximum(max_abs, torch.tensor(eps, device=x.device))
    scale = (max_abs / _MAX_CODE)  # [..., G, 1]

    if scale_jitter != 0.0:
        u = torch_uniform01(tuple(scale.shape), sigma=sigma, domain=domain + "/jitter", device=x.device, dtype=torch.float32)
        jitter = (2.0 * u - 1.0) * float(scale_jitter)
        scale = scale * (1.0 + jitter)

    y = (xg.to(torch.float32) / scale)  # [..., G, group]
    codes = _fp4_quantize_scaled(y)  # [..., G, group] int8
    deq = cb[codes.to(torch.int64)].to(torch.float32) * scale  # [..., G, group]
    deq = deq.to(torch.float16)

    # Undo padding and restore axis
    deq_last = deq.reshape(*x_last.shape[:-1], -1)
    codes_last = codes.reshape(*x_last.shape[:-1], -1)
    if pad:
        deq_last = deq_last[..., : x_last.shape[-1]]
        codes_last = codes_last[..., : x_last.shape[-1]]

    deq_out = deq_last.movedim(-1, axis)
    codes_out = codes_last.movedim(-1, axis)

    # Broadcastable scale: keep group axis in place
    scale_last = scale  # [..., G, 1]
    scale_full = scale_last.repeat_interleave(group_size, dim=-1)  # [..., G, group]
    scale_full = scale_full.reshape(*x_last.shape[:-1], -1)
    if pad:
        scale_full = scale_full[..., : x_last.shape[-1]]
    scale_out = scale_full.movedim(-1, axis)

    return FP4QuantResult(codes=codes_out, scale=scale_out, dequant=deq_out)
