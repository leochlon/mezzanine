from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch

from .rng import torch_uniform01


@dataclass(frozen=True)
class QuantResult:
    q: torch.Tensor
    scale: torch.Tensor  # broadcastable to q
    zero_point: torch.Tensor | None = None  # reserved for asymmetric quant


def _safe_scale_from_max(max_abs: torch.Tensor, *, qmax: float) -> torch.Tensor:
    # Avoid division by zero: if all zeros, scale=1.
    eps = torch.finfo(torch.float32).eps
    max_abs = torch.maximum(max_abs.to(torch.float32), torch.tensor(eps, device=max_abs.device))
    return max_abs / float(qmax)


def quantize_int8_symmetric(
    x: torch.Tensor,
    *,
    axis: int | None = None,
    scale: torch.Tensor | None = None,
) -> QuantResult:
    """Symmetric INT8 quantization with int32-friendly accumulation.

    Q(x) = clamp(round(x/scale), -127, 127).to(int8), dequant = q * scale.

    Parameters
    ----------
    x:
        Input float tensor.
    axis:
        If provided, compute per-axis scale on that axis (e.g., per-row).
    scale:
        Optional externally provided scale.

    Returns
    -------
    QuantResult
    """
    if scale is None:
        if axis is None:
            max_abs = x.detach().abs().max()
        else:
            max_abs = x.detach().abs().amax(dim=axis, keepdim=True)
        scale = _safe_scale_from_max(max_abs, qmax=127.0)

    q = torch.clamp(torch.round(x / scale), -127, 127).to(torch.int8)
    return QuantResult(q=q, scale=scale.to(torch.float32))


def dequantize_int8(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.to(torch.float32) * scale


def quantize_int8_unbiased(
    x: torch.Tensor,
    *,
    sigma: int | bytes | str,
    domain: str,
    axis: int | None = None,
    scale: torch.Tensor | None = None,
) -> QuantResult:
    """INT8 quantization with **stochastic rounding** (unbiased).

    This uses stochastic rounding so (ignoring clipping) the dequantized value is an
    unbiased estimator of the input.

    Returns q (int8) and scale (float32).
    """
    if scale is None:
        if axis is None:
            max_abs = x.detach().abs().max()
        else:
            max_abs = x.detach().abs().amax(dim=axis, keepdim=True)
        scale = _safe_scale_from_max(max_abs, qmax=127.0)

    y = x / scale
    y_floor = torch.floor(y)
    frac = (y - y_floor).to(torch.float32)

    u = torch_uniform01(tuple(x.shape), sigma=sigma, domain=domain + "/u", device=x.device, dtype=torch.float32)
    y_q = y_floor + (u < frac).to(y_floor.dtype)

    y_q = torch.clamp(y_q, -127, 127)
    q = y_q.to(torch.int8)
    return QuantResult(q=q, scale=scale.to(torch.float32))


def quantize_float8(
    x: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float8_e4m3fn,
    scale: torch.Tensor | None = None,
    axis: int | None = None,
) -> QuantResult:
    """Float8 quantization with explicit scaling.

    We store a scale so dequantization is: q.to(fp16) * scale.
    """
    finfo = torch.finfo(dtype)
    qmax = float(finfo.max)

    if scale is None:
        if axis is None:
            max_abs = x.detach().abs().max()
        else:
            max_abs = x.detach().abs().amax(dim=axis, keepdim=True)
        scale = _safe_scale_from_max(max_abs, qmax=qmax)

    q = (x / scale).to(dtype)
    return QuantResult(q=q, scale=scale.to(torch.float32))


def dequantize_float8(q: torch.Tensor, scale: torch.Tensor, *, out_dtype: torch.dtype = torch.float16) -> torch.Tensor:
    return q.to(out_dtype) * scale
