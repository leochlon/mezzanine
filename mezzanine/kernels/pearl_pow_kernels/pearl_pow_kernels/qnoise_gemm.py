
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import torch

from .quant import (
    dequantize_float8,
    dequantize_int8,
    quantize_float8,
    quantize_int8_symmetric,
    quantize_int8_unbiased,
)
from .rng import torch_noise
from .trace import TraceConfig, sampled_gemm_trace

# Optional Triton fused GEMM+hash
try:  # pragma: no cover
    from .gemm_triton_fused import gemm_with_fused_trace, triton_available
except Exception:  # pragma: no cover
    gemm_with_fused_trace = None
    def triton_available() -> bool:
        return False


@dataclass(frozen=True)
class QNoiseGemmConfig:
    """Config for noise+quantize GEMM (with transcript hashing)."""

    noise_dist: Literal["normal", "rademacher", "uniform"] = "normal"
    noise_scale: float = 0.01  # std for normal; amplitude for uniform/rademacher

    quant: Literal["float8", "int8"] = "float8"
    float8_dtype: torch.dtype = torch.float8_e4m3fn

    # INT8 quantization options
    int8_axis: int | None = None  # None => per-tensor, 0/1 => per-axis
    int8_unbiased: bool = True  # stochastic rounding for unbiased estimator

    # Output dtype
    out_dtype: torch.dtype = torch.float32

    # Trace hashing
    trace: TraceConfig = TraceConfig(quant="fp16")


def qnoise_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    sigma: int | bytes | str,
    config: QNoiseGemmConfig = QNoiseGemmConfig(),
) -> Tuple[torch.Tensor, bytes, dict]:
    """Compute C = Q(A+E) @ Q(B+F) and a sampled transcript hash.

    Reference path:
      - add deterministic noise (seeded by sigma),
      - quantize+dequantize (float8 or int8),
      - matmul, then hash a sampled subset of tiles.

    Optional CUDA/Triton path (float8 mode only):
      - uses a fused GEMM+hash kernel with deterministic in-kernel noise,
      - intended for overhead experiments (it does not model float8 quantization).
    """
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("A and B must be 2D tensors")
    M, K = A.shape
    K2, N = B.shape
    if K != K2:
        raise ValueError(f"Shape mismatch: A is {A.shape}, B is {B.shape}")

    use_fast = (
        triton_available()
        and config.quant == "float8"
        and A.device.type == "cuda"
        and B.device.type == "cuda"
        and (not A.requires_grad)
        and (not B.requires_grad)
        and A.dtype in (torch.float16, torch.bfloat16)
        and B.dtype in (torch.float16, torch.bfloat16)
        and gemm_with_fused_trace is not None
    )

    if use_fast:
        C, digest, coords, meta2 = gemm_with_fused_trace(
            A, B,
            sigma=sigma,
            trace=config.trace,
            out_dtype=config.out_dtype,
            permute_ktiles=False,
            noise_scale=float(config.noise_scale),
        )
        meta = {
            "scheme": "qnoise_gemm",
            "noise_dist": config.noise_dist,
            "noise_scale": float(config.noise_scale),
            "quant": config.quant,
            "trace_coords": coords,
            **meta2,
        }
        return C, digest, meta

    # --- Reference path ---
    device = A.device
    E = torch_noise(tuple(A.shape), sigma=sigma, domain="qnoise/E", dist=config.noise_dist, scale=config.noise_scale, device=device, dtype=A.dtype)
    F = torch_noise(tuple(B.shape), sigma=sigma, domain="qnoise/F", dist=config.noise_dist, scale=config.noise_scale, device=device, dtype=B.dtype)
    A_noisy = A + E
    B_noisy = B + F

    if config.quant == "float8":
        Qa = quantize_float8(A_noisy, dtype=config.float8_dtype)
        Qb = quantize_float8(B_noisy, dtype=config.float8_dtype)
        A_q = dequantize_float8(Qa.q, Qa.scale, out_dtype=torch.float16)
        B_q = dequantize_float8(Qb.q, Qb.scale, out_dtype=torch.float16)
        C = (A_q @ B_q).to(config.out_dtype)

        transcript, coords = sampled_gemm_trace(A_q, B_q, sigma=sigma, config=config.trace)

    elif config.quant == "int8":
        if config.int8_unbiased:
            Qa = quantize_int8_unbiased(A_noisy, sigma=sigma, domain="qnoise/int8A", axis=config.int8_axis)
            Qb = quantize_int8_unbiased(B_noisy, sigma=sigma, domain="qnoise/int8B", axis=config.int8_axis)
        else:
            Qa = quantize_int8_symmetric(A_noisy, axis=config.int8_axis)
            Qb = quantize_int8_symmetric(B_noisy, axis=config.int8_axis)

        A_q = dequantize_int8(Qa.q, Qa.scale).to(torch.float16)
        B_q = dequantize_int8(Qb.q, Qb.scale).to(torch.float16)
        C = (A_q @ B_q).to(config.out_dtype)
        transcript, coords = sampled_gemm_trace(A_q, B_q, sigma=sigma, config=config.trace)
    else:
        raise ValueError(f"Unknown quant={config.quant}")

    meta = {
        "scheme": "qnoise_gemm",
        "noise_dist": config.noise_dist,
        "noise_scale": float(config.noise_scale),
        "quant": config.quant,
        "trace_coords": coords,
        "backend": "torch+cpu_trace",
    }
    return C, transcript, meta
