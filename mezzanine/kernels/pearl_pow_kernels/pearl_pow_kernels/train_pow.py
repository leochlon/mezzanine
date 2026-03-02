from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn

from .fp4 import fp4_quantize_groupwise
from .quant import (
    dequantize_float8,
    dequantize_int8,
    quantize_float8,
    quantize_int8_symmetric,
    quantize_int8_unbiased,
)
from .rng import torch_noise
from .rot_gemm import rot_encode
from .trace import TraceConfig, sampled_gemm_trace


@dataclass(frozen=True)
class TrainPowConfig:
    scheme: Literal["rot", "qnoise", "fp4"] = "qnoise"

    # Shared trace config (optional)
    trace: TraceConfig = TraceConfig(quant="fp16")

    # qnoise params
    noise_dist: Literal["normal", "rademacher", "uniform"] = "normal"
    noise_scale: float = 0.01
    qnoise_quant: Literal["float8", "int8"] = "float8"
    float8_dtype: torch.dtype = torch.float8_e4m3fn
    int8_axis: int | None = None
    int8_unbiased: bool = True

    # fp4 params
    fp4_group_size: int = 32
    fp4_scale_jitter: float = 0.01


class _QNoiseGemmSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A: torch.Tensor, B: torch.Tensor, sigma_obj, cfg: TrainPowConfig) -> torch.Tensor:
        sigma = sigma_obj  # arbitrary python object allowed in ctx, but not in outputs

        E = torch_noise(tuple(A.shape), sigma=sigma, domain="train/qnoise/E", dist=cfg.noise_dist, scale=cfg.noise_scale, device=A.device, dtype=A.dtype)
        F = torch_noise(tuple(B.shape), sigma=sigma, domain="train/qnoise/F", dist=cfg.noise_dist, scale=cfg.noise_scale, device=B.device, dtype=B.dtype)
        A_noisy = A + E
        B_noisy = B + F

        if cfg.qnoise_quant == "float8":
            Qa = quantize_float8(A_noisy, dtype=cfg.float8_dtype)
            Qb = quantize_float8(B_noisy, dtype=cfg.float8_dtype)
            A_q = dequantize_float8(Qa.q, Qa.scale, out_dtype=torch.float16)
            B_q = dequantize_float8(Qb.q, Qb.scale, out_dtype=torch.float16)
            C = A_q @ B_q
        elif cfg.qnoise_quant == "int8":
            if cfg.int8_unbiased:
                Qa = quantize_int8_unbiased(A_noisy, sigma=sigma, domain="train/qnoise/int8A", axis=cfg.int8_axis)
                Qb = quantize_int8_unbiased(B_noisy, sigma=sigma, domain="train/qnoise/int8B", axis=cfg.int8_axis)
            else:
                Qa = quantize_int8_symmetric(A_noisy, axis=cfg.int8_axis)
                Qb = quantize_int8_symmetric(B_noisy, axis=cfg.int8_axis)
            A_q = dequantize_int8(Qa.q, Qa.scale).to(torch.float16)
            B_q = dequantize_int8(Qb.q, Qb.scale).to(torch.float16)
            C = A_q @ B_q
        else:
            raise ValueError(f"Unknown qnoise_quant={cfg.qnoise_quant}")

        ctx.save_for_backward(A_q, B_q)
        return C.to(A.dtype)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        A_q, B_q = ctx.saved_tensors
        g = grad_out.to(torch.float32)
        # STE: treat quantization as identity, gradient through GEMM only.
        grad_A = g @ B_q.to(torch.float32).t()
        grad_B = A_q.to(torch.float32).t() @ g
        return grad_A.to(grad_out.dtype), grad_B.to(grad_out.dtype), None, None


class _FP4GemmSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A: torch.Tensor, B: torch.Tensor, sigma_obj, cfg: TrainPowConfig) -> torch.Tensor:
        sigma = sigma_obj
        Qa = fp4_quantize_groupwise(
            A,
            group_size=cfg.fp4_group_size,
            axis=-1,
            sigma=sigma,
            domain="train/fp4/A",
            scale_jitter=cfg.fp4_scale_jitter,
        )
        Qb = fp4_quantize_groupwise(
            B,
            group_size=cfg.fp4_group_size,
            axis=0,
            sigma=sigma,
            domain="train/fp4/B",
            scale_jitter=cfg.fp4_scale_jitter,
        )
        A_q = Qa.dequant
        B_q = Qb.dequant
        C = (A_q @ B_q).to(A.dtype)
        ctx.save_for_backward(A_q, B_q)
        return C

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        A_q, B_q = ctx.saved_tensors
        g = grad_out.to(torch.float32)
        grad_A = g @ B_q.to(torch.float32).t()
        grad_B = A_q.to(torch.float32).t() @ g
        return grad_A.to(grad_out.dtype), grad_B.to(grad_out.dtype), None, None


def train_pow_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    sigma: int | bytes | str,
    cfg: TrainPowConfig,
    return_transcript: bool = False,
) -> Tuple[torch.Tensor, Optional[bytes]]:
    """Training-friendly GEMM wrapper.

    - scheme="rot" is fully differentiable (exact gradients).
    - scheme in {"qnoise","fp4"} uses STE for quantization.

    If return_transcript=True, also returns a deterministic sampled trace hash.
    """
    if cfg.scheme == "rot":
        A_rot, B_rot = rot_encode(A, B, sigma=sigma)
        C = A_rot @ B_rot
        if return_transcript:
            digest, _ = sampled_gemm_trace(A_rot, B_rot, sigma=sigma, config=cfg.trace)
            return C, digest
        return C, None

    if cfg.scheme == "qnoise":
        C = _QNoiseGemmSTE.apply(A, B, sigma, cfg)
        if return_transcript:
            # Recompute encoded tensors for the trace (reference-only).
            # In production, hash is updated inside the fused kernel.
            E = torch_noise(tuple(A.shape), sigma=sigma, domain="train/qnoise/E", dist=cfg.noise_dist, scale=cfg.noise_scale, device=A.device, dtype=A.dtype)
            F = torch_noise(tuple(B.shape), sigma=sigma, domain="train/qnoise/F", dist=cfg.noise_dist, scale=cfg.noise_scale, device=B.device, dtype=B.dtype)
            A_noisy = A + E
            B_noisy = B + F
            if cfg.qnoise_quant == "float8":
                Qa = quantize_float8(A_noisy, dtype=cfg.float8_dtype)
                Qb = quantize_float8(B_noisy, dtype=cfg.float8_dtype)
                A_q = dequantize_float8(Qa.q, Qa.scale, out_dtype=torch.float16)
                B_q = dequantize_float8(Qb.q, Qb.scale, out_dtype=torch.float16)
            else:
                if cfg.int8_unbiased:
                    Qa = quantize_int8_unbiased(A_noisy, sigma=sigma, domain="train/qnoise/int8A", axis=cfg.int8_axis)
                    Qb = quantize_int8_unbiased(B_noisy, sigma=sigma, domain="train/qnoise/int8B", axis=cfg.int8_axis)
                else:
                    Qa = quantize_int8_symmetric(A_noisy, axis=cfg.int8_axis)
                    Qb = quantize_int8_symmetric(B_noisy, axis=cfg.int8_axis)
                A_q = dequantize_int8(Qa.q, Qa.scale).to(torch.float16)
                B_q = dequantize_int8(Qb.q, Qb.scale).to(torch.float16)
            digest, _ = sampled_gemm_trace(A_q, B_q, sigma=sigma, config=cfg.trace)
            return C, digest
        return C, None

    if cfg.scheme == "fp4":
        C = _FP4GemmSTE.apply(A, B, sigma, cfg)
        if return_transcript:
            Qa = fp4_quantize_groupwise(
                A, group_size=cfg.fp4_group_size, axis=-1, sigma=sigma, domain="train/fp4/A", scale_jitter=cfg.fp4_scale_jitter
            )
            Qb = fp4_quantize_groupwise(
                B, group_size=cfg.fp4_group_size, axis=0, sigma=sigma, domain="train/fp4/B", scale_jitter=cfg.fp4_scale_jitter
            )
            digest, _ = sampled_gemm_trace(Qa.dequant, Qb.dequant, sigma=sigma, config=cfg.trace)
            return C, digest
        return C, None

    raise ValueError(f"Unknown scheme={cfg.scheme}")


class PowLinear(nn.Module):
    """Drop-in Linear layer that routes matmul through a configured scheme.

    This is a pragmatic bridge for experiments on training/finetuning: it lets you
    swap the linear-algebra path and check whether training remains stable.

    Usage:
        layer = PowLinear(in_features, out_features, cfg=TrainPowConfig(...))
        y = layer(x, sigma=block_seed)

    Notes:
        - For qnoise/fp4, gradients use STE.
        - For rot, gradients are exact (since the rotation cancels algebraically).
    """

    def __init__(self, in_features: int, out_features: int, *, bias: bool = True, cfg: TrainPowConfig | None = None):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.cfg = cfg if cfg is not None else TrainPowConfig()
        # Init like nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor, *, sigma: int | bytes | str = 0, return_transcript: bool = False):
        # x: [B, in_features], weight: [out, in]
        A = x
        B = self.weight.t()
        y, digest = train_pow_gemm(A, B, sigma=sigma, cfg=self.cfg, return_transcript=return_transcript)
        if self.bias is not None:
            y = y + self.bias
        return (y, digest) if return_transcript else y
