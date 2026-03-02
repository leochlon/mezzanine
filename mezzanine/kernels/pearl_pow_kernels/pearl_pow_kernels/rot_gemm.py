
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch

from .hadamard import fwht, is_power_of_two
from .rng import torch_sign_vector
from .trace import TraceConfig, sampled_gemm_trace

# Optional Triton fused GEMM+hash
try:  # pragma: no cover
    from .gemm_triton_fused import gemm_with_fused_trace, triton_available
except Exception:  # pragma: no cover
    gemm_with_fused_trace = None
    def triton_available() -> bool:
        return False


@dataclass(frozen=True)
class RotGemmConfig:
    """Configuration for the random-rotation scheme.

    CPU reference uses randomized Hadamard stages.
    CUDA fast path uses a deterministic K-tile permutation gauge (AB-invariant) and
    can fuse hashing into the GEMM kernel epilogue.
    """

    stages: int = 2
    trace: TraceConfig = TraceConfig()
    return_encoded: bool = False


def _stage_sign(sigma: int | bytes | str, K: int, stage: int, device, dtype) -> torch.Tensor:
    return torch_sign_vector(K, sigma=sigma, domain=f"rot/signs/{stage}", device=device, dtype=dtype)


def rot_encode(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    sigma: int | bytes | str,
    stages: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode (A,B) -> (A',B') using a pseudorandom orthogonal rotation S.

    CPU/reference implementation: randomized Hadamard rotation.

    Requirements:
      - inner dimension K must be a power of two.
    """
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("A and B must be 2D tensors")
    M, K = A.shape
    K2, N = B.shape
    if K != K2:
        raise ValueError(f"Shape mismatch: A is {A.shape}, B is {B.shape}")
    if not is_power_of_two(int(K)):
        raise ValueError(f"K must be power-of-two for Hadamard rotation, got K={K}")
    if stages <= 0:
        raise ValueError("stages must be positive")

    device = A.device
    dtype = A.dtype
    inv_sqrtK = 1.0 / math.sqrt(float(K))

    A_rot = A
    for s in range(stages):
        # Multiply by H0 on the right: A <- A H0
        A_rot = fwht(A_rot, dim=1) * inv_sqrtK
        # Then multiply by D_s on the right: column-wise sign flip
        d = _stage_sign(sigma, int(K), s, device, dtype)
        A_rot = A_rot * d

    B_rot = B
    for s in range(stages):
        # First apply H0 on the left
        B_rot = fwht(B_rot, dim=0) * inv_sqrtK
        # Then apply D_s on the left (row-wise sign flip)
        d = _stage_sign(sigma, int(K), s, device, dtype)
        B_rot = d[:, None] * B_rot

    return A_rot, B_rot


def rot_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    sigma: int | bytes | str,
    config: RotGemmConfig = RotGemmConfig(),
) -> Tuple[torch.Tensor, bytes, dict]:
    """Compute RotGEMM output and transcript.

    CUDA fast path:
      - uses Triton TensorCore GEMM
      - fuses transcript hashing into the GEMM kernel epilogue
      - uses a cheap K-tile permutation gauge (AB-invariant) instead of explicit Hadamard encode
        (avoids explicit Hadamard encoding)

    CPU (or autograd) path:
      - uses randomized Hadamard encode + torch matmul + CPU trace reference.
    """
    use_fast = (
        triton_available()
        and A.device.type == "cuda"
        and B.device.type == "cuda"
        and (not A.requires_grad)
        and (not B.requires_grad)
        and A.dtype in (torch.float16, torch.bfloat16)
        and B.dtype in (torch.float16, torch.bfloat16)
        and gemm_with_fused_trace is not None
    )

    if use_fast:
        # Output is exactly A@B; permutation changes intermediate schedule only.
        C, digest, coords, meta2 = gemm_with_fused_trace(
            A, B,
            sigma=sigma,
            trace=config.trace,
            out_dtype=A.dtype,
            permute_ktiles=True,
            noise_scale=0.0,
        )
        meta = {
            "scheme": "rot_gemm",
            "stages": int(config.stages),
            "K": int(A.shape[1]),
            "trace_coords": coords,
            **meta2,
        }
        return C, digest, meta

    # Reference path
    A_rot, B_rot = rot_encode(A, B, sigma=sigma, stages=config.stages)
    C = A_rot @ B_rot
    transcript, coords = sampled_gemm_trace(A_rot, B_rot, sigma=sigma, config=config.trace)
    meta = {
        "scheme": "rot_gemm",
        "stages": int(config.stages),
        "trace_coords": coords,
        "K": int(A.shape[1]),
        "backend": "torch+cpu_trace",
    }
    if config.return_encoded:
        meta["A_rot"] = A_rot
        meta["B_rot"] = B_rot
    return C, transcript, meta
