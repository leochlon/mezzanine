"""pearl_pow_kernels

Experimental reference implementations for "GEMM + transcript hash" schemes.

Public API:
- rot_gemm.rot_gemm
- qnoise_gemm.qnoise_gemm
- fp4_gemm.fp4_scale_hash_gemm
- train_pow.PowLinear (drop-in nn.Module; can optionally return a transcript)
- hash128.TCHash128 (incremental 128-bit hash helper)
- trace.sampled_gemm_trace (hash intermediate GEMM tiles)
"""

from .hash128 import TCHash128, tc_hash128
from .rot_gemm import rot_gemm
from .qnoise_gemm import qnoise_gemm
from .fp4_gemm import fp4_scale_hash_gemm
from .train_pow import PowLinear

__all__ = [
    "TCHash128",
    "tc_hash128",
    "rot_gemm",
    "qnoise_gemm",
    "fp4_scale_hash_gemm",
    "PowLinear",
]
