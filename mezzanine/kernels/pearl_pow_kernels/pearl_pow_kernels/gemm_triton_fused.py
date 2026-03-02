
from __future__ import annotations

"""
Optional Triton implementation of GEMM with fused transcript hashing.

When Triton is available, this module can compute a sampled transcript hash inside the
same kernel that computes C = A @ B (hashing in the GEMM epilogue), so only a small
hash payload is written out.

This is primarily used by:
- `mezzanine/kernels/overhead_benchmark.py`
- the fast paths in `rot_gemm`, `qnoise_gemm`, and `fp4_gemm` wrappers
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from .rng import derive_seed_u64, rand_u64
from .hash128 import TCHash128
from .trace import TraceConfig

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


def triton_available() -> bool:
    return _TRITON_AVAILABLE


@dataclass(frozen=True)
class TritonGemmConfig:
    block_m: int = 128
    block_n: int = 128
    block_k: int = 32
    num_warps: int = 8
    num_stages: int = 4


@dataclass
class FusedGemmWorkspace:
    """Preallocated buffers for GEMM+hash launches.

    This is used to eliminate per-iteration allocations in benchmarks and
    production call sites.
    """

    C: torch.Tensor
    hash_out: torch.Tensor  # int32[S,4] (can be empty when S=0)
    sm: torch.Tensor        # int32[S]
    sn: torch.Tensor        # int32[S]
    coords3: List[Tuple[int, int, int]]
    block_m: int
    block_n: int
    block_k: int
    # Cached compile/runtime parameters for fast repeated launches.
    seed_u32: int
    permute_ktiles: bool
    perm_a: int
    perm_b: int
    noise_scale: float
    use_noise: bool
    num_warps: int
    num_stages: int


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


if _TRITON_AVAILABLE:  # pragma: no cover
    @triton.jit
    def _mix32(x):
        # 32-bit avalanche mix (murmur-like)
        x = x ^ (x >> 16)
        x = x * 0x7FEB352D
        x = x ^ (x >> 15)
        # NOTE: some Triton versions reject integer literals that do not fit
        # in signed int32. Use the two's-complement signed representation.
        # 0x846CA68B == 2221713035 (uint32) == -2073254261 (int32)
        x = x * (-2073254261)
        x = x ^ (x >> 16)
        return x

    # NOTE
    # ----
    # We intentionally do NOT use @triton.autotune here.
    #
    # Reason: environments vary in Triton version and launcher semantics. In some
    # releases, providing `num_warps` / `num_stages` at launch time while also
    # using autotune configs that set these values triggers:
    #   "got multiple values for keyword argument 'num_warps'".
    #
    # For a production-like benchmark/validator we prefer correctness and
    # portability. We therefore expose block sizes and warp/stage counts via
    # TritonGemmConfig and compile a kernel specialized to those parameters.
    @triton.jit
    def gemm_fused_hash_kernel(
        A_ptr, B_ptr, C_ptr,
        HASH_ptr,  # int32[NUM_SAMPLES,4] or empty
        SM_ptr, SN_ptr,  # int32[NUM_SAMPLES]
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        seed_u32,  # uint32-ish
        perm_a, perm_b,  # int32 (affine on k-tiles)
        noise_scale,  # float32 scalar
        PERMUTE_KTILES: tl.constexpr,
        USE_NOISE: tl.constexpr,
        NUM_SAMPLES: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(axis=0)
        pid_n = tl.program_id(axis=1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        k_tiles = tl.cdiv(K, BLOCK_K)

        # Cheap deterministic rank-1 noise for A (compiled out when USE_NOISE=False)
        if USE_NOISE:
            u = tl.randn(seed_u32, offs_m)  # [BM]
        else:
            u = 0.0

        # Loop over K tiles
        for kt in range(0, k_tiles):
            if PERMUTE_KTILES:
                # Requires k_tiles to be power-of-two for correctness; wrapper enforces.
                ktp = (perm_a * kt + perm_b) & (k_tiles - 1)
            else:
                ktp = kt
            k0 = ktp * BLOCK_K
            offs_k = k0 + tl.arange(0, BLOCK_K)

            a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
            b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

            a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0).to(tl.float16)
            b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0).to(tl.float16)

            if USE_NOISE:
                v = tl.randn(seed_u32 + 1337, offs_k)  # [BK]
                a = (a.to(tl.float32) + (u[:, None] * v[None, :]) * noise_scale).to(tl.float16)

            acc += tl.dot(a, b)

        # Store C
        c = acc.to(tl.float16)
        c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

        # Epilogue hash for sampled tiles only.
        #
        # Compatibility note: some Triton versions do not support direct constant
        # indexing into tl.tensor objects (e.g. `acc[0, 0]`). To keep the kernel
        # portable across Triton releases, we reload a few output scalars from C via
        # pointers. This still executes inside the same GEMM kernel and adds only a
        # handful of global loads.
        if NUM_SAMPLES > 0:
            m0 = pid_m * BLOCK_M
            n0 = pid_n * BLOCK_N
            p00 = C_ptr + (m0 + 0) * stride_cm + (n0 + 0) * stride_cn
            p01 = C_ptr + (m0 + 0) * stride_cm + (n0 + 1) * stride_cn
            p10 = C_ptr + (m0 + 1) * stride_cm + (n0 + 0) * stride_cn
            p11 = C_ptr + (m0 + 1) * stride_cm + (n0 + 1) * stride_cn
            x0 = tl.load(p00, mask=((m0 + 0) < M) & ((n0 + 0) < N), other=0.0).to(tl.float32)
            x1 = tl.load(p01, mask=((m0 + 0) < M) & ((n0 + 1) < N), other=0.0).to(tl.float32)
            x2 = tl.load(p10, mask=((m0 + 1) < M) & ((n0 + 0) < N), other=0.0).to(tl.float32)
            x3 = tl.load(p11, mask=((m0 + 1) < M) & ((n0 + 1) < N), other=0.0).to(tl.float32)
            v0 = tl.cast(x0 * 1024.0, tl.int32)
            v1 = tl.cast(x1 * 1024.0, tl.int32)
            v2 = tl.cast(x2 * 1024.0, tl.int32)
            v3 = tl.cast(x3 * 1024.0, tl.int32)

            pm = tl.cast(pid_m, tl.int32)
            pn = tl.cast(pid_n, tl.int32)
            # Use signed int32 equivalents to avoid out-of-range constants.
            # 0x9E3779B1 == 2654435761 (uint32) == -1640531535 (int32)
            # 0x85EBCA6B == 2246822507 (uint32) == -2048144789 (int32)
            key = _mix32(pm * (-1640531535) ^ pn * (-2048144789) ^ tl.cast(seed_u32, tl.int32))

            h0 = _mix32(v0 ^ key)
            h1 = _mix32(v1 ^ (key + 0x27D4EB2D))
            h2 = _mix32(v2 ^ (key ^ 0x165667B1))
            # 0x9E3779B9 == 2654435769 (uint32) == -1640531527 (int32)
            h3 = _mix32(v3 ^ (key + (-1640531527)))

            # Vectorized match against the sampled tile list.
            s_ids = tl.arange(0, NUM_SAMPLES)
            sm = tl.load(SM_ptr + s_ids).to(tl.int32)
            sn = tl.load(SN_ptr + s_ids).to(tl.int32)
            match = (pm == sm) & (pn == sn)
            base = HASH_ptr + s_ids * 4
            z = tl.zeros((NUM_SAMPLES,), dtype=tl.int32)
            tl.store(base + 0, z + h0, mask=match)
            tl.store(base + 1, z + h1, mask=match)
            tl.store(base + 2, z + h2, mask=match)
            tl.store(base + 3, z + h3, mask=match)


def _choose_perm_params(sigma: int | bytes | str, k_tiles: int) -> Tuple[int, int]:
    """Choose (a,b) for affine permutation kt -> (a*kt + b) mod k_tiles.

    Requires k_tiles to be a power of two; a must be odd.
    """
    if k_tiles <= 1:
        return 1, 0
    if (k_tiles & (k_tiles - 1)) != 0:
        return 1, 0
    seed = derive_seed_u64(sigma, domain="rot/perm")
    a = int((seed | 1) % k_tiles)
    if a == 0:
        a = 1
    b = int((seed >> 32) % k_tiles)
    return a, b


def _sample_tiles(
    M: int, N: int, *,
    block_m: int, block_n: int,
    num_samples: int,
    sigma: int | bytes | str,
) -> List[Tuple[int, int]]:
    m_tiles = _ceil_div(M, block_m)
    n_tiles = _ceil_div(N, block_n)
    total = m_tiles * n_tiles
    if total == 0 or num_samples <= 0:
        return []
    seed_u64 = derive_seed_u64(sigma, domain="trace/samples2d")
    u = rand_u64((num_samples, 2), seed_u64=seed_u64)
    coords: List[Tuple[int, int]] = []
    for i in range(num_samples):
        mi = int(u[i, 0] % m_tiles)
        ni = int(u[i, 1] % n_tiles)
        coords.append((mi, ni))
    # De-dup deterministic
    seen = set()
    out: List[Tuple[int, int]] = []
    for c in coords:
        if c not in seen:
            out.append(c)
            seen.add(c)
        if len(out) >= num_samples:
            break
    mi = 0
    ni = 0
    while len(out) < num_samples:
        out.append((mi % m_tiles, ni % n_tiles))
        ni += 1
        if ni >= n_tiles:
            ni = 0
            mi += 1
    return out


def gemm_with_fused_trace(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    sigma: int | bytes | str,
    trace: TraceConfig = TraceConfig(),
    out_dtype: torch.dtype = torch.float16,
    permute_ktiles: bool = False,
    noise_scale: float = 0.0,
    config: Optional[TritonGemmConfig] = None,
    return_digest: bool = True,
) -> Tuple[torch.Tensor, bytes, List[Tuple[int, int, int]], dict]:
    """Compute C = A@B and transcript hash in the SAME Triton kernel (CUDA only).

    The transcript hash is computed in the GEMM epilogue (microkernel-adjacent).

    Returns (C, digest_16B, coords3d, meta).
    """
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available. Install triton and run on CUDA.")

    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("A and B must be 2D")
    M, K = A.shape
    K2, N = B.shape
    if K != K2:
        raise ValueError(f"Shape mismatch: A {A.shape}, B {B.shape}")

    if A.device.type != "cuda" or B.device.type != "cuda":
        raise ValueError("gemm_with_fused_trace requires CUDA tensors")

    # Kernel expects fp16/bf16 inputs.
    if A.dtype not in (torch.float16, torch.bfloat16):
        A = A.to(torch.float16)
    if B.dtype not in (torch.float16, torch.bfloat16):
        B = B.to(torch.float16)

    cfg = config or TritonGemmConfig()
    BM, BN, BK = int(cfg.block_m), int(cfg.block_n), int(cfg.block_k)

    # Permutation: we permute *K-tiles* (block schedule), not individual k indices.
    k_tiles = _ceil_div(int(K), BK)
    permute = bool(permute_ktiles) and (K % BK == 0) and ((k_tiles & (k_tiles - 1)) == 0)
    perm_a, perm_b = (1, 0)
    if permute:
        perm_a, perm_b = _choose_perm_params(sigma, k_tiles)

    # Samples are in output tile space defined by (BM, BN).
    S = int(trace.num_samples)
    coords2 = _sample_tiles(int(M), int(N), block_m=BM, block_n=BN, num_samples=S, sigma=sigma)
    coords3 = [(mi, ni, 0) for (mi, ni) in coords2]

    if S > 0:
        sm = torch.tensor([c[0] for c in coords2], device=A.device, dtype=torch.int32)
        sn = torch.tensor([c[1] for c in coords2], device=A.device, dtype=torch.int32)
        hash_out = torch.zeros((S, 4), device=A.device, dtype=torch.int32)
    else:
        sm = torch.empty((0,), device=A.device, dtype=torch.int32)
        sn = torch.empty((0,), device=A.device, dtype=torch.int32)
        hash_out = torch.empty((0, 4), device=A.device, dtype=torch.int32)

    C = torch.empty((M, N), device=A.device, dtype=torch.float16)

    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))

    seed_u64 = derive_seed_u64(sigma, domain="triton/seed")
    seed_u32 = int(seed_u64 & 0xFFFFFFFF)

    use_noise = float(noise_scale) != 0.0

    gemm_fused_hash_kernel[grid](
        A, B, C,
        hash_out, sm, sn,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        seed_u32,
        int(perm_a), int(perm_b),
        float(noise_scale),
        PERMUTE_KTILES=permute,
        USE_NOISE=use_noise,
        NUM_SAMPLES=S,
        BLOCK_M=BM,
        BLOCK_N=BN,
        BLOCK_K=BK,
        num_warps=int(cfg.num_warps),
        num_stages=int(cfg.num_stages),
    )

    C_out = C.to(out_dtype)

    if return_digest:
        payload = hash_out.detach().cpu().numpy().tobytes(order="C") if S > 0 else b""
        digest = TCHash128(seed=sigma).update_bytes(payload).digest()
    else:
        # Avoid any device->host transfer in the hot path.
        digest = b"\x00" * 16

    meta = {
        "backend": "triton",
        "block_m": BM,
        "block_n": BN,
        "block_k": BK,
        "permute_ktiles": bool(permute),
        "perm_a": int(perm_a),
        "perm_b": int(perm_b),
        "noise_scale": float(noise_scale),
        "trace_coords": coords3,
    }
    return C_out, digest, coords3, meta


def prepare_fused_workspace(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    sigma: int | bytes | str,
    trace: TraceConfig,
    permute_ktiles: bool = False,
    noise_scale: float = 0.0,
    config: Optional[TritonGemmConfig] = None,
) -> FusedGemmWorkspace:
    """Prepare a reusable workspace for repeated GEMM+hash launches.

    This function performs all one-time host-side work (sampling, tensor
    allocations, seed derivation, permutation parameter selection) so that the
    hot loop can be a single Triton kernel launch.
    """
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available. Install triton and run on CUDA.")
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("A and B must be 2D")
    M, K = A.shape
    K2, N = B.shape
    if K != K2:
        raise ValueError(f"Shape mismatch: A {A.shape}, B {B.shape}")
    if A.device.type != "cuda" or B.device.type != "cuda":
        raise ValueError("prepare_fused_workspace requires CUDA tensors")

    cfg = config or TritonGemmConfig()
    BM, BN, BK = int(cfg.block_m), int(cfg.block_n), int(cfg.block_k)

    # K-tile permutation parameters.
    k_tiles = _ceil_div(int(K), BK)
    permute = bool(permute_ktiles) and (K % BK == 0) and ((k_tiles & (k_tiles - 1)) == 0)
    perm_a, perm_b = (1, 0)
    if permute:
        perm_a, perm_b = _choose_perm_params(sigma, k_tiles)

    # Tile sampling (in output tile space).
    S = int(trace.num_samples)
    coords2 = _sample_tiles(int(M), int(N), block_m=BM, block_n=BN, num_samples=S, sigma=sigma)
    coords3 = [(mi, ni, 0) for (mi, ni) in coords2]

    if S > 0:
        sm = torch.tensor([c[0] for c in coords2], device=A.device, dtype=torch.int32)
        sn = torch.tensor([c[1] for c in coords2], device=A.device, dtype=torch.int32)
        hash_out = torch.zeros((S, 4), device=A.device, dtype=torch.int32)
    else:
        sm = torch.empty((0,), device=A.device, dtype=torch.int32)
        sn = torch.empty((0,), device=A.device, dtype=torch.int32)
        hash_out = torch.empty((0, 4), device=A.device, dtype=torch.int32)

    # Output buffer. We store fp16 in-kernel; callers can cast if needed.
    C = torch.empty((M, N), device=A.device, dtype=torch.float16)

    seed_u64 = derive_seed_u64(sigma, domain="triton/seed")
    seed_u32 = int(seed_u64 & 0xFFFFFFFF)
    use_noise = float(noise_scale) != 0.0

    return FusedGemmWorkspace(
        C=C,
        hash_out=hash_out,
        sm=sm,
        sn=sn,
        coords3=coords3,
        block_m=BM,
        block_n=BN,
        block_k=BK,
        seed_u32=seed_u32,
        permute_ktiles=bool(permute),
        perm_a=int(perm_a),
        perm_b=int(perm_b),
        noise_scale=float(noise_scale),
        use_noise=bool(use_noise),
        num_warps=int(cfg.num_warps),
        num_stages=int(cfg.num_stages),
    )


def launch_gemm_fused_inplace(
    A: torch.Tensor,
    B: torch.Tensor,
    ws: FusedGemmWorkspace,
) -> None:
    """Launch the GEMM+hash kernel writing into the preallocated workspace."""
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available. Install triton and run on CUDA.")
    if A.device.type != "cuda" or B.device.type != "cuda":
        raise ValueError("launch_gemm_fused_inplace requires CUDA tensors")
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("A and B must be 2D")
    M, K = A.shape
    K2, N = B.shape
    if K != K2:
        raise ValueError(f"Shape mismatch: A {A.shape}, B {B.shape}")
    if ws.C.shape != (M, N):
        raise ValueError(f"Workspace C has shape {ws.C.shape}, expected {(M, N)}")

    BM, BN, BK = ws.block_m, ws.block_n, ws.block_k
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    S = int(ws.sm.numel())

    gemm_fused_hash_kernel[grid](
        A, B, ws.C,
        ws.hash_out, ws.sm, ws.sn,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        ws.C.stride(0), ws.C.stride(1),
        int(ws.seed_u32),
        int(ws.perm_a), int(ws.perm_b),
        float(ws.noise_scale),
        PERMUTE_KTILES=bool(ws.permute_ktiles),
        USE_NOISE=bool(ws.use_noise),
        NUM_SAMPLES=S,
        BLOCK_M=int(BM),
        BLOCK_N=int(BN),
        BLOCK_K=int(BK),
        num_warps=int(ws.num_warps),
        num_stages=int(ws.num_stages),
    )


def gemm_triton(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    out_dtype: torch.dtype = torch.float16,
    permute_ktiles: bool = False,
    noise_scale: float = 0.0,
    config: Optional[TritonGemmConfig] = None,
) -> Tuple[torch.Tensor, dict]:
    """Baseline Triton GEMM using the *same kernel schedule* as the fused-trace path.

    Returns (C, meta). No device->host transfers.
    """
    C, _digest, _coords, meta = gemm_with_fused_trace(
        A, B,
        sigma=0,
        trace=TraceConfig(num_samples=0),
        out_dtype=out_dtype,
        permute_ktiles=permute_ktiles,
        noise_scale=noise_scale,
        config=config,
        return_digest=False,
    )
    return C, meta
