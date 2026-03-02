#!/usr/bin/env python3
"""
paged_gqa_decode_triton.py

Single-file, user-facing module: ** Triton kernel** for decode-time attention
over a **paged KV cache** with Grouped Query Attention (GQA).

What this is
------------
- Decode attention only (query length = 1).
- KV stored in fixed-size blocks (paged KV cache).
- Supports GQA: Hq query heads, Hkv KV heads, with group size g = Hq / Hkv.
- Optional **schedule gauge**: pass a permutation of logical block indices to
  traverse blocks in an order that improves locality (e.g., physical-sorted).

What this is NOT
----------------
- Not a full LLM runtime, not a prefill kernel.
- Not a cache allocator. You provide block_table and the block tensors.

Minimum API
-----------
- PagedKVCache: tiny container for the paged KV tensors + mapping.
- build_perm_lbi(...): precompute traversal schedule (logical or physical_sorted).
- paged_gqa_decode(...): call the single Triton kernel.

Design notes
------------
- Uses online softmax accumulation (stable).
- Uses tl.dot for score and value accumulation.
- For small group sizes (e.g. g=8), pads the group dimension to 16 internally
  to stay on the tl.dot (tensor-core) path while only storing valid heads.

Typical usage
-------------
>>> import torch
>>> from paged_gqa_decode_triton import PagedKVCache, build_perm_lbi, paged_gqa_decode
>>> cache = PagedKVCache(k_blocks, v_blocks, block_table, ctx_lens, block_size=16, layout="bhsd")
>>> perm = build_perm_lbi(cache.block_table, cache.ctx_lens, cache.block_size, order="physical_sorted")
>>> out = paged_gqa_decode(q_bhd, cache, perm_lbi=perm, blocks_per_iter=16)

All tensors must be on CUDA.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Dict

import math
import torch

try:
    import triton
    import triton.language as tl
except Exception as e:  # pragma: no cover
    triton = None
    tl = None
    _TRITON_IMPORT_ERROR = e


# -----------------------------
# Public data structure
# -----------------------------


@dataclass(frozen=True)
class PagedKVCache:
    """
    Paged KV cache container.

    k, v:
      - layout="bhsd": [n_blocks_total, Hkv, block_size, D]
      - layout="bshd": [n_blocks_total, block_size, Hkv, D]

    block_table:
      int32 tensor [B, max_blocks_in_batch], mapping logical block index -> physical block id

    ctx_lens:
      int32 tensor [B], context length (number of valid tokens) per request

    block_size:
      int, tokens per block
    """

    k: torch.Tensor
    v: torch.Tensor
    block_table: torch.Tensor
    ctx_lens: torch.Tensor
    block_size: int
    layout: Literal["bhsd", "bshd"] = "bhsd"

    @property
    def device(self) -> torch.device:
        return self.k.device

    @property
    def Hkv(self) -> int:
        if self.layout == "bhsd":
            return int(self.k.shape[1])
        return int(self.k.shape[2])

    @property
    def D(self) -> int:
        return int(self.k.shape[-1])

    def strides(self) -> Dict[str, int]:
        """
        Return element-strides for K/V blocks and block_table.
        """
        # K/V block strides
        #   bhsd: [blk, H, S, D]
        #   bshd: [blk, S, H, D]
        if self.layout == "bhsd":
            k_blk, k_h, k_s, k_d = self.k.stride()
            v_blk, v_h, v_s, v_d = self.v.stride()
        else:
            k_blk, k_s, k_h, k_d = self.k.stride()
            v_blk, v_s, v_h, v_d = self.v.stride()

        bt_b, bt_i = self.block_table.stride()
        return {
            "k_blk": int(k_blk),
            "k_s": int(k_s),
            "k_h": int(k_h),
            "k_d": int(k_d),
            "v_blk": int(v_blk),
            "v_s": int(v_s),
            "v_h": int(v_h),
            "v_d": int(v_d),
            "bt_b": int(bt_b),
            "bt_i": int(bt_i),
        }


# -----------------------------
# Schedule precomputation
# -----------------------------


def build_perm_lbi(
    block_table: torch.Tensor,
    ctx_lens: torch.Tensor,
    block_size: int,
    *,
    order: Literal["logical", "physical_sorted"] = "physical_sorted",
) -> torch.Tensor:
    """
    Build PERM_LBI[b, i] = logical block index visited at traversal index i.

    - order="logical": perm[b, i] = i
    - order="physical_sorted": perm[b, :] is argsort(block_table[b, :nblocks])

    Returns:
      int32 tensor [B, max_blocks]
    """
    assert block_table.ndim == 2 and block_table.dtype == torch.int32
    assert ctx_lens.ndim == 1 and ctx_lens.dtype == torch.int32
    B, max_blocks = block_table.shape
    device = block_table.device

    if order == "logical":
        base = torch.arange(max_blocks, device=device, dtype=torch.int32)
        return base[None, :].expand(B, max_blocks).contiguous()

    # physical_sorted
    perm = torch.zeros((B, max_blocks), device=device, dtype=torch.int32)
    # NOTE: This is an offline/precompute operation; Python loop is OK.
    for b in range(B):
        ctx = int(ctx_lens[b].item())
        nblocks = (ctx + (block_size - 1)) // block_size
        if nblocks <= 0:
            continue
        phys = block_table[b, :nblocks].to(torch.int64)
        lbi_sorted = torch.argsort(phys).to(torch.int32)
        perm[b, :nblocks] = lbi_sorted
    return perm


# -----------------------------
# Convenience: allocate a contiguous paged KV cache
# -----------------------------


@torch.no_grad()
def allocate_paged_kv_cache(
    *,
    B: int,
    Hkv: int,
    D: int,
    max_seq: int,
    block_size: int = 16,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | str = "cuda",
    layout: Literal["bhsd", "bshd"] = "bhsd",
) -> PagedKVCache:
    """
    Allocate a simple contiguous paged KV cache.

    This is a *convenience* allocator for integration/testing:
      - physical blocks are laid out contiguously per batch element
      - block_table[b, lbi] = b * nblocks + lbi

    Args:
      max_seq: maximum context length to support for this cache
    """
    assert B > 0 and Hkv > 0 and D > 0 and max_seq >= 0
    nblocks = (max_seq + block_size - 1) // block_size if max_seq > 0 else 0
    total_blocks = max(1, B * max(1, nblocks))

    if layout == "bhsd":
        k = torch.empty((total_blocks, Hkv, block_size, D), device=device, dtype=dtype)
        v = torch.empty((total_blocks, Hkv, block_size, D), device=device, dtype=dtype)
    else:
        k = torch.empty((total_blocks, block_size, Hkv, D), device=device, dtype=dtype)
        v = torch.empty((total_blocks, block_size, Hkv, D), device=device, dtype=dtype)

    block_table = torch.empty((B, max(1, nblocks)), device=device, dtype=torch.int32)
    for b in range(B):
        base = b * max(1, nblocks)
        block_table[b, :] = torch.arange(
            base, base + max(1, nblocks), device=device, dtype=torch.int32
        )

    ctx_lens = torch.zeros((B,), device=device, dtype=torch.int32)
    return PagedKVCache(
        k=k,
        v=v,
        block_table=block_table,
        ctx_lens=ctx_lens,
        block_size=block_size,
        layout=layout,
    )


# -----------------------------
# Single Triton kernel
# -----------------------------


@triton.jit
def _paged_gqa_decode_kernel(
    Q_ptr,  # [B, Hq, D]
    K_ptr,  # paged blocks
    V_ptr,
    BT_ptr,  # [B, max_blocks]
    PERM_ptr,  # [B, max_blocks]
    CTX_ptr,  # [B]
    BIDS_ptr,  # [B_bucket] batch ids
    O_ptr,  # [B, Hq, D]
    # strides (in elements)
    stride_qb: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kblk: tl.constexpr,
    stride_ks: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_vblk: tl.constexpr,
    stride_vs: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vd: tl.constexpr,
    stride_btb: tl.constexpr,
    stride_bti: tl.constexpr,
    stride_pb: tl.constexpr,
    stride_pi: tl.constexpr,
    stride_ob: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_od: tl.constexpr,
    # params
    Hkv: tl.constexpr,
    G: tl.constexpr,
    GQ: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCKS_PER_ITER: tl.constexpr,
    D: tl.constexpr,
    MAX_BLOCKS,  # runtime loop bound
    SCALE: tl.constexpr,
    LAYOUT_BHSD: tl.constexpr,  # 1 if bhsd else 0
    USE_BF16: tl.constexpr,  # 1 if bf16 else 0
):
    """
    One program handles one (bucket_index, kv_head), producing G query heads.
    Internally pads the group dimension to GQ >= 16 so tl.dot stays valid.
    Only valid heads are stored.
    """
    pid = tl.program_id(0)
    b_idx = pid // Hkv
    kvh = pid - b_idx * Hkv
    b = tl.load(BIDS_ptr + b_idx).to(tl.int32)
    b_i64 = b.to(tl.int64)  # avoid int32 overflow in pointer arithmetic for large KV

    ctx = tl.load(CTX_ptr + b).to(tl.int32)
    is_empty = ctx == 0
    # For ctx==0, we still execute a stable code path but force output to zeros at the end.
    ctx_eff = tl.where(is_empty, 1, ctx)
    nblocks = (ctx_eff + (BLOCK_SIZE - 1)) // BLOCK_SIZE

    # Query heads for this kv head: hq = kvh*G + t
    t = tl.arange(0, GQ)
    mask_t = t < G
    hq = kvh * G + t  # NOTE: hq may be OOB for t>=G, but masked load/store is used.
    hq_i64 = hq.to(tl.int64)

    offs_d = tl.arange(0, D)
    tl.multiple_of(offs_d, 8)
    tl.max_contiguous(offs_d, 16)

    q_ptrs = (
        Q_ptr
        + b_i64 * stride_qb
        + hq_i64[:, None] * stride_qh
        + offs_d[None, :] * stride_qd
    )
    # Invalid heads (t>=G) are loaded as zeros.
    q = tl.load(q_ptrs, mask=mask_t[:, None], other=0.0)  # [GQ, D]

    # Online softmax state
    m = tl.full([GQ], -float("inf"), tl.float32)
    l_sum = tl.zeros([GQ], tl.float32)
    acc = tl.zeros([GQ, D], tl.float32)

    # Token offsets for a tile of BLOCKS_PER_ITER blocks.
    offs_n = tl.arange(0, BLOCK_SIZE * BLOCKS_PER_ITER)
    blk_in_tile = offs_n // BLOCK_SIZE
    pos_in_blk = offs_n - blk_in_tile * BLOCK_SIZE

    for tbi_base in range(0, MAX_BLOCKS, BLOCKS_PER_ITER):
        tbi = tbi_base + blk_in_tile
        tbi_i64 = tbi.to(tl.int64)
        blk_ok = tbi < nblocks

        # Traversal schedule (logical block indices), then gather physical ids via block_table.
        lbi = tl.load(
            PERM_ptr + b_i64 * stride_pb + tbi_i64 * stride_pi,
            mask=blk_ok,
            other=0,
        ).to(tl.int32)
        lbi_i64 = lbi.to(tl.int64)

        phys = tl.load(
            BT_ptr + b_i64 * stride_btb + lbi_i64 * stride_bti,
            mask=blk_ok,
            other=0,
        ).to(tl.int32)
        phys_i64 = phys.to(tl.int64)

        start = lbi_i64 * BLOCK_SIZE
        abs_pos = start + pos_in_blk
        mask_n = blk_ok & (abs_pos < ctx_eff)

        # Pointers [BLOCK_N, D]
        if LAYOUT_BHSD:
            k_ptrs = (
                K_ptr
                + phys_i64[:, None] * stride_kblk
                + kvh * stride_kh
                + pos_in_blk[:, None] * stride_ks
                + offs_d[None, :] * stride_kd
            )
            v_ptrs = (
                V_ptr
                + phys_i64[:, None] * stride_vblk
                + kvh * stride_vh
                + pos_in_blk[:, None] * stride_vs
                + offs_d[None, :] * stride_vd
            )
        else:
            k_ptrs = (
                K_ptr
                + phys_i64[:, None] * stride_kblk
                + pos_in_blk[:, None] * stride_ks
                + kvh * stride_kh
                + offs_d[None, :] * stride_kd
            )
            v_ptrs = (
                V_ptr
                + phys_i64[:, None] * stride_vblk
                + pos_in_blk[:, None] * stride_vs
                + kvh * stride_vh
                + offs_d[None, :] * stride_vd
            )

        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)  # [BLOCK_N, D]
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

        # scores: [GQ, BLOCK_N]
        scores = tl.dot(q, tl.trans(k)).to(tl.float32) * SCALE
        scores = tl.where(mask_n[None, :], scores, -float("inf"))

        m_blk = tl.max(scores, axis=1)
        m_new = tl.maximum(m, m_blk)

        exp_scores = tl.exp(scores - m_new[:, None])
        l_sum_new = l_sum * tl.exp(m - m_new) + tl.sum(exp_scores, axis=1)

        alpha = tl.exp(m - m_new)

        exp_cast = exp_scores.to(tl.bfloat16) if USE_BF16 else exp_scores.to(tl.float16)
        acc = acc * alpha[:, None] + tl.dot(exp_cast, v).to(tl.float32)

        m = m_new
        l_sum = l_sum_new

    out = acc / l_sum[:, None]

    # If the request has an empty cache (ctx==0), define output as zeros.
    out = tl.where(is_empty, 0.0, out)

    out_ptrs = (
        O_ptr
        + b_i64 * stride_ob
        + hq_i64[:, None] * stride_oh
        + offs_d[None, :] * stride_od
    )
    # Only store valid query heads.
    tl.store(
        out_ptrs,
        out.to(tl.bfloat16) if USE_BF16 else out.to(tl.float16),
        mask=mask_t[:, None],
    )


# -----------------------------
# User-facing wrapper
# -----------------------------


@torch.no_grad()
def paged_gqa_decode(
    q: torch.Tensor,  # [B, Hq, D]
    cache: PagedKVCache,
    *,
    perm_lbi: Optional[torch.Tensor] = None,  # [B, max_blocks] int32
    batch_ids: Optional[torch.Tensor] = None,  # [B_bucket] int32, defaults to arange(B)
    out: Optional[torch.Tensor] = None,
    blocks_per_iter: int = 16,
    max_blocks: Optional[int] = None,
    num_warps: int = 4,
    num_stages: int = 2,
) -> torch.Tensor:
    """
    Decode attention over a paged KV cache (GQA grouped by KV head).

    Args:
      q: [B, Hq, D] bf16/fp16
      cache: PagedKVCache containing K/V blocks and mappings
      perm_lbi: optional traversal schedule (logical block ids). If None, uses logical order.
      batch_ids: optional subset of batch indices to compute; defaults to [0..B-1].
      blocks_per_iter: traversal tile size in blocks (typical 8..16). Controls BLOCK_N.
      max_blocks: runtime loop bound. If None, uses perm_lbi.shape[1] (or cache.block_table.shape[1]).
    """
    if triton is None:
        raise RuntimeError(f"Triton import failed: {_TRITON_IMPORT_ERROR}")

    assert q.is_cuda and cache.k.is_cuda and cache.v.is_cuda
    assert q.dtype in (torch.float16, torch.bfloat16)
    assert cache.k.dtype == q.dtype and cache.v.dtype == q.dtype
    assert (
        cache.block_table.dtype == torch.int32 and cache.ctx_lens.dtype == torch.int32
    )
    assert cache.layout in ("bhsd", "bshd")

    B, Hq, D = q.shape
    assert D == cache.D, f"q.D={D} but cache.D={cache.D}"
    Hkv = cache.Hkv
    assert Hq % Hkv == 0, f"Hq={Hq} must be divisible by Hkv={Hkv}"
    G = Hq // Hkv
    GQ = 16 if G < 16 else G  # internal pad for tl.dot

    if batch_ids is None:
        batch_ids = torch.arange(B, device=q.device, dtype=torch.int32)
    else:
        assert batch_ids.dtype == torch.int32 and batch_ids.is_cuda

    if perm_lbi is None:
        perm_lbi = build_perm_lbi(
            cache.block_table, cache.ctx_lens, cache.block_size, order="logical"
        )
    else:
        assert perm_lbi.dtype == torch.int32 and perm_lbi.is_cuda
        assert perm_lbi.shape == cache.block_table.shape

    if max_blocks is None:
        max_blocks = int(perm_lbi.shape[1])
    assert isinstance(max_blocks, int) and max_blocks > 0

    # Output
    if out is None:
        out = torch.empty_like(q)
    else:
        assert out.shape == q.shape and out.dtype == q.dtype and out.device == q.device

    st = cache.strides()
    layout_bhsd = 1 if cache.layout == "bhsd" else 0
    use_bf16 = 1 if q.dtype == torch.bfloat16 else 0

    B_bucket = int(batch_ids.numel())
    grid = (B_bucket * Hkv,)

    _paged_gqa_decode_kernel[grid](
        q,
        cache.k,
        cache.v,
        cache.block_table,
        perm_lbi,
        cache.ctx_lens,
        batch_ids,
        out,
        stride_qb=q.stride(0),
        stride_qh=q.stride(1),
        stride_qd=q.stride(2),
        stride_kblk=st["k_blk"],
        stride_ks=st["k_s"],
        stride_kh=st["k_h"],
        stride_kd=st["k_d"],
        stride_vblk=st["v_blk"],
        stride_vs=st["v_s"],
        stride_vh=st["v_h"],
        stride_vd=st["v_d"],
        stride_btb=st["bt_b"],
        stride_bti=st["bt_i"],
        stride_pb=perm_lbi.stride(0),
        stride_pi=perm_lbi.stride(1),
        stride_ob=out.stride(0),
        stride_oh=out.stride(1),
        stride_od=out.stride(2),
        Hkv=Hkv,
        G=G,
        GQ=GQ,
        BLOCK_SIZE=cache.block_size,
        BLOCKS_PER_ITER=blocks_per_iter,
        D=D,
        MAX_BLOCKS=max_blocks,
        SCALE=1.0 / math.sqrt(D),
        LAYOUT_BHSD=layout_bhsd,
        USE_BF16=use_bf16,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


# -----------------------------
# Tiny helper: write new KV (no extra kernels)
# -----------------------------


@torch.no_grad()
def write_kv_token_(
    cache: PagedKVCache,
    *,
    b: int,
    pos: int,
    k_new: torch.Tensor,  # [Hkv, D] in cache.dtype/device
    v_new: torch.Tensor,  # [Hkv, D]
) -> None:
    """
    Write a single token's KV into the paged cache (in-place), using torch indexing only.

    This is intended as a simple integration helper (decode step update).
    For production, you may want a fused write kernel, but that would be a second Triton kernel.

    Args:
      b: batch index
      pos: absolute token position in the sequence (0-based)
      k_new, v_new: [Hkv, D] for this token
    """
    assert cache.block_table.dtype == torch.int32
    assert cache.layout in ("bhsd", "bshd")

    block_size = cache.block_size
    lbi = pos // block_size
    off = pos - lbi * block_size
    phys = int(cache.block_table[b, lbi].item())

    if cache.layout == "bhsd":
        # [blk, H, S, D]
        cache.k[phys, :, off, :] = k_new
        cache.v[phys, :, off, :] = v_new
    else:
        # [blk, S, H, D]
        cache.k[phys, off, :, :] = k_new
        cache.v[phys, off, :, :] = v_new

    # Update ctx_len if needed (caller can batch-update for speed).
    if int(cache.ctx_lens[b].item()) < pos + 1:
        cache.ctx_lens[b] = int(pos + 1)
