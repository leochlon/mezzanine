#!/usr/bin/env python3
"""bench_wow_hf_hero_unsloth_style.py

Unsloth-style hero benchmark for **decode-time attention** over a **paged KV cache**.

This script is intentionally *presentation-first*:
- shows **Step time vs context length** (lower is better)
- shows **Peak VRAM vs context length** (lower is better)
- summarizes per-model **speedup**, **VRAM reduction**, and **"longer context"** (max T before baseline OOM)

It is designed to pair with the single-kernel module:
  paged_gqa_decode_triton.py

Before vs After
---------------
Before (common in paged-KV systems):
  PACK(paged -> dense) + dense attention backend (FlashAttn if available else SDPA)

After:
  native paged decode attention (no pack)

Outputs
-------
outdir/
  results.csv / results.json
  hero.png / hero.pdf
  hero_table.csv

Example
-------
python bench_wow_hf_hero_unsloth_style.py \
  --ours_module paged_gqa_decode_triton \
  --models TinyLlama/TinyLlama-1.1B-Chat-v1.0,Qwen/Qwen2-1.5B-Instruct,mistralai/Mistral-7B-Instruct-v0.2,Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --seqlens 512,1024,2048,4096,8192,16384,32768 \
  --hero_T 8192 \
  --dtype bf16 --block 16 --bpi 16 \
  --mem_gb 28 \
  --warmup 20 --iters 80 \
  --outdir out_wow

Notes
-----
- We only download HF *configs* (fast). We do **not** download weights.
- Baseline will show OOM at large contexts (that is part of the story).
- Memory is reported using torch CUDA peak stats (reserved + allocated).
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import re
import statistics
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

try:  # torch is only required for full benchmark runs
    import torch
except Exception:  # pragma: no cover
    torch = None

if torch is None:

    def no_grad():  # type: ignore
        def deco(fn):
            return fn

        return deco
else:
    no_grad = torch.no_grad


def _require(pkg: str, err: Exception) -> None:
    raise RuntimeError(
        f"Missing dependency: {pkg}. Install with: pip install {pkg}\n"
        f"Original error: {err}"
    )


try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as e:  # pragma: no cover
    _require("matplotlib", e)


try:
    import pandas as pd
except Exception as e:  # pragma: no cover
    _require("pandas", e)


try:
    from transformers import AutoConfig
except Exception:  # pragma: no cover
    AutoConfig = None


# -----------------------------
# Plot style (Unsloth-ish)
# -----------------------------


def apply_unsloth_style() -> Dict[str, str]:
    """Apply a clean Unsloth-like plotting style."""
    palette = {
        "ink": "#0B0F0E",
        "slate": "#3B3F3E",
        "bg": "#FFFFFF",
        "panel": "#FFFFFF",
        "grid": "#E6E6E6",
        "accent": "#14B789",  # teal/green
        "accent_dark": "#0F8B6A",
        "baseline": "#6B6F73",  # darker gray for dashed lines
        "baseline_light": "#C9D1D6",
    }

    plt.rcParams.update(
        {
            "figure.facecolor": palette["bg"],
            "savefig.facecolor": palette["bg"],
            "axes.facecolor": palette["panel"],
            "axes.edgecolor": palette["grid"],
            "axes.labelcolor": palette["ink"],
            "axes.titleweight": "semibold",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.color": palette["slate"],
            "ytick.color": palette["slate"],
            "text.color": palette["ink"],
            "grid.color": palette["grid"],
            "grid.linewidth": 0.8,
            "grid.alpha": 1.0,
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Sora",
                "Space Grotesk",
                "IBM Plex Sans",
                "DejaVu Sans",
            ],
        }
    )
    return palette


# -----------------------------
# Utilities
# -----------------------------


def dtype_from_str(s: str) -> torch.dtype:
    s = s.lower()
    if s == "bf16":
        return torch.bfloat16
    if s in ("fp16", "float16"):
        return torch.float16
    raise ValueError(f"Unsupported dtype: {s} (use bf16 or fp16)")


def short_model_label(model_id: str) -> str:
    name = model_id.split("/")[-1]
    name = re.sub(r"-Instruct.*$", "", name)
    name = re.sub(r"-Chat.*$", "", name)
    name = re.sub(r"-v\d+.*$", "", name)
    return name


def try_import_flash_attn_with_kvcache():
    try:
        from flash_attn import flash_attn_with_kvcache  # type: ignore

        return flash_attn_with_kvcache
    except Exception:
        pass
    try:
        from flash_attn.flash_attn_interface import flash_attn_with_kvcache  # type: ignore

        return flash_attn_with_kvcache
    except Exception:
        return None


def call_flash_attn_kvcache(
    flash_fn,
    q_bhd: torch.Tensor,  # [B,Hq,D]
    k_bthd: torch.Tensor,  # [B,T,Hkv,D]
    v_bthd: torch.Tensor,
    seqlens: torch.Tensor,  # [B] int32
    *,
    causal: bool = True,
) -> torch.Tensor:
    """Probe a few flash-attn call patterns across versions."""
    q_b1hd = q_bhd[:, None, :, :]
    seqlens_i32 = seqlens.to(torch.int32)
    patterns = [
        {"causal": causal, "cache_seqlens": seqlens_i32},
        {"cache_seqlens": seqlens_i32},
        {"causal": causal, "seqlens": seqlens_i32},
        {"seqlens": seqlens_i32},
    ]
    for kw in patterns:
        try:
            out = flash_fn(q_b1hd, k_bthd, v_bthd, **kw)
            return out
        except (TypeError, RuntimeError):
            pass

    # cu_seqlens fallback
    B = q_bhd.shape[0]
    cu = torch.zeros((B + 1,), device=q_bhd.device, dtype=torch.int32)
    cu[1:] = torch.cumsum(seqlens.to(torch.int32), dim=0)
    T = k_bthd.shape[1]
    try:
        return flash_fn(
            q_b1hd, k_bthd, v_bthd, causal=causal, cu_seqlens_k=cu, max_seqlen_k=T
        )
    except (TypeError, RuntimeError):
        return flash_fn(q_b1hd, k_bthd, v_bthd, cu_seqlens_k=cu, max_seqlen_k=T)


def bench_cuda_ms(
    fn: Callable[[], torch.Tensor], *, warmup: int, iters: int
) -> Tuple[float, float, float]:
    """Return (median_ms, cv, mean_ms) using CUDA events."""
    torch.cuda.synchronize()
    for _ in range(warmup):
        _ = fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times: List[float] = []
    for _ in range(iters):
        start.record()
        out = fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
        if not torch.is_tensor(out):
            raise RuntimeError("Benchmark function must return a tensor")

    med = statistics.median(times)
    mean = statistics.fmean(times)
    stdev = statistics.pstdev(times) if len(times) > 1 else 0.0
    cv = stdev / mean if mean > 0 else 0.0
    return med, cv, mean


def measure_peak_mem_bytes(fn: Callable[[], torch.Tensor]) -> Tuple[int, int]:
    """Return (peak_alloc_bytes, peak_reserved_bytes) for one run of fn."""
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    out = fn()
    torch.cuda.synchronize()
    if not torch.is_tensor(out):
        raise RuntimeError("Function must return a tensor")
    alloc = int(torch.cuda.max_memory_allocated())
    reserv = int(torch.cuda.max_memory_reserved())
    return alloc, reserv


def bytes_to_gb(x: int) -> float:
    return float(x) / (1024.0**3)


def is_oom(e: BaseException) -> bool:
    msg = str(e).lower()
    return "out of memory" in msg or "cuda error: out of memory" in msg


# -----------------------------
# Model spec
# -----------------------------


def _get_first(
    cfg: Any, keys: Sequence[str], default: Optional[int] = None
) -> Optional[int]:
    for k in keys:
        if hasattr(cfg, k):
            v = getattr(cfg, k)
            if isinstance(v, int) and v > 0:
                return v
    return default


@dataclass
class ModelSpec:
    model_id: str
    label: str
    approx_params_b: float
    n_layers: int
    max_seq: int
    Hq: int
    Hkv: int
    D: int
    g: int


def derive_model_spec(model_id: str) -> ModelSpec:
    if AutoConfig is None:
        raise RuntimeError("transformers is required to derive model specs.")
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

    Hq = int(_get_first(cfg, ["num_attention_heads", "n_head", "num_heads"]))
    Hkv = int(_get_first(cfg, ["num_key_value_heads", "num_kv_heads"], default=Hq))
    hidden = int(_get_first(cfg, ["hidden_size", "n_embd", "d_model"]))

    if hidden % Hq != 0:
        raise ValueError(
            f"hidden_size {hidden} not divisible by num_attention_heads {Hq} for {model_id}"
        )
    D = hidden // Hq

    if Hq % Hkv != 0:
        raise ValueError(
            f"Hq {Hq} must be divisible by Hkv {Hkv} for GQA; model={model_id}"
        )
    g = Hq // Hkv

    n_layers = int(
        _get_first(cfg, ["num_hidden_layers", "n_layer", "num_layers"], default=1)
    )
    max_seq = int(
        _get_first(
            cfg,
            [
                "max_position_embeddings",
                "n_positions",
                "max_seq_len",
                "seq_length",
                "model_max_length",
            ],
            default=2048,
        )
    )

    # quick param estimate (good enough for ordering in plots)
    vocab = int(_get_first(cfg, ["vocab_size"], default=32000))
    inter = int(_get_first(cfg, ["intermediate_size", "n_inner"], default=4 * hidden))
    hidden_kv = Hkv * D
    attn_params = (
        hidden * hidden + hidden * hidden_kv + hidden * hidden_kv + hidden * hidden
    )
    mlp_params = 3 * hidden * inter
    emb_params = vocab * hidden
    total = emb_params + n_layers * (attn_params + mlp_params)
    approx_params_b = total / 1e9

    return ModelSpec(
        model_id=model_id,
        label=short_model_label(model_id),
        approx_params_b=approx_params_b,
        n_layers=n_layers,
        max_seq=max_seq,
        Hq=Hq,
        Hkv=Hkv,
        D=D,
        g=g,
    )


def choose_B_for_budget(
    *,
    mem_gb: float,
    T: int,
    Hkv: int,
    D: int,
    dtype: torch.dtype,
    block: int,
    variant: str,
) -> int:
    """Very rough B chooser to fit a target memory budget.

    variant:
      - "ours": paged KV only
      - "pack": paged KV + dense pack buffers
    """
    bytes_per = 2 if dtype in (torch.float16, torch.bfloat16) else 4
    Tpad = ((T + block - 1) // block) * block

    kv_bytes = 2 * Tpad * Hkv * D * bytes_per  # K+V
    pack_bytes = 2 * T * Hkv * D * bytes_per  # K+V dense

    if variant == "ours":
        per_req = kv_bytes
        safety = 1.25
    elif variant == "pack":
        per_req = kv_bytes + pack_bytes
        safety = 1.45
    else:
        raise ValueError(f"Unknown variant for budget: {variant}")

    budget = mem_gb * (1024**3)
    B = int(budget // (safety * per_req))
    return max(1, B)


# -----------------------------
# Baseline pack
# -----------------------------


@no_grad()
def pack_paged_to_dense(
    *,
    cache,
    k_dense: torch.Tensor,
    v_dense: torch.Tensor,
    chunk_B: int = 8,
) -> None:
    """Pack paged KV blocks into dense [B,T,Hkv,D] in logical token order."""
    assert k_dense.ndim == 4 and v_dense.ndim == 4
    assert k_dense.shape == v_dense.shape
    B, T, Hkv, D = k_dense.shape

    block = int(cache.block_size)
    max_blocks = int(cache.block_table.shape[1])
    Tpad = max_blocks * block
    if T > Tpad:
        raise ValueError(f"Dense T={T} exceeds cache capacity Tpad={Tpad}")

    # fast path if all ctx are full T
    ctx_min = int(cache.ctx_lens.min().item())
    ctx_max = int(cache.ctx_lens.max().item())
    const_full_ctx = ctx_min == ctx_max == T

    if not const_full_ctx:
        k_dense.zero_()
        v_dense.zero_()

    for b0 in range(0, B, chunk_B):
        b1 = min(B, b0 + chunk_B)
        bc = b1 - b0
        if bc <= 0:
            continue

        if const_full_ctx:
            ctx_chunk = None
            t_max = T
        else:
            ctx_chunk = cache.ctx_lens[b0:b1].to(torch.int64)
            t_max = int(ctx_chunk.max().item())
            t_max = min(t_max, T)
            if t_max <= 0:
                continue

        phys = cache.block_table[b0:b1, :max_blocks].reshape(-1).to(torch.int64)

        if cache.layout == "bhsd":
            k_blocks = cache.k.index_select(0, phys)
            v_blocks = cache.v.index_select(0, phys)
            k_lin = (
                k_blocks.view(bc, max_blocks, Hkv, block, D)
                .permute(0, 1, 3, 2, 4)
                .reshape(bc, Tpad, Hkv, D)
            )
            v_lin = (
                v_blocks.view(bc, max_blocks, Hkv, block, D)
                .permute(0, 1, 3, 2, 4)
                .reshape(bc, Tpad, Hkv, D)
            )
        else:
            k_blocks = cache.k.index_select(0, phys)
            v_blocks = cache.v.index_select(0, phys)
            k_lin = k_blocks.view(bc, max_blocks, block, Hkv, D).reshape(
                bc, Tpad, Hkv, D
            )
            v_lin = v_blocks.view(bc, max_blocks, block, Hkv, D).reshape(
                bc, Tpad, Hkv, D
            )

        k_dense[b0:b1, :t_max].copy_(k_lin[:, :t_max])
        v_dense[b0:b1, :t_max].copy_(v_lin[:, :t_max])

        if not const_full_ctx:
            for i in range(bc):
                ci = int(ctx_chunk[i].item())
                if ci < t_max:
                    k_dense[b0 + i, ci:t_max].zero_()
                    v_dense[b0 + i, ci:t_max].zero_()


def dense_attn_fn_factory(
    *,
    flash_fn,
    Hq: int,
    g: int,
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """Return dense attention function (FlashAttn if available else SDPA).

    Returned signature:
      f(q_bhd, k_bthd, v_bthd, seqlens_i32) -> out[B,Hq,D]
    """

    if flash_fn is not None:

        def _flash(
            q_bhd: torch.Tensor,
            k_bthd: torch.Tensor,
            v_bthd: torch.Tensor,
            seqlens: torch.Tensor,
        ) -> torch.Tensor:
            out = call_flash_attn_kvcache(
                flash_fn, q_bhd, k_bthd, v_bthd, seqlens, causal=True
            )
            if out.ndim == 4:
                return out[:, 0, :, :]
            if out.ndim == 3:
                return out
            raise RuntimeError(f"Unexpected flash output shape: {tuple(out.shape)}")

        return _flash

    # SDPA fallback.
    # Prefer enable_gqa=True if the torch version supports it.
    has_enable_gqa = False
    try:
        sig = inspect.signature(torch.nn.functional.scaled_dot_product_attention)
        has_enable_gqa = "enable_gqa" in sig.parameters
    except Exception:
        has_enable_gqa = False

    def _sdpa(
        q_bhd: torch.Tensor,
        k_bthd: torch.Tensor,
        v_bthd: torch.Tensor,
        seqlens: torch.Tensor,
    ) -> torch.Tensor:
        # SDPA wants [B,H,S,D]
        q_ = q_bhd[:, :, None, :]  # [B,Hq,1,D]
        k_h = k_bthd.permute(0, 2, 1, 3)  # [B,Hkv,T,D]
        v_h = v_bthd.permute(0, 2, 1, 3)

        if has_enable_gqa:
            out = torch.nn.functional.scaled_dot_product_attention(
                q_, k_h, v_h, is_causal=False, enable_gqa=True
            )
            return out[:, :, 0, :]

        # Worst-case fallback: head-repeat as a view.
        # Note: expand+reshape is a view; SDPA kernels may still materialize internally.
        B, Hkv, T, D = k_h.shape
        k_rep = k_h[:, :, None, :, :].expand(B, Hkv, g, T, D).reshape(B, Hq, T, D)
        v_rep = v_h[:, :, None, :, :].expand(B, Hkv, g, T, D).reshape(B, Hq, T, D)
        out = torch.nn.functional.scaled_dot_product_attention(
            q_, k_rep, v_rep, is_causal=False
        )
        return out[:, :, 0, :]

    return _sdpa


# -----------------------------
# Result row
# -----------------------------


@dataclass
class Row:
    model_id: str
    model_label: str
    approx_params_b: float
    n_layers: int
    max_seq: int
    Hq: int
    Hkv: int
    D: int
    g: int
    T: int
    B: int
    dtype: str
    layout: str
    block: int
    bpi: int
    variant: str
    median_ms: float
    mean_ms: float
    cv: float
    kv_tokens_s: float
    peak_alloc_gb: float
    peak_reserved_gb: float
    oom: bool


DEFAULT_MODELS = ",".join(
    [
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "Qwen/Qwen2-1.5B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.2",
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
    ]
)


def _coerce_bool_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().isin(["true", "1", "yes"])


def coerce_results_df(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = [
        "approx_params_b",
        "n_layers",
        "max_seq",
        "Hq",
        "Hkv",
        "D",
        "g",
        "T",
        "B",
        "block",
        "bpi",
        "median_ms",
        "mean_ms",
        "cv",
        "kv_tokens_s",
        "peak_alloc_gb",
        "peak_reserved_gb",
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "oom" in df.columns:
        df["oom"] = _coerce_bool_series(df["oom"])
    return df


def specs_from_results(df: pd.DataFrame) -> List[ModelSpec]:
    specs = []
    for model_id, sub in df.groupby("model_id"):
        row = sub.iloc[0]
        specs.append(
            ModelSpec(
                model_id=model_id,
                label=str(row["model_label"]),
                approx_params_b=float(row["approx_params_b"]),
                n_layers=int(row["n_layers"]),
                max_seq=int(row["max_seq"]),
                Hq=int(row["Hq"]),
                Hkv=int(row["Hkv"]),
                D=int(row["D"]),
                g=int(row["g"]),
            )
        )
    return specs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=str, default=DEFAULT_MODELS)
    ap.add_argument(
        "--seqlens", type=str, default="512,1024,2048,4096,8192,16384,32768"
    )
    ap.add_argument("--hero_T", type=int, default=8192)
    ap.add_argument("--allow_exceed_maxpos", action="store_true")

    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--layout", type=str, default="bhsd", choices=["bhsd", "bshd"])
    ap.add_argument("--block", type=int, default=16)
    ap.add_argument("--bpi", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument(
        "--mem_gb",
        type=float,
        default=28.0,
        help="Approx VRAM budget for choosing a per-model fixed batch size B (0 disables auto and uses --B).",
    )
    ap.add_argument("--B", type=int, default=0, help="Force batch size (0=auto)")
    ap.add_argument("--max_B", type=int, default=256)

    ap.add_argument(
        "--ours_module",
        type=str,
        default="paged_gqa_decode_triton",
        help="Module name providing PagedKVCache/build_perm_lbi/allocate_paged_kv_cache/paged_gqa_decode.",
    )
    ap.add_argument("--outdir", type=str, default="out_wow")
    ap.add_argument("--no_flash", action="store_true")
    ap.add_argument(
        "--results_csv", type=str, default="", help="Plot-only: path to results.csv"
    )
    ap.add_argument(
        "--hero_table_csv",
        type=str,
        default="",
        help="Plot-only: path to hero_table.csv",
    )
    ap.add_argument(
        "--plot_only",
        action="store_true",
        help="Skip benchmarks; render plots from CSVs",
    )
    ap.add_argument(
        "--baseline_label",
        type=str,
        default="",
        help="Plot-only: baseline label (flash/sdpa)",
    )
    ap.add_argument(
        "--device_label", type=str, default="", help="Plot-only: footer device label"
    )

    ap.add_argument(
        "--flagship_model",
        type=str,
        default="",
        help="Optional model_id to use for the bottom 2-panels. Default: largest by approx params.",
    )

    args = ap.parse_args()

    plot_only = bool(args.plot_only or args.results_csv)
    if not plot_only:
        if torch is None:
            raise RuntimeError("PyTorch is required for benchmarks.")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")

    # Reduce some Triton debug noise in some environments
    os.environ.setdefault("TRITON_DISABLE_LINE_INFO", "1")

    if plot_only:
        if not args.results_csv:
            raise RuntimeError("--plot_only requires --results_csv")
        df = pd.read_csv(args.results_csv)
        df = coerce_results_df(df)
        summary_df = pd.read_csv(args.hero_table_csv) if args.hero_table_csv else None
        if summary_df is not None:
            for col in [
                "~Params (B)",
                "B",
                "T_cmp",
                "Speedup (x)",
                "VRAM saving (%)",
                "Max T ours",
                "Max T baseline",
                "Longer context (x)",
            ]:
                if col in summary_df.columns:
                    summary_df[col] = pd.to_numeric(summary_df[col], errors="coerce")
        specs = specs_from_results(df)
        if args.flagship_model:
            flagship = next(
                (s for s in specs if s.model_id == args.flagship_model), None
            )
            if flagship is None:
                flagship = max(specs, key=lambda s: s.approx_params_b)
        else:
            flagship = max(specs, key=lambda s: s.approx_params_b)
        baseline_kind = args.baseline_label or "sdpa"
        device_label = args.device_label or "offline"
    else:
        dtype = dtype_from_str(args.dtype)
        import importlib

        ours = importlib.import_module(args.ours_module)

        flash_fn = None if args.no_flash else try_import_flash_attn_with_kvcache()
        baseline_kind = "flash" if flash_fn is not None else "sdpa"
        device_label = torch.cuda.get_device_name(0)

    seqlens_req = [int(x) for x in args.seqlens.split(",") if x.strip()]
    model_ids = [m.strip() for m in args.models.split(",") if m.strip()]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if plot_only:
        # Jump to summary/plot using loaded CSVs.
        rows = []
        meta = {}
        flash_fn = None
        # If summary_df wasn't provided, rebuild it from df below.
        # Skip the benchmarking loop entirely.

    summary_df = summary_df if plot_only else None

    if not plot_only:
        # Derive specs from configs
        specs = [derive_model_spec(mid) for mid in model_ids]

        # Select flagship
        if args.flagship_model:
            flagship = next(
                (s for s in specs if s.model_id == args.flagship_model), None
            )
            if flagship is None:
                raise ValueError(
                    f"--flagship_model={args.flagship_model} not in --models"
                )
        else:
            flagship = max(specs, key=lambda s: s.approx_params_b)

        print("=== WOW HF Hero Benchmark (Unsloth-style) ===")
        print(
            f"device={torch.cuda.get_device_name(0)} torch={torch.__version__} dtype={dtype} baseline={baseline_kind}"
        )
        print("Models:")
        for s in specs:
            print(
                f"  - {s.label:18s} ~{s.approx_params_b:.2f}B layers={s.n_layers:<3d} "
                f"Hq={s.Hq:<3d} Hkv={s.Hkv:<3d} D={s.D:<3d} g={s.g:<2d} max_seq={s.max_seq}"
            )
        print(f"Flagship panels: {flagship.label} ({flagship.model_id})")

        device = torch.device("cuda")

        rows: List[Row] = []

        # Bench loop
        for spec in specs:
            # Clamp seqlens to model max_seq unless synthetic.
            Ts = [
                t
                for t in seqlens_req
                if (args.allow_exceed_maxpos or t <= spec.max_seq)
            ]
            if len(Ts) == 0:
                print(f"[skip] {spec.label}: no seqlens within max_seq={spec.max_seq}")
                continue

            Ts = sorted(set(Ts))
            T_max = max(Ts)

            # Choose a fixed B for this model
            if args.B > 0:
                B = int(args.B)
            else:
                if args.mem_gb <= 0:
                    B = 64
                else:
                    # Choose B such that OURS fits at the max T in the sweep.
                    B = choose_B_for_budget(
                        mem_gb=args.mem_gb,
                        T=T_max,
                        Hkv=spec.Hkv,
                        D=spec.D,
                        dtype=dtype,
                        block=args.block,
                        variant="ours",
                    )
            B = max(1, min(B, int(args.max_B)))

            # Use a consistent generator
            gen = torch.Generator(device=device)
            gen.manual_seed(args.seed + 123)

            # Dense attention function (baseline)
            dense_attn = dense_attn_fn_factory(flash_fn=flash_fn, Hq=spec.Hq, g=spec.g)

            # Pre-allocate a batch id vector
            batch_ids = torch.arange(B, device=device, dtype=torch.int32)

            for T in Ts:
                # Build paged KV cache
                try:
                    cache = ours.allocate_paged_kv_cache(
                        B=B,
                        Hkv=spec.Hkv,
                        D=spec.D,
                        max_seq=T,
                        block_size=args.block,
                        dtype=dtype,
                        device=device,
                        layout=args.layout,
                    )

                    # Mark full context
                    cache.ctx_lens.fill_(T)

                    # Fill KV blocks (synthetic)
                    cache.k.normal_(mean=0.0, std=0.5)
                    cache.v.normal_(mean=0.0, std=0.5)

                    # Perm schedule (logical)
                    perm = ours.build_perm_lbi(
                        cache.block_table,
                        cache.ctx_lens,
                        cache.block_size,
                        order="logical",
                    )

                    # Query
                    q = torch.randn(
                        (B, spec.Hq, spec.D), device=device, dtype=dtype, generator=gen
                    )

                except RuntimeError as e:
                    if is_oom(e):
                        # If we can't even allocate the cache, record OOM for both variants.
                        for variant in ("pack_plus_dense_attn", "ours_paged"):
                            rows.append(
                                Row(
                                    model_id=spec.model_id,
                                    model_label=spec.label,
                                    approx_params_b=spec.approx_params_b,
                                    n_layers=spec.n_layers,
                                    max_seq=spec.max_seq,
                                    Hq=spec.Hq,
                                    Hkv=spec.Hkv,
                                    D=spec.D,
                                    g=spec.g,
                                    T=T,
                                    B=B,
                                    dtype=args.dtype,
                                    layout=args.layout,
                                    block=args.block,
                                    bpi=args.bpi,
                                    variant=variant,
                                    median_ms=float("nan"),
                                    mean_ms=float("nan"),
                                    cv=float("nan"),
                                    kv_tokens_s=float("nan"),
                                    peak_alloc_gb=float("nan"),
                                    peak_reserved_gb=float("nan"),
                                    oom=True,
                                )
                            )
                        print(
                            f"[{spec.label:18s} T={T:<6d} B={B:<4d}] OOM allocating paged cache"
                        )
                        torch.cuda.empty_cache()
                        continue
                    raise

                seqlens_i32 = cache.ctx_lens.to(torch.int32)
                max_blocks = int(cache.block_table.shape[1])

                # ---- OURS functions
                @no_grad()
                def fn_ours() -> torch.Tensor:
                    return ours.paged_gqa_decode(
                        q,
                        cache,
                        perm_lbi=perm,
                        batch_ids=batch_ids,
                        blocks_per_iter=args.bpi,
                        max_blocks=max_blocks,
                        num_warps=4,
                        num_stages=2,
                    )

                # ---- BASELINE functions (allocate dense buffers lazily)
                k_dense = None
                v_dense = None

                def ensure_dense():
                    nonlocal k_dense, v_dense
                    if k_dense is None:
                        k_dense = torch.empty(
                            (B, T, spec.Hkv, spec.D), device=device, dtype=dtype
                        )
                        v_dense = torch.empty(
                            (B, T, spec.Hkv, spec.D), device=device, dtype=dtype
                        )

                @no_grad()
                def fn_pack_plus_attn() -> torch.Tensor:
                    ensure_dense()
                    pack_paged_to_dense(
                        cache=cache, k_dense=k_dense, v_dense=v_dense, chunk_B=8
                    )
                    return dense_attn(q, k_dense, v_dense, seqlens_i32)

                # -----------------
                # Correctness spot-check (only once per model)
                # -----------------
                if T == Ts[0]:
                    try:
                        o_ref = fn_pack_plus_attn().float()
                        o_new = fn_ours().float()
                        diff = (o_ref - o_new).abs()
                        l2_rel = float(diff.norm() / (o_ref.norm() + 1e-8))
                        print(
                            f"[agree] {spec.label:18s} T={T:<6d} B={B:<4d} max_abs={diff.max().item():.3e} l2_rel={l2_rel:.3e}"
                        )
                    except RuntimeError as e:
                        if is_oom(e):
                            print(
                                f"[agree] {spec.label:18s} T={T:<6d} B={B:<4d} skipped (OOM)"
                            )
                        else:
                            raise

                # -----------------
                # Benchmark OURS
                # -----------------
                try:
                    med, cv, mean = bench_cuda_ms(
                        fn_ours, warmup=args.warmup, iters=args.iters
                    )
                    alloc_b, reserv_b = measure_peak_mem_bytes(fn_ours)
                    kv_tokens = float(B * T)
                    kv_tps = kv_tokens / (med / 1000.0)
                    rows.append(
                        Row(
                            model_id=spec.model_id,
                            model_label=spec.label,
                            approx_params_b=spec.approx_params_b,
                            n_layers=spec.n_layers,
                            max_seq=spec.max_seq,
                            Hq=spec.Hq,
                            Hkv=spec.Hkv,
                            D=spec.D,
                            g=spec.g,
                            T=T,
                            B=B,
                            dtype=args.dtype,
                            layout=args.layout,
                            block=args.block,
                            bpi=args.bpi,
                            variant="ours_paged",
                            median_ms=med,
                            mean_ms=mean,
                            cv=cv,
                            kv_tokens_s=kv_tps,
                            peak_alloc_gb=bytes_to_gb(alloc_b),
                            peak_reserved_gb=bytes_to_gb(reserv_b),
                            oom=False,
                        )
                    )
                except RuntimeError as e:
                    if is_oom(e):
                        rows.append(
                            Row(
                                model_id=spec.model_id,
                                model_label=spec.label,
                                approx_params_b=spec.approx_params_b,
                                n_layers=spec.n_layers,
                                max_seq=spec.max_seq,
                                Hq=spec.Hq,
                                Hkv=spec.Hkv,
                                D=spec.D,
                                g=spec.g,
                                T=T,
                                B=B,
                                dtype=args.dtype,
                                layout=args.layout,
                                block=args.block,
                                bpi=args.bpi,
                                variant="ours_paged",
                                median_ms=float("nan"),
                                mean_ms=float("nan"),
                                cv=float("nan"),
                                kv_tokens_s=float("nan"),
                                peak_alloc_gb=float("nan"),
                                peak_reserved_gb=float("nan"),
                                oom=True,
                            )
                        )
                        print(f"[{spec.label:18s} T={T:<6d} B={B:<4d}] ours: OOM")
                    else:
                        raise

                # -----------------
                # Benchmark BASELINE (PACK + dense attn)
                # -----------------
                try:
                    # Ensure dense alloc happens outside timing (like a steady-state server)
                    ensure_dense()

                    med, cv, mean = bench_cuda_ms(
                        fn_pack_plus_attn, warmup=args.warmup, iters=args.iters
                    )
                    alloc_b, reserv_b = measure_peak_mem_bytes(fn_pack_plus_attn)

                    kv_tokens = float(B * T)
                    kv_tps = kv_tokens / (med / 1000.0)
                    rows.append(
                        Row(
                            model_id=spec.model_id,
                            model_label=spec.label,
                            approx_params_b=spec.approx_params_b,
                            n_layers=spec.n_layers,
                            max_seq=spec.max_seq,
                            Hq=spec.Hq,
                            Hkv=spec.Hkv,
                            D=spec.D,
                            g=spec.g,
                            T=T,
                            B=B,
                            dtype=args.dtype,
                            layout=args.layout,
                            block=args.block,
                            bpi=args.bpi,
                            variant="pack_plus_dense_attn",
                            median_ms=med,
                            mean_ms=mean,
                            cv=cv,
                            kv_tokens_s=kv_tps,
                            peak_alloc_gb=bytes_to_gb(alloc_b),
                            peak_reserved_gb=bytes_to_gb(reserv_b),
                            oom=False,
                        )
                    )
                except RuntimeError as e:
                    if is_oom(e):
                        rows.append(
                            Row(
                                model_id=spec.model_id,
                                model_label=spec.label,
                                approx_params_b=spec.approx_params_b,
                                n_layers=spec.n_layers,
                                max_seq=spec.max_seq,
                                Hq=spec.Hq,
                                Hkv=spec.Hkv,
                                D=spec.D,
                                g=spec.g,
                                T=T,
                                B=B,
                                dtype=args.dtype,
                                layout=args.layout,
                                block=args.block,
                                bpi=args.bpi,
                                variant="pack_plus_dense_attn",
                                median_ms=float("nan"),
                                mean_ms=float("nan"),
                                cv=float("nan"),
                                kv_tokens_s=float("nan"),
                                peak_alloc_gb=float("nan"),
                                peak_reserved_gb=float("nan"),
                                oom=True,
                            )
                        )
                        print(f"[{spec.label:18s} T={T:<6d} B={B:<4d}] baseline: OOM")
                    else:
                        raise

                # Print a one-line status if both exist
                df_tmp = pd.DataFrame(
                    [
                        asdict(r)
                        for r in rows
                        if r.model_id == spec.model_id and r.T == T
                    ]
                )
                ours_ok = df_tmp[df_tmp.variant == "ours_paged"]
                base_ok = df_tmp[df_tmp.variant == "pack_plus_dense_attn"]
                if (
                    len(ours_ok)
                    and len(base_ok)
                    and (not bool(ours_ok.iloc[0].oom))
                    and (not bool(base_ok.iloc[0].oom))
                ):
                    sp = float(base_ok.iloc[0].median_ms) / float(
                        ours_ok.iloc[0].median_ms
                    )
                    print(
                        f"[{spec.label:18s} T={T:<6d} B={B:<4d}] pack+attn={float(base_ok.iloc[0].median_ms):7.3f}ms  "
                        f"ours={float(ours_ok.iloc[0].median_ms):7.3f}ms  speedup={sp:5.2f}x"
                    )

                # Free big dense buffers between T steps to allow larger T (baseline buffers are huge)
                # This matches the 'can we run?' story for longer context.
                del k_dense, v_dense
                torch.cuda.empty_cache()

            # free per-model
            torch.cuda.empty_cache()

        # Save results
        df = pd.DataFrame([asdict(r) for r in rows])
        df.to_csv(outdir / "results.csv", index=False)

        meta = {
            "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "dtype": args.dtype,
            "layout": args.layout,
            "block": args.block,
            "bpi": args.bpi,
            "warmup": args.warmup,
            "iters": args.iters,
            "baseline": baseline_kind,
            "flash_available": flash_fn is not None,
            "models": [asdict(s) for s in specs],
            "flagship_model": flagship.model_id,
        }
        (outdir / "results.json").write_text(json.dumps(meta, indent=2))

    # -----------------------------
    # Build summary table (Unsloth benchmark-table style)
    # -----------------------------
    if summary_df is None:
        summary_rows: List[Dict[str, Any]] = []

        for spec in specs:
            sub = df[df.model_id == spec.model_id]
            if len(sub) == 0:
                continue

            # Max context before OOM (within the attempted sweep)
            base_ok = sub[(sub.variant == "pack_plus_dense_attn") & (~sub["oom"])]
            ours_ok = sub[(sub.variant == "ours_paged") & (~sub["oom"])]

            base_Tmax = int(base_ok["T"].max()) if len(base_ok) else 0
            ours_Tmax = int(ours_ok["T"].max()) if len(ours_ok) else 0
            longer = (
                (ours_Tmax / base_Tmax)
                if base_Tmax > 0
                else float("inf")
                if ours_Tmax > 0
                else float("nan")
            )

            # Choose a comparison T: prefer hero_T, else largest common successful T.
            hero_T = (
                args.hero_T
                if args.allow_exceed_maxpos
                else min(args.hero_T, spec.max_seq)
            )

            base_hero = base_ok[base_ok["T"] == hero_T]
            ours_hero = ours_ok[ours_ok["T"] == hero_T]

            if len(base_hero) and len(ours_hero):
                T_cmp = hero_T
                b_ms = float(base_hero.iloc[0].median_ms)
                o_ms = float(ours_hero.iloc[0].median_ms)
                b_mem = float(base_hero.iloc[0].peak_reserved_gb)
                o_mem = float(ours_hero.iloc[0].peak_reserved_gb)
            else:
                common_Ts = sorted(
                    set(base_ok["T"].tolist()) & set(ours_ok["T"].tolist())
                )
                if len(common_Ts) == 0:
                    continue
                T_cmp = int(max(common_Ts))
                b = base_ok[base_ok["T"] == T_cmp].iloc[0]
                o = ours_ok[ours_ok["T"] == T_cmp].iloc[0]
                b_ms = float(b.median_ms)
                o_ms = float(o.median_ms)
                b_mem = float(b.peak_reserved_gb)
                o_mem = float(o.peak_reserved_gb)

            speedup = (b_ms / o_ms) if o_ms > 0 else float("nan")
            vram_saving = (
                ((b_mem - o_mem) / b_mem * 100.0) if b_mem > 0 else float("nan")
            )

            summary_rows.append(
                {
                    "Model": spec.label,
                    "~Params (B)": round(spec.approx_params_b, 2),
                    "B": int(sub.B.max()) if "B" in sub else None,
                    "T_cmp": T_cmp,
                    "Speedup (x)": speedup,
                    "VRAM saving (%)": vram_saving,
                    "Max T ours": ours_Tmax,
                    "Max T baseline": base_Tmax,
                    "Longer context (x)": longer,
                }
            )

        summary_df = pd.DataFrame(summary_rows)

    summary_df.to_csv(outdir / "hero_table.csv", index=False)

    # -----------------------------
    # Hero figure
    # -----------------------------

    palette = apply_unsloth_style()

    # Global callouts
    max_speedup = float("nan")
    max_vram_save = float("nan")
    if len(summary_df):
        max_speedup = float(summary_df["Speedup (x)"].max())
        max_vram_save = float(summary_df["VRAM saving (%)"].max())

    max_longer = float("nan")
    if len(summary_df):
        # filter inf
        tmp = summary_df[
            summary_df["Longer context (x)"]
            .replace([math.inf, -math.inf], math.nan)
            .notna()
        ]
        if len(tmp):
            max_longer = float(tmp["Longer context (x)"].max())

    # Flagship curves
    subF = df[df.model_id == flagship.model_id]
    subF_base = subF[subF.variant == "pack_plus_dense_attn"].sort_values("T")
    subF_ours = subF[subF.variant == "ours_paged"].sort_values("T")

    fig = plt.figure(figsize=(15.5, 8.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0])

    ax_table = fig.add_subplot(gs[0, :])
    ax_time = fig.add_subplot(gs[1, 0])
    ax_mem = fig.add_subplot(gs[1, 1])

    # ---- Title / headline
    headline_parts = []
    if not math.isnan(max_speedup):
        headline_parts.append(f"up to {max_speedup:.1f}× faster")
    if not math.isnan(max_vram_save):
        headline_parts.append(f"up to {max_vram_save:.0f}% less VRAM")
    if not math.isnan(max_longer) and max_longer > 1.01:
        headline_parts.append(f"up to {max_longer:.1f}× longer context")

    title_text = "SyDecode: PACK-free paged KV decode"
    subtitle_text = ""
    if headline_parts:
        subtitle_text = "(" + ", ".join(headline_parts) + ")"

    title_fs = 20
    subtitle_fs = title_fs
    fig.text(
        0.5,
        0.975,
        title_text,
        ha="center",
        va="top",
        fontsize=title_fs,
        fontweight="semibold",
    )
    if subtitle_text:
        fig.text(
            0.5,
            0.93,
            subtitle_text,
            ha="center",
            va="top",
            fontsize=subtitle_fs,
            fontweight="semibold",
        )

    fig.subplots_adjust(
        top=0.90, bottom=0.07, left=0.04, right=0.985, hspace=0.28, wspace=0.22
    )

    # ---- Table panel
    ax_table.axis("off")
    if len(summary_df) == 0:
        ax_table.text(0.5, 0.5, "(no summary data)", ha="center", va="center")
    else:
        disp = summary_df.copy()
        # prettify
        disp["Speedup (x)"] = disp["Speedup (x)"].map(
            lambda x: f"{x:.1f}×" if pd.notna(x) else "—"
        )
        disp["VRAM saving (%)"] = disp["VRAM saving (%)"].map(
            lambda x: f"{x:.0f}%" if pd.notna(x) else "—"
        )
        disp["Longer context (x)"] = disp["Longer context (x)"].map(
            lambda x: ("∞" if x == math.inf else f"{x:.1f}×") if pd.notna(x) else "—"
        )

        cols = [
            "Model",
            "~Params (B)",
            "Speedup (x)",
            "VRAM saving (%)",
            "Longer context (x)",
        ]
        disp = disp[cols]

        table = ax_table.table(
            cellText=disp.values.tolist(),
            colLabels=disp.columns.tolist(),
            cellLoc="center",
            colLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(20)
        table.scale(1.0, 1.7)

        speedup_idx = int(disp.columns.get_loc("Speedup (x)"))
        vram_idx = int(disp.columns.get_loc("VRAM saving (%)"))

        # Header styling
        for (r, c), cell in table.get_celld().items():
            cell.set_linewidth(0.6)
            cell.set_edgecolor(palette["grid"])
            if r == 0:
                cell.set_text_props(weight="semibold", color=palette["ink"])
                cell.set_facecolor("#F7F7F7")
            else:
                cell.set_facecolor("#FFFFFF")
                if c in (speedup_idx, vram_idx):
                    cell.set_text_props(weight="bold", color=palette["ink"])
                else:
                    cell.set_text_props(color=palette["slate"])

    # ---- Flagship time curve
    ax_time.set_title(f"{flagship.label}: Step Time (ms) [Lower is Better]")
    ax_time.set_xlabel("Sequence length")
    ax_time.set_ylabel("Time per decode step (ms)")

    def plot_series(
        ax,
        sdf: pd.DataFrame,
        ycol: str,
        *,
        label: str,
        color: str,
        ls: str,
        marker: str,
    ):
        ok = sdf[~sdf["oom"]]
        if len(ok) == 0:
            return
        ax.plot(
            ok["T"],
            ok[ycol],
            label=label,
            color=color,
            linestyle=ls,
            marker=marker,
            linewidth=2.2,
            markersize=6,
        )

    plot_series(
        ax_time,
        subF_base,
        "median_ms",
        label=f"Baseline: PACK + {baseline_kind}",
        color=palette["baseline"],
        ls="--",
        marker="x",
    )
    plot_series(
        ax_time,
        subF_ours,
        "median_ms",
        label="SyDecode (paged-native)",
        color=palette["accent"],
        ls="-",
        marker="o",
    )

    ax_time.set_xscale("log", base=2)
    ax_time.grid(True)
    ax_time.legend(loc="best")

    # Annotate baseline OOM
    base_oom_Ts = subF_base[subF_base["oom"]]["T"].tolist()
    if base_oom_Ts:
        oom_T = int(min(base_oom_Ts))
        # Put text near the top.
        y_top = ax_time.get_ylim()[1]
        ax_time.axvline(
            oom_T, color=palette["baseline_light"], linestyle=":", linewidth=1.2
        )
        ax_time.text(
            oom_T,
            y_top * 0.95,
            f"baseline OOM @ {oom_T}",
            rotation=90,
            ha="right",
            va="top",
            fontsize=9,
            color=palette["baseline"],
        )

    # ---- Flagship memory curve
    ax_mem.set_title(f"{flagship.label}: Memory Usage (GB) [Lower is Better]")
    ax_mem.set_xlabel("Sequence length")
    ax_mem.set_ylabel("Peak reserved VRAM (GB)")

    plot_series(
        ax_mem,
        subF_base,
        "peak_reserved_gb",
        label=f"Baseline: PACK + {baseline_kind}",
        color=palette["baseline"],
        ls="--",
        marker="x",
    )
    plot_series(
        ax_mem,
        subF_ours,
        "peak_reserved_gb",
        label="SyDecode (paged-native)",
        color=palette["accent"],
        ls="-",
        marker="o",
    )

    ax_mem.set_xscale("log", base=2)
    ax_mem.grid(True)
    ax_mem.legend(loc="best")

    if base_oom_Ts:
        oom_T = int(min(base_oom_Ts))
        y_top = ax_mem.get_ylim()[1]
        ax_mem.axvline(
            oom_T, color=palette["baseline_light"], linestyle=":", linewidth=1.2
        )
        ax_mem.text(
            oom_T,
            y_top * 0.95,
            f"baseline OOM @ {oom_T}",
            rotation=90,
            ha="right",
            va="top",
            fontsize=9,
            color=palette["baseline"],
        )

    # Footer
    fig.text(
        0.99,
        0.01,
        f"{device_label} • {args.dtype.upper()} • baseline={baseline_kind} • block={args.block} • bpi={args.bpi} • B<= {args.max_B}",
        ha="right",
        va="bottom",
        fontsize=9,
        color=palette["slate"],
    )

    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(outdir / "hero.png", dpi=220)
    fig.savefig(outdir / "hero.pdf")

    if not plot_only:
        print(f"\nWrote: {outdir / 'results.csv'}")
        print(f"Wrote: {outdir / 'results.json'}")
    else:
        print(f"\nLoaded: {args.results_csv}")
        if args.hero_table_csv:
            print(f"Loaded: {args.hero_table_csv}")
    print(f"Wrote: {outdir / 'hero_table.csv'}")
    print(f"Wrote: {outdir / 'hero.png'}")
    print(f"Wrote: {outdir / 'hero.pdf'}")


if __name__ == "__main__":
    main()
