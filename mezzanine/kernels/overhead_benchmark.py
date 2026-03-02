#!/usr/bin/env python3
"""overhead_benchmark.py

Benchmark incremental overhead of computing a sampled transcript hash inside a GEMM
kernel epilogue (Triton).

Overhead is computed as:
  overhead% = (T_fused / T_base - 1) * 100

Timing excludes any host digest readback unless `--emit-digest` is requested.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Dict, List

import torch

from .pearl_pow_kernels.gemm_triton_fused import (
    prepare_fused_workspace,
    launch_gemm_fused_inplace,
    triton_available,
)
from .pearl_pow_kernels.trace import TraceConfig


def _cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def bench_ms(fn, *, warmup: int, iters: int) -> float:
    if torch.cuda.is_available():
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        for _ in range(warmup):
            fn()
        _cuda_sync()
        times = []
        for _ in range(iters):
            starter.record()
            fn()
            ender.record()
            _cuda_sync()
            times.append(starter.elapsed_time(ender))
        times.sort()
        return float(times[len(times) // 2])
    else:
        for _ in range(warmup):
            fn()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        t1 = time.perf_counter()
        return (t1 - t0) * 1000.0 / iters


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
    ap.add_argument("--sizes", default="1024,2048,4096")
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--max-overhead-pct", type=float, default=5.0)
    ap.add_argument("--noise-scale", type=float, default=0.0)
    ap.add_argument("--permute-ktiles", action="store_true")
    ap.add_argument("--emit-digest", action="store_true")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available")
    if args.device == "cuda" and not triton_available():
        raise SystemExit("Triton not available in this env")

    device = torch.device(args.device)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    sizes = [int(x) for x in args.sizes.split(",") if x.strip()]

    print("\n=== Fused GEMM+hash overhead benchmark (Triton, inplace) ===")
    info = {
        "torch_version": torch.__version__,
        "device": str(device),
        "dtype": str(dtype),
        "cuda_device_name": torch.cuda.get_device_name(0)
        if device.type == "cuda"
        else None,
        "cuda_capability": torch.cuda.get_device_capability(0)
        if device.type == "cuda"
        else None,
        "triton_available": triton_available(),
    }
    print(json.dumps(info, indent=2))
    print(
        f"sizes={sizes} warmup={args.warmup} iters={args.iters} samples={args.samples} "
        f"max_overhead_pct={args.max_overhead_pct}"
    )
    print(
        f"permute_ktiles={bool(args.permute_ktiles)} noise_scale={float(args.noise_scale)}\n"
    )

    results: List[Dict] = []
    failures: List[str] = []

    for sz in sizes:
        M = N = K = sz
        print(f"--- Shape M=N=K={sz} ---")
        torch.manual_seed(0)
        A = torch.randn((M, K), device=device, dtype=dtype)
        B = torch.randn((K, N), device=device, dtype=dtype)

        # Preallocate baseline workspace (NUM_SAMPLES=0)
        ws_base = prepare_fused_workspace(
            A,
            B,
            sigma=0,
            trace=TraceConfig(num_samples=0),
            permute_ktiles=bool(args.permute_ktiles),
            noise_scale=float(args.noise_scale),
        )

        # Preallocate fused workspace (NUM_SAMPLES=args.samples)
        ws_fused = prepare_fused_workspace(
            A,
            B,
            sigma=123,
            trace=TraceConfig(num_samples=int(args.samples)),
            permute_ktiles=bool(args.permute_ktiles),
            noise_scale=float(args.noise_scale),
        )

        base_ms = bench_ms(
            lambda: launch_gemm_fused_inplace(A, B, ws_base),
            warmup=args.warmup,
            iters=args.iters,
        )
        print(f"triton_baseline_gemm:    {base_ms:.3f} ms")

        fused_ms = bench_ms(
            lambda: launch_gemm_fused_inplace(A, B, ws_fused),
            warmup=args.warmup,
            iters=args.iters,
        )
        overhead = (fused_ms / base_ms - 1.0) * 100.0
        print(f"triton_fused_hash_gemm:  {fused_ms:.3f} ms  overhead={overhead:+.2f}%")

        if args.emit_digest:
            # One-off digest readback (not timed)
            payload = (
                ws_fused.hash_out.detach().cpu().numpy().tobytes(order="C")
                if ws_fused.hash_out.numel()
                else b""
            )
            import hashlib

            dig = hashlib.blake2s(payload, digest_size=16).digest()
            print(f"digest (16B) = {dig.hex()}  samples={len(ws_fused.coords3)}")

        if overhead > args.max_overhead_pct:
            failures.append(
                f"shape={sz} overhead {overhead:.2f}% > {args.max_overhead_pct}%"
            )

        results.append(
            {
                "shape": sz,
                "base_ms": base_ms,
                "fused_ms": fused_ms,
                "overhead_pct": overhead,
                "samples": int(args.samples),
                "permute_ktiles": bool(args.permute_ktiles),
                "noise_scale": float(args.noise_scale),
            }
        )
        print()

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(
                {"env": info, "results": results, "failures": failures}, f, indent=2
            )
        print(f"Wrote JSON: {args.json_out}\n")

    if failures:
        print("=== VALIDATION FAILED ===")
        for x in failures:
            print(" -", x)
        return 1

    print("=== VALIDATION PASSED ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
