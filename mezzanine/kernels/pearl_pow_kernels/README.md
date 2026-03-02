# pearl-pow-kernels

Small, testable reference implementations for "GEMM + transcript hash" style experiments
used by `mezzanine/kernels/overhead_benchmark.py`.

The modules here follow the same basic shape:

1) deterministically transform/quantize `(A, B)` from a seed `sigma`,
2) compute `C = A @ B` (or an approximate surrogate), and
3) compute a sampled 128-bit transcript hash of intermediate tiles.

Included modules:

- `rot_gemm`: randomized Hadamard-style encoding + GEMM + sampled trace hash (optional Triton fused path)
- `qnoise_gemm`: add deterministic noise + quantize (`float8` or `int8`) + GEMM + sampled trace hash
- `fp4_gemm`: groupwise FP4-like quantization with optional scale jitter + GEMM + sampled trace hash
- `train_pow`: training helpers (e.g. `PowLinear`) that route matmul through the schemes
- `hash128`, `trace`, `rng`: incremental hash, trace sampling, deterministic RNG helpers
- `poi`: activation transcript helper (hash a sampled subset of activations)

## Quickstart

From this directory:

```bash
pip install -e .
pytest
```

## Notes

- Transcript hashes are intended for determinism/regression checks and toy "trace commitment" experiments;
  they are not a cryptographic security primitive.
- The Triton kernel is optional and only used when Triton + CUDA are available.
