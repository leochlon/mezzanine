from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import List, Literal, Sequence, Tuple

import numpy as np
import torch

from .hash128 import TCHash128
from .rng import derive_seed_u64, rand_u64, torch_sign_vector


@dataclass(frozen=True)
class POIConfig:
    """Configuration for a deterministic activation transcript (sample-and-hash)."""

    samples_per_tensor: int = 128
    quant: Literal["fp16", "fp32", "int8"] = "fp16"
    include_shape: bool = True


def activation_transcript(
    activations: Sequence[torch.Tensor],
    *,
    sigma: int | bytes | str,
    cfg: POIConfig = POIConfig(),
) -> Tuple[bytes, List[torch.Tensor]]:
    """Compute a deterministic transcript hash of a subset of activations.

    This is a lightweight "trace commitment" helper: it hashes a deterministic sample
    of the provided tensors (optionally including shape metadata).

    Returns
    -------
    (digest_bytes, sampled_indices_per_tensor)
    """
    h = TCHash128(seed=sigma)
    sampled: List[torch.Tensor] = []

    base_seed = derive_seed_u64(sigma, domain="poi/base")

    for t_idx, act in enumerate(activations):
        a = act.detach()
        flat = a.reshape(-1)
        n = flat.numel()
        if n == 0:
            sampled.append(torch.empty((0,), dtype=torch.int64))
            continue

        seed_u64 = int(base_seed ^ (t_idx * 0x9E3779B97F4A7C15)) & ((1 << 64) - 1)
        u = rand_u64((cfg.samples_per_tensor,), seed_u64=seed_u64)
        idx = (u % np.uint64(n)).astype(np.int64)
        idx_t = torch.from_numpy(idx).to(dtype=torch.int64)

        if cfg.include_shape:
            h.update_bytes(struct.pack("<I", t_idx))
            h.update_bytes(struct.pack("<I", a.ndim))
            for d in a.shape:
                h.update_bytes(struct.pack("<I", int(d)))

        # Hash sampled values (gather on CPU)
        vals = flat[idx_t].to(dtype=torch.float32, device="cpu")
        h.update_tensor(vals, quant=cfg.quant)
        sampled.append(idx_t)

    return h.digest(), sampled
