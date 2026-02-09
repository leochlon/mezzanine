from __future__ import annotations

"""Public API helpers.

Recipes are the reproducible entrypoint, but many users want a simple function call
to measure a warrant gap (instability under symmetries) on their own data.

The functions here are intentionally minimal and composable.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from .core.cache import LatentCache
from .encoders.base import Encoder
from .predictors.base import Predictor
from .pipelines.text_distill import warrant_gap_from_views


@dataclass
class MeasureResult:
    """Returned by `measure(...)`. Stored as JSON via `asdict()` if needed."""
    n: int
    k: int
    mode: str  # "embedding" or "prediction"
    metrics: Dict[str, float]


def _l2norm(Z: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(Z, axis=1, keepdims=True)
    return Z / np.clip(d, 1e-9, None)


def _cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = _l2norm(a)
    b = _l2norm(b)
    return (a * b).sum(axis=1)


def measure(
    *,
    inputs: Sequence[Any],
    make_views: Callable[[Sequence[Any], int, int], List[List[Any]]],
    encoder: Encoder,
    predictor: Optional[Predictor] = None,
    k: int = 16,
    seed: int = 0,
    cache: Optional[LatentCache] = None,
    world_fingerprint: Optional[str] = None,
    split: str = "test",
    tag: str = "measure",
) -> MeasureResult:
    """Unified warrant-gap measurement.

    Args:
        inputs: base inputs (N items).
        make_views: function that returns K view-lists (each length N).
                   Signature: (inputs, seed, k) -> views
                   where views[j][i] is the j-th symmetry view of example i.
        encoder: maps view inputs -> embeddings.
        predictor: optional; if provided, measures instability of *predicted probabilities*.
                   If omitted, measures instability of *embeddings*.
        k: number of views (including canonical view 0, if you choose).
        seed: RNG seed passed into make_views.
        cache/world_fingerprint/split/tag: optional latent caching.

    Returns:
        MeasureResult with simple scalar metrics.
    """
    N = len(inputs)
    views = make_views(inputs, seed, k)
    if len(views) != k:
        raise ValueError(f"make_views returned {len(views)} views, expected k={k}")

    enc_fp = encoder.fingerprint()
    world_fp = world_fingerprint or "no_world"

    Z_views: List[np.ndarray] = []
    for j in range(k):
        tag_j = f"{tag}_view{j}"
        if cache is not None:
            key = cache.make_key(world_fingerprint=world_fp, encoder_fingerprint=enc_fp, split=split, tag=tag_j, extra={"k": k, "seed": seed})
            got = cache.get(key)
            if got is not None:
                Zj, _meta = got
            else:
                Zj = encoder.encode(list(views[j]))
                cache.put(key, Zj, meta={"n": len(views[j]), "view": j})
        else:
            Zj = encoder.encode(list(views[j]))
        Z_views.append(Zj)

    if predictor is None:
        # Embedding invariance metrics
        cos01 = _cos(Z_views[0], Z_views[1]) if k >= 2 else np.ones(N, dtype=np.float32)
        cos_mean = float(cos01.mean())
        cos_std = float(cos01.std())

        metrics = {"cos_mean_view0_vs_1": cos_mean, "cos_std_view0_vs_1": cos_std}
        return MeasureResult(n=N, k=k, mode="embedding", metrics=metrics)

    # Prediction instability metrics
    # Predictor consumes raw inputs by default; but many predictors will use encoder internally.
    # Here we assume predictor can score each view directly.
    P_views = np.stack([predictor.predict_proba(views[j]) for j in range(k)], axis=1)  # [N,K,C]
    gap = warrant_gap_from_views(P_views)
    return MeasureResult(n=N, k=k, mode="prediction", metrics=gap)
