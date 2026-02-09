from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


def latent_diversity(z: np.ndarray, pairs: int = 20000, seed: int = 0) -> Dict[str, Any]:
    """Quick collapse diagnostic: off-diagonal cosine similarity statistics.

    If offdiag_cos_mean is extremely high (e.g. > 0.995), your embedding space is effectively collapsed
    for this domain, and downstream dynamics conditioning will look meaningless in cosine metrics.
    """
    rng = np.random.default_rng(seed)
    n = z.shape[0]
    if n < 3:
        return {"offdiag_cos_mean": None, "offdiag_cos_std": None, "pairs_used": 0}
    i = rng.integers(0, n, size=(pairs,))
    j = rng.integers(0, n, size=(pairs,))
    m = i != j
    i = i[m]; j = j[m]
    cos = np.sum(z[i] * z[j], axis=1)
    return {
        "offdiag_cos_mean": float(np.mean(cos)),
        "offdiag_cos_std": float(np.std(cos)),
        "pairs_used": int(len(cos)),
    }
