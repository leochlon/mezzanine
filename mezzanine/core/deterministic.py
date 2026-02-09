from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


def seed_everything(seed: int) -> None:
    """Best-effort global seeding for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if _HAS_TORCH:
        torch.manual_seed(seed)  # type: ignore[name-defined]
        if torch.cuda.is_available():  # type: ignore[name-defined]
            torch.cuda.manual_seed_all(seed)  # type: ignore[name-defined]
        # Determinism knobs (may reduce speed)
        torch.backends.cudnn.deterministic = True  # type: ignore[name-defined]
        torch.backends.cudnn.benchmark = False  # type: ignore[name-defined]


def deterministic_subsample_indices(n_total: int, n: int, seed: int) -> np.ndarray:
    """Return a deterministic subset of indices [0..n_total-1] of size n."""
    if n > n_total:
        raise ValueError(f"n={n} cannot exceed n_total={n_total}")
    rng = np.random.default_rng(seed)
    return rng.choice(n_total, size=n, replace=False)
