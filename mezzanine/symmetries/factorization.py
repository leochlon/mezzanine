from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from .base import Symmetry


@dataclass
class FactorizationSymmetryConfig:
    """How to vary a factorization.

    In practice this is task-specific (e.g., tool-call orderings, reasoning decompositions),
    so this class provides a minimal general default: shuffle key-value pairs.
    """
    shuffle: bool = True


class FactorizationSymmetry(Symmetry):
    NAME = "factorization"
    DESCRIPTION = "Shuffle a dict's key-value decomposition (generic factorization symmetry)."

    def __init__(self, cfg: FactorizationSymmetryConfig):
        self.cfg = cfg

    def sample(self, x: Dict[str, Any], *, seed: int) -> List[Tuple[str, Any]]:
        items = list(x.items())
        if not self.cfg.shuffle or len(items) <= 1:
            return items
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(items)).tolist()
        return [items[i] for i in perm]


# Register
from ..registry import SYMMETRIES
SYMMETRIES.register('factorization')( FactorizationSymmetry )
