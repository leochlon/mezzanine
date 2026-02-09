from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Sequence, TypeVar

import numpy as np

from .base import Symmetry

T = TypeVar("T")


@dataclass
class OrderSymmetryConfig:
    keep_first: int = 0  # keep first k items fixed, permute the rest


class OrderSymmetry(Symmetry):
    NAME = "order"
    DESCRIPTION = "Permutation/order symmetry for sequences (optionally keeping first k fixed)."

    def __init__(self, cfg: OrderSymmetryConfig):
        self.cfg = cfg

    def sample(self, x: Sequence[T], *, seed: int) -> list[T]:
        rng = np.random.default_rng(seed)
        x = list(x)
        k = int(self.cfg.keep_first)
        head = x[:k]
        tail = x[k:]
        perm = rng.permutation(len(tail)).tolist()
        return head + [tail[i] for i in perm]


# Register
from ..registry import SYMMETRIES
SYMMETRIES.register('order')( OrderSymmetry )
