"""Symmetry plugin template.

A symmetry defines a family of "views" of the same underlying instance,
e.g., crop/flip, order/permutation, factorization, action-shuffle, etc.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, TypeVar

import numpy as np

from mezzanine.symmetries.base import Symmetry
from mezzanine.registry import SYMMETRIES

T = TypeVar("T")


@dataclass
class MySymmetryConfig:
    strength: float = 1.0


@SYMMETRIES.register("my_symmetry")
class MySymmetry(Symmetry):
    NAME = "my_symmetry"
    DESCRIPTION = "One-line description of what this symmetry does."

    def __init__(self, cfg: MySymmetryConfig):
        self.cfg = cfg

    def sample(self, x: Sequence[T], *, seed: int) -> list[T]:
        rng = np.random.default_rng(seed)
        x = list(x)
        # TODO: implement transform using rng
        return x
