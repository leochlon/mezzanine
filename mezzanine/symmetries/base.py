from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Dict, Optional


class Symmetry(abc.ABC):
    """A symmetry family: different 'views' of the same underlying evidence.

    Examples:
      - image transforms (crop/flip, camera choice)
      - order / permutation of a list of statements
      - factorization / tool-call ordering
      - action representation choices (windowing, offsets)

    Symmetries are used in two places:
      - measurement: estimate how unstable a model is under symmetry variation
      - distillation: train a student to approximate the symmetry-marginal expectation
    """

    NAME: str = "symmetry"
    DESCRIPTION: str = ""

    @abc.abstractmethod
    def sample(self, x: Any, *, seed: int) -> Any:
        """Produce one symmetry realization of x."""
        raise NotImplementedError

    def batch(self, x: Any, k: int, *, seed: int) -> list[Any]:
        """Produce k symmetry samples deterministically."""
        return [self.sample(x, seed=seed + i) for i in range(k)]
