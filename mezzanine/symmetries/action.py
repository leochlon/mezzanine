from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .base import Symmetry


@dataclass
class ActionShuffleConfig:
    mode: str = "permute_examples"  # or "permute_dims"


class ActionShuffleSymmetry(Symmetry):
    """Action-shuffle counterfactual.

    This symmetry is used as a *sanity check*: if your distilled dynamics truly depend on actions,
    then shuffling actions across examples should hurt.

    Note: shuffling actions is naturally a *batch-level* operation, so we also expose `shuffle_batch`.
    """
    NAME = "action_shuffle"
    DESCRIPTION = "Counterfactual symmetry: shuffle actions across examples (batch-level)."

    def __init__(self, cfg: ActionShuffleConfig = ActionShuffleConfig()):
        self.cfg = cfg

    def sample(self, x: np.ndarray, *, seed: int) -> np.ndarray:
        # For single items, return as-is. Use shuffle_batch for the real counterfactual.
        return x

    def shuffle_batch(self, a: np.ndarray, *, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        if self.cfg.mode == "permute_examples":
            perm = rng.permutation(len(a))
            return a[perm]
        if self.cfg.mode == "permute_dims":
            perm = rng.permutation(a.shape[1])
            return a[:, perm]
        raise ValueError(f"Unknown mode={self.cfg.mode}")


# Register
from ..registry import SYMMETRIES
SYMMETRIES.register('action_shuffle')( ActionShuffleSymmetry )
