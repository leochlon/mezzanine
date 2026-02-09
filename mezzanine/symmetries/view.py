from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np
from PIL import Image

from .base import Symmetry


@dataclass
class ViewSymmetryConfig:
    crop_frac: float = 0.9
    hflip: bool = True


class ViewSymmetry(Symmetry):
    """Simple vision symmetry: random crop + optional horizontal flip."""
    NAME = "view"
    DESCRIPTION = "Random crop + optional horizontal flip (vision view symmetry)."

    def __init__(self, cfg: ViewSymmetryConfig):
        self.cfg = cfg

    def sample(self, x: np.ndarray, *, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        img = Image.fromarray(x)
        w, h = img.size
        frac = float(self.cfg.crop_frac)
        cw, ch = int(w * frac), int(h * frac)
        if cw < 1 or ch < 1:
            return x
        x0 = int(rng.integers(0, max(1, w - cw + 1)))
        y0 = int(rng.integers(0, max(1, h - ch + 1)))
        img = img.crop((x0, y0, x0 + cw, y0 + ch)).resize((w, h))
        if self.cfg.hflip and bool(rng.integers(0, 2)):
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return np.asarray(img)


# Register
from ..registry import SYMMETRIES
SYMMETRIES.register('view')( ViewSymmetry )
