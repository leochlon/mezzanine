from __future__ import annotations

"""MD-specific symmetries for Lennard–Jones (LJ) atomic systems.

These symmetries are *physically irrelevant* degrees of freedom for an isotropic
LJ fluid in a cubic periodic box:

- **Permutation**: particle indices are exchangeable (identical atoms).
- **Global SE(3)**: rigid rotation + translation of the configuration.
- **Periodic image choice**: each particle can be represented by any periodic image.
- **Coordinate noise**: finite-precision / measurement noise (use small sigma).

Each symmetry maps an example dict:
  {
    "pos": np.ndarray[N,3] float32/float64,
    "box": float (cubic L),
    ...
  }
to a new example dict with transformed coordinates.
"""

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from .base import Symmetry


def _wrap(x: np.ndarray, L: float) -> np.ndarray:
    return np.mod(x, float(L))


def _random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    """Uniform random rotation in SO(3) via random quaternion."""
    u1 = float(rng.random())
    u2 = float(rng.random())
    u3 = float(rng.random())
    q1 = np.sqrt(1.0 - u1) * np.sin(2.0 * np.pi * u2)
    q2 = np.sqrt(1.0 - u1) * np.cos(2.0 * np.pi * u2)
    q3 = np.sqrt(u1) * np.sin(2.0 * np.pi * u3)
    q4 = np.sqrt(u1) * np.cos(2.0 * np.pi * u3)
    # quaternion (x,y,z,w) = (q1,q2,q3,q4)
    x, y, z, w = q1, q2, q3, q4
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    return R


@dataclass
class LJPermutationConfig:
    """Randomly permute particle indices."""


class LJPermutationSymmetry(Symmetry):
    NAME = "lj_permutation"
    DESCRIPTION = "Permute particle indices (exchangeable identical atoms)."

    def __init__(self, cfg: LJPermutationConfig | None = None):
        self.cfg = cfg or LJPermutationConfig()

    def sample(self, x: Any, *, seed: int) -> Dict[str, Any]:
        rng = np.random.default_rng(seed)
        pos = np.asarray(x["pos"])
        perm = rng.permutation(pos.shape[0])
        out = dict(x)
        out["pos"] = pos[perm].copy()
        return out


@dataclass
class LJSE3Config:
    rotate: bool = True
    translate: bool = True
    wrap: bool = True


class LJSE3Symmetry(Symmetry):
    NAME = "lj_se3"
    DESCRIPTION = "Global rigid rotation + translation (SE(3)) in a cubic box."

    def __init__(self, cfg: LJSE3Config | None = None):
        self.cfg = cfg or LJSE3Config()

    def sample(self, x: Any, *, seed: int) -> Dict[str, Any]:
        rng = np.random.default_rng(seed)
        pos = np.asarray(x["pos"], dtype=np.float64)
        L = float(x["box"])

        out = dict(x)

        # Rotate around box center so we don't need an unwrapped COM.
        center = np.array([0.5 * L, 0.5 * L, 0.5 * L], dtype=np.float64)
        p = pos - center
        if self.cfg.rotate:
            R = _random_rotation_matrix(rng)
            p = p @ R.T
        p = p + center

        if self.cfg.translate:
            t = rng.uniform(0.0, L, size=(1, 3))
            p = p + t

        if self.cfg.wrap:
            p = _wrap(p, L)

        out["pos"] = p.astype(np.float32)
        return out


@dataclass
class LJImageChoiceConfig:
    max_image: int = 1
    wrap_after: bool = False


class LJImageChoiceSymmetry(Symmetry):
    NAME = "lj_image_choice"
    DESCRIPTION = "Randomly choose a periodic image for each particle (per-particle integer box shifts)."

    def __init__(self, cfg: LJImageChoiceConfig | None = None):
        self.cfg = cfg or LJImageChoiceConfig()

    def sample(self, x: Any, *, seed: int) -> Dict[str, Any]:
        rng = np.random.default_rng(seed)
        pos = np.asarray(x["pos"], dtype=np.float64)
        L = float(x["box"])
        m = int(self.cfg.max_image)
        if m < 0:
            raise ValueError("max_image must be >= 0")

        # Each particle can be represented by any periodic image.
        # integers() uses a half-open interval [low, high); this yields [-m, ..., m].
        shifts = rng.integers(-m, m + 1, size=pos.shape)
        p = pos + shifts.astype(np.float64) * L
        if self.cfg.wrap_after:
            p = _wrap(p, L)

        out = dict(x)
        out["pos"] = p.astype(np.float32)
        return out


@dataclass
class LJCoordinateNoiseConfig:
    sigma: float = 0.01
    wrap: bool = True


class LJCoordinateNoiseSymmetry(Symmetry):
    NAME = "lj_coord_noise"
    DESCRIPTION = "Add small Gaussian noise to positions (measurement/roundoff symmetry)."

    def __init__(self, cfg: LJCoordinateNoiseConfig | None = None):
        self.cfg = cfg or LJCoordinateNoiseConfig()

    def sample(self, x: Any, *, seed: int) -> Dict[str, Any]:
        rng = np.random.default_rng(seed)
        pos = np.asarray(x["pos"], dtype=np.float64)
        L = float(x["box"])
        sigma = float(self.cfg.sigma)
        if sigma < 0:
            raise ValueError("sigma must be >= 0")
        p = pos + rng.normal(0.0, sigma, size=pos.shape)
        if self.cfg.wrap:
            p = _wrap(p, L)
        out = dict(x)
        out["pos"] = p.astype(np.float32)
        return out


# Register
from ..registry import SYMMETRIES

SYMMETRIES.register("lj_permutation")(LJPermutationSymmetry)
SYMMETRIES.register("lj_se3")(LJSE3Symmetry)
SYMMETRIES.register("lj_image_choice")(LJImageChoiceSymmetry)
SYMMETRIES.register("lj_coord_noise")(LJCoordinateNoiseSymmetry)
