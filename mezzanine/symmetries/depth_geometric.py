"""D4 geometric symmetry for dense prediction (depth maps, segmentation, etc.).

The dihedral group D4 has 8 elements:
  0: identity
  1: rotate 90° CW
  2: rotate 180°
  3: rotate 270° CW
  4: vertical flip (top↔bottom)
  5: horizontal flip (left↔right)
  6: transpose (reflect across main diagonal)
  7: anti-transpose (reflect across anti-diagonal)

For depth maps, applying transform g to the input image and then applying
g⁻¹ to the predicted depth map should yield the same result if the model
is truly equivariant.  The warrant gap measures how much this fails.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..registry import SYMMETRIES
from .base import Symmetry


# ── D4 transforms on HWC / HW arrays ────────────────────────────────────


def _apply_d4(img: np.ndarray, idx: int) -> np.ndarray:
    """Apply D4 group element *idx* (0-7) to an image (H,W), (H,W,C), or (C,H,W)."""
    if idx == 0:
        return img
    elif idx == 1:  # rot90 CW
        return np.rot90(
            img, k=-1, axes=(-2, -1) if img.ndim == 3 and img.shape[0] <= 4 else (0, 1)
        )
    elif idx == 2:  # rot180
        return np.rot90(
            img, k=2, axes=(-2, -1) if img.ndim == 3 and img.shape[0] <= 4 else (0, 1)
        )
    elif idx == 3:  # rot270 CW
        return np.rot90(
            img, k=1, axes=(-2, -1) if img.ndim == 3 and img.shape[0] <= 4 else (0, 1)
        )
    elif idx == 4:  # vflip
        return np.flip(img, axis=-2 if img.ndim == 3 and img.shape[0] <= 4 else 0)
    elif idx == 5:  # hflip
        return np.flip(img, axis=-1 if img.ndim == 3 and img.shape[0] <= 4 else 1)
    elif idx == 6:  # transpose
        axes = (-2, -1) if img.ndim == 3 and img.shape[0] <= 4 else (0, 1)
        return np.swapaxes(img, *axes)
    elif idx == 7:  # anti-transpose = transpose then rot180
        axes = (-2, -1) if img.ndim == 3 and img.shape[0] <= 4 else (0, 1)
        t = np.swapaxes(img, *axes)
        return np.rot90(t, k=2, axes=axes)
    else:
        raise ValueError(f"D4 index must be 0-7, got {idx}")


def _inverse_d4(idx: int) -> int:
    """Return the D4 index of the inverse element."""
    # Inverses: 0→0, 1→3, 2→2, 3→1, 4→4, 5→5, 6→6, 7→7
    return {0: 0, 1: 3, 2: 2, 3: 1, 4: 4, 5: 5, 6: 6, 7: 7}[idx]


def apply_d4(img: np.ndarray, idx: int) -> np.ndarray:
    """Apply D4 transform, returning a contiguous copy."""
    return np.ascontiguousarray(_apply_d4(img, idx))


def inverse_d4(img: np.ndarray, idx: int) -> np.ndarray:
    """Apply the inverse D4 transform, returning a contiguous copy."""
    return np.ascontiguousarray(_apply_d4(img, _inverse_d4(idx)))


# ── Symmetry class ───────────────────────────────────────────────────────

D4_ALL = list(range(8))
D4_VFLIP_ONLY = [0, 4]
D4_FLIPS = [0, 4, 5]
D4_ROTATIONS = [0, 1, 2, 3]

# Named subgroups for easy ablation
SUBGROUPS = {
    "d4": D4_ALL,
    "vflip": D4_VFLIP_ONLY,
    "flips": D4_FLIPS,
    "rotations": D4_ROTATIONS,
    "identity": [0],
}


@dataclass
class DepthGeometricSymmetryConfig:
    """Configuration for depth geometric symmetry.

    subgroup: which D4 elements to use.  One of:
        "d4" (all 8), "vflip" (identity + vflip), "flips" (identity + vflip + hflip),
        "rotations" (4 rotations), "identity" (no-op, for baseline).
        Or a list of ints [0-7].
    """

    subgroup: str | list[int] = "d4"


class DepthGeometricSymmetry(Symmetry):
    """D4 geometric symmetry for dense spatial predictions.

    Use case: monocular depth models that have viewpoint bias (e.g.,
    'bottom of image = closer').  Orbit-averaging over geometric transforms
    cancels such biases.

    The `sample` method transforms the input image.  After running the model,
    use `inverse_d4` to map the prediction back to the original frame before
    averaging.
    """

    NAME = "depth_geometric"
    DESCRIPTION = (
        "D4 dihedral geometric transforms (rotations + reflections) for "
        "dense prediction equivariance. Subgroups available for ablation."
    )

    def __init__(self, cfg: DepthGeometricSymmetryConfig | None = None):
        if cfg is None:
            cfg = DepthGeometricSymmetryConfig()
        self.cfg = cfg
        if isinstance(cfg.subgroup, str):
            self.elements = SUBGROUPS[cfg.subgroup]
        else:
            self.elements = list(cfg.subgroup)

    @property
    def k(self) -> int:
        """Number of elements in the active subgroup."""
        return len(self.elements)

    def sample(self, x: np.ndarray, *, seed: int) -> np.ndarray:
        """Return a random D4 transform of x (from the active subgroup)."""
        rng = np.random.default_rng(seed)
        idx = self.elements[int(rng.integers(0, len(self.elements)))]
        return apply_d4(x, idx)

    def batch(self, x: np.ndarray, k: int = 0, *, seed: int = 0) -> list[np.ndarray]:
        """Return all elements of the active subgroup (deterministic orbit).

        For D4, k is ignored — we always return the full subgroup since it's small.
        """
        return [apply_d4(x, idx) for idx in self.elements]

    def inverse(self, x: np.ndarray, view_index: int) -> np.ndarray:
        """Apply inverse transform to map a prediction back to the original frame.

        Args:
            x: predicted depth/output in the transformed frame
            view_index: index into self.elements (NOT the raw D4 index)
        """
        d4_idx = self.elements[view_index]
        return inverse_d4(x, d4_idx)

    def orbit_average(self, predictions: list[np.ndarray]) -> np.ndarray:
        """Average predictions after inverse-transforming each back to the original frame.

        Args:
            predictions: list of depth maps, one per element in self.elements,
                         each in the transformed coordinate frame.

        Returns:
            Averaged depth map in the original coordinate frame.
        """
        assert len(predictions) == len(self.elements), (
            f"Expected {len(self.elements)} predictions, got {len(predictions)}"
        )
        aligned = [self.inverse(pred, i) for i, pred in enumerate(predictions)]
        return np.mean(aligned, axis=0)


# Register
SYMMETRIES.register("depth_geometric")(DepthGeometricSymmetry)
