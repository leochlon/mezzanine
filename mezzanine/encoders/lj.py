from __future__ import annotations

"""Encoders (frozen descriptors) for Lennard–Jones (LJ) configurations.

These are **deterministic, physics-aligned** encoders intended for training
experiments:

- `LJFlattenEncoder`: baseline that flattens coordinates (sensitive to SE(3) and permutation).
- `LJRDFEncoder`: radial distribution function-like features from pair distances.

The encoders take as input examples shaped like:
  {"pos": np.ndarray[N,3], "box": float, ...}
and return a float32 embedding array of shape [B, D].
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from scipy.spatial import cKDTree

from ..core.cache import hash_dict
from .base import Encoder


def _wrap(x: np.ndarray, L: float) -> np.ndarray:
    return np.mod(x, float(L))


def _minimum_image(dr: np.ndarray, L: float) -> np.ndarray:
    return dr - float(L) * np.round(dr / float(L))


@dataclass
class LJFlattenEncoderConfig:
    """Flatten coordinates into a fixed vector.

    Notes
    -----
    This is intentionally *not* invariant to:
      - particle permutations
      - global rotations

    It is meant as a baseline to demonstrate large warrant gaps under
    realistic MD symmetries.
    """

    center: str = "box"  # "none" | "box" | "mean"
    scale_by_box: bool = False


class LJFlattenEncoder(Encoder):
    NAME = "lj_flatten"
    DESCRIPTION = "Baseline: flatten particle coordinates (variant under SE(3)/permutation)."

    def __init__(self, cfg: LJFlattenEncoderConfig):
        if cfg.center not in ("none", "box", "mean"):
            raise ValueError("center must be one of: none, box, mean")
        self.cfg = cfg

    def fingerprint(self) -> str:
        d = asdict(self.cfg)
        d["__class__"] = self.__class__.__name__
        return hash_dict(d)

    def encode(self, inputs: List[Any]) -> np.ndarray:
        if len(inputs) == 0:
            return np.zeros((0, 0), dtype=np.float32)
        # Assume fixed N across dataset.
        first = np.asarray(inputs[0]["pos"])
        N = int(first.shape[0])
        out = np.zeros((len(inputs), 3 * N), dtype=np.float32)
        for b, ex in enumerate(inputs):
            pos = np.asarray(ex["pos"], dtype=np.float32)
            L = float(ex["box"])
            if self.cfg.center == "box":
                pos = pos - 0.5 * L
            elif self.cfg.center == "mean":
                pos = pos - pos.mean(axis=0, keepdims=True)
            if self.cfg.scale_by_box:
                pos = pos / max(1e-9, L)
            out[b] = pos.reshape(-1)
        return out.astype(np.float32, copy=False)


@dataclass
class LJRDFEncoderConfig:
    """Radial distribution function-like descriptor.

    This encoder computes a histogram of pair distances (with cubic PBC and
    minimum-image convention) and converts it to an RDF-style feature vector.

    Compared to LJFlattenEncoder, this is:
      - translation invariant
      - rotation invariant
      - permutation invariant

    It is a standard, realistic physics descriptor for simple fluids.
    """

    n_bins: int = 128
    r_max: float = 3.5
    # If True, append log(rho) as an extra scalar.
    include_log_rho: bool = False
    # If True, return raw counts instead of g(r) normalization.
    counts_only: bool = False


class LJRDFEncoder(Encoder):
    NAME = "lj_rdf"
    DESCRIPTION = "RDF descriptor from pair distances (SE(3)+perm invariant)."

    def __init__(self, cfg: LJRDFEncoderConfig):
        if cfg.n_bins <= 0:
            raise ValueError("n_bins must be > 0")
        if cfg.r_max <= 0:
            raise ValueError("r_max must be > 0")
        self.cfg = cfg
        self._edges = np.linspace(0.0, float(cfg.r_max), int(cfg.n_bins) + 1, dtype=np.float64)
        self._dr = float(self._edges[1] - self._edges[0])

    def fingerprint(self) -> str:
        d = asdict(self.cfg)
        d["__class__"] = self.__class__.__name__
        return hash_dict(d)

    def _rdf_one(self, pos: np.ndarray, L: float) -> np.ndarray:
        pos = np.asarray(pos, dtype=np.float64)
        L = float(L)
        pos = _wrap(pos, L)

        N = int(pos.shape[0])
        V = L**3
        rho = float(N) / V

        # Effective max distance under minimum image is L/2.
        r_eff = min(float(self.cfg.r_max), 0.5 * L)

        # Use cKDTree to avoid O(N^2) work for moderate r_eff.
        tree = cKDTree(pos, boxsize=L)
        pairs = tree.query_pairs(r=r_eff, output_type="ndarray")
        if pairs.size == 0:
            counts = np.zeros((int(self.cfg.n_bins),), dtype=np.float64)
        else:
            i = pairs[:, 0]
            j = pairs[:, 1]
            dr = pos[i] - pos[j]
            dr = _minimum_image(dr, L)
            r = np.sqrt((dr * dr).sum(axis=1))
            # Histogram only up to r_eff; bins beyond r_eff are zero.
            counts, _ = np.histogram(r, bins=self._edges)
            counts = counts.astype(np.float64)

        if self.cfg.counts_only:
            feat = counts
        else:
            # Convert to g(r).
            # counts are unordered pairs in shells.
            # g(r) = counts / (0.5 * N * rho * 4π r^2 dr)
            centers = 0.5 * (self._edges[:-1] + self._edges[1:])
            shell_vol = 4.0 * np.pi * centers**2 * self._dr
            denom = 0.5 * float(N) * float(rho) * shell_vol
            feat = counts / np.clip(denom, 1e-12, None)

            # For r > r_eff, counts are zero; keep as zero.

        feat = feat.astype(np.float32)
        if self.cfg.include_log_rho:
            feat = np.concatenate([feat, np.array([np.log(max(1e-12, rho))], dtype=np.float32)], axis=0)
        return feat

    def encode(self, inputs: List[Any]) -> np.ndarray:
        if len(inputs) == 0:
            return np.zeros((0, 0), dtype=np.float32)
        feats = [self._rdf_one(ex["pos"], float(ex["box"])) for ex in inputs]
        return np.stack(feats, axis=0).astype(np.float32, copy=False)


# Register
from ..registry import ENCODERS

ENCODERS.register("lj_flatten")(LJFlattenEncoder)
ENCODERS.register("lj_rdf")(LJRDFEncoder)
