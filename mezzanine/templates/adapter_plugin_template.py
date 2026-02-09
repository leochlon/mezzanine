"""Adapter plugin template.

Copy this file into your project (or into mezzanine/worlds/) and edit.

An adapter exposes a "world" to Mezzanine as (train, test) examples + meta.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from mezzanine.core.cache import hash_dict
from mezzanine.core.deterministic import deterministic_subsample_indices
from mezzanine.worlds.base import WorldAdapter
from mezzanine.registry import ADAPTERS


@dataclass
class MyAdapterConfig:
    # --- Minimal fields ---
    seed: int = 0
    n_train: int = 1000
    n_test: int = 200
    # TODO: add dataset-specific fields (paths, env IDs, split names, etc.)


@ADAPTERS.register("my_adapter")
class MyAdapter(WorldAdapter):
    """Describe what your adapter loads and what each example looks like."""

    NAME = "my_adapter"
    DESCRIPTION = "One-line description of what this adapter provides."

    def __init__(self, cfg: MyAdapterConfig):
        self.cfg = cfg

    def fingerprint(self) -> str:
        """A stable fingerprint used for caching + reproducibility.

        Include *everything* that changes the realized data distribution:
          - dataset name/version
          - split names
          - deterministic subsampling parameters
          - preprocessing flags
          - random seed
        """
        d = asdict(self.cfg)
        d["__class__"] = self.__class__.__name__
        return hash_dict(d)

    def load(self) -> Dict[str, Any]:
        """Return: {"train": [...], "test": [...], "meta": {...}}.

        Each example should be a plain dict with JSON-serializable fields.

        Common patterns:
          - supervised text: {"text": str, "label": int}
          - supervised vision: {"image": np.ndarray(H,W,3), "label": int}
          - transitions: {"obs_t": ..., "obs_tp": ..., "action": ...}
        """
        # TODO: implement your loading logic.
        train: List[Dict[str, Any]] = []
        test: List[Dict[str, Any]] = []
        meta: Dict[str, Any] = {"note": "fill me"}

        # Example deterministic subsampling:
        # idx = deterministic_subsample_indices(len(full), self.cfg.n_train, self.cfg.seed)

        return {"train": train, "test": test, "meta": meta}
