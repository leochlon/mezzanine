from __future__ import annotations

import abc
from typing import Any, Dict, Optional


class WorldAdapter(abc.ABC):
    """A world/data adapter.

    A "world" is any source of evidence:
      - documents / text corpora
      - code execution traces
      - games / simulators / physics
      - robotics trajectories
      - multimodal datasets

    WorldAdapters provide:
      - a stable fingerprint (for caching)
      - a way to materialize the split into in-memory examples (or indexable views)

    For v1.0 we keep the API intentionally small; specific recipes/pipelines may require
    additional fields in the returned dicts (e.g. images, actions, labels).
    """

    NAME: str = "world"
    DESCRIPTION: str = ""

    @abc.abstractmethod
    def fingerprint(self) -> str:
        """Stable identifier for this world configuration + subsampling."""
        raise NotImplementedError

    @abc.abstractmethod
    def load(self) -> Dict[str, Any]:
        """Materialize data.

        Convention (not enforced):
          - returns {"train": [...], "test": [...], "meta": {...}}
        """
        raise NotImplementedError
