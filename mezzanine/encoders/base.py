from __future__ import annotations

import abc
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import numpy as np


class Encoder(abc.ABC):
    """Frozen backbone -> embedding vectors.

    Encoders must provide:
      - a serializable config dict (for hashing / caching)
      - an encode method
    """

    NAME: str = "encoder"
    DESCRIPTION: str = ""

    @abc.abstractmethod
    def fingerprint(self) -> str:
        """Stable hash identifying the encoder config (NOT including the world)."""
        raise NotImplementedError

    @abc.abstractmethod
    def encode(self, batch: Any) -> np.ndarray:
        raise NotImplementedError
