"""Encoder plugin template.

Encoders map raw inputs (texts/images/observations) -> embeddings used for:
  - measuring warrant gaps (instability under symmetries)
  - distilling symmetry-marginalized invariants
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Any

import numpy as np

from mezzanine.core.cache import hash_dict
from mezzanine.encoders.base import Encoder
from mezzanine.registry import ENCODERS


@dataclass
class MyEncoderConfig:
    checkpoint: str = "my/checkpoint"
    batch_size: int = 32


@ENCODERS.register("my_encoder")
class MyEncoder(Encoder):
    NAME = "my_encoder"
    DESCRIPTION = "One-line description of what this encoder wraps."

    def __init__(self, cfg: MyEncoderConfig, device: str = "cuda"):
        self.cfg = cfg
        self.device = device
        # TODO: load model weights, set eval() etc.

    def fingerprint(self) -> str:
        d = asdict(self.cfg)
        d["__class__"] = self.__class__.__name__
        return hash_dict(d)

    def encode(self, inputs: List[Any]) -> np.ndarray:  # type: ignore[name-defined]
        # TODO: implement batching, device transfer, fp16 if desired
        # Return shape: [N, D] float32, preferably L2-normalized
        raise NotImplementedError
