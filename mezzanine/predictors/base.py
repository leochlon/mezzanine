from __future__ import annotations

import abc
from typing import Any, Dict, List, Sequence

import numpy as np


class Predictor(abc.ABC):
    """A Predictor produces probability distributions over discrete outcomes.

    This is optional: many Mezzanine measurements operate purely on embeddings.

    Typical uses:
      - measure belief instability under symmetries (TV / Jensen gap)
      - define distillation targets as symmetry-marginalized teacher distributions
    """

    @abc.abstractmethod
    def fingerprint(self) -> str:
        """Stable fingerprint for caching/reproducibility."""
        raise NotImplementedError

    @abc.abstractmethod
    def predict_proba(self, inputs: Sequence[Any], **kwargs: Any) -> np.ndarray:
        """Return probabilities with shape [N, C]."""
        raise NotImplementedError
