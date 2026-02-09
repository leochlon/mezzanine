from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from ..core.cache import hash_dict
from ..core.deterministic import deterministic_subsample_indices
from .base import WorldAdapter


@dataclass
class HFDatasetAdapterConfig:
    dataset: str
    subset: Optional[str] = None
    train_split: str = "train"
    test_split: str = "test"
    text_field: str = "text"
    label_field: str = "label"
    n_train: int = 20000
    n_test: int = 4000
    seed: int = 0


class HFDatasetAdapter(WorldAdapter):
    """Generic HuggingFace `datasets` adapter for supervised text classification-like tasks."""
    NAME = "hf_dataset"
    DESCRIPTION = "HuggingFace datasets adapter (text + label)."

    def __init__(self, cfg: HFDatasetAdapterConfig):
        self.cfg = cfg

    def fingerprint(self) -> str:
        d = asdict(self.cfg)
        d["__class__"] = self.__class__.__name__
        return hash_dict(d)

    def load(self) -> Dict[str, Any]:
        try:
            from datasets import load_dataset  # type: ignore
        except Exception as e:
            raise RuntimeError("HF datasets adapter requires `datasets`. Install: pip install mezzanine[datasets]") from e

        ds = load_dataset(self.cfg.dataset, self.cfg.subset) if self.cfg.subset else load_dataset(self.cfg.dataset)
        train = ds[self.cfg.train_split]
        test = ds[self.cfg.test_split]

        tr_idx = deterministic_subsample_indices(len(train), min(self.cfg.n_train, len(train)), self.cfg.seed)
        te_idx = deterministic_subsample_indices(len(test), min(self.cfg.n_test, len(test)), self.cfg.seed + 1)

        train_ex = [{"text": train[int(i)][self.cfg.text_field], "label": int(train[int(i)][self.cfg.label_field])} for i in tr_idx]
        test_ex = [{"text": test[int(i)][self.cfg.text_field], "label": int(test[int(i)][self.cfg.label_field])} for i in te_idx]

        meta = {
            "dataset": self.cfg.dataset,
            "subset": self.cfg.subset,
            "train_split": self.cfg.train_split,
            "test_split": self.cfg.test_split,
            "n_train": len(train_ex),
            "n_test": len(test_ex),
            "seed": self.cfg.seed,
        }
        # If available, include HF dataset fingerprints for traceability
        try:
            meta["train_fingerprint"] = getattr(train, "_fingerprint", None)
            meta["test_fingerprint"] = getattr(test, "_fingerprint", None)
        except Exception:
            pass

        return {"train": train_ex, "test": test_ex, "meta": meta}


# Register
from ..registry import ADAPTERS
ADAPTERS.register("hf_dataset")(HFDatasetAdapter)
