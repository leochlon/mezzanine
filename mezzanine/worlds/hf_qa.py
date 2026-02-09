from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from ..core.cache import hash_dict
from ..core.deterministic import deterministic_subsample_indices
from .base import WorldAdapter


@dataclass
class HFQADatasetAdapterConfig:
    """HuggingFace QA-style dataset adapter (passage + question + boolean/int answer).

    Designed for lightweight LLM distillation recipes (e.g., BoolQ).
    """
    dataset: str = "boolq"
    subset: Optional[str] = None
    train_split: str = "train"
    test_split: str = "validation"
    passage_field: str = "passage"
    question_field: str = "question"
    answer_field: str = "answer"  # bool or int
    n_train: int = 2048
    n_test: int = 512
    seed: int = 0


class HFQADatasetAdapter(WorldAdapter):
    NAME = "hf_qa"
    DESCRIPTION = "HF QA dataset adapter (passage+question+answer), e.g. boolq."

    def __init__(self, cfg: HFQADatasetAdapterConfig):
        self.cfg = cfg

    def fingerprint(self) -> str:
        return hash_dict(asdict(self.cfg))

    def load(self) -> Dict[str, Any]:
        try:
            from datasets import load_dataset  # type: ignore
        except Exception as e:
            raise RuntimeError("HF QA adapter requires `datasets`. Install: pip install mezzanine[datasets]") from e

        ds = load_dataset(self.cfg.dataset, self.cfg.subset) if self.cfg.subset else load_dataset(self.cfg.dataset)
        train = ds[self.cfg.train_split]
        test = ds[self.cfg.test_split]

        tr_idx = deterministic_subsample_indices(len(train), min(self.cfg.n_train, len(train)), self.cfg.seed)
        te_idx = deterministic_subsample_indices(len(test), min(self.cfg.n_test, len(test)), self.cfg.seed + 1)

        def _ex(row: Any) -> Dict[str, Any]:
            passage = row[self.cfg.passage_field]
            question = row[self.cfg.question_field]
            ans = row[self.cfg.answer_field]
            # bool -> int
            if isinstance(ans, bool):
                y = int(ans)
            else:
                y = int(ans)
            return {"passage": passage, "question": question, "label": y}

        train_ex = [_ex(train[int(i)]) for i in tr_idx]
        test_ex = [_ex(test[int(i)]) for i in te_idx]

        meta = {
            "dataset": self.cfg.dataset,
            "subset": self.cfg.subset,
            "train_split": self.cfg.train_split,
            "test_split": self.cfg.test_split,
            "n_train": len(train_ex),
            "n_test": len(test_ex),
            "passage_field": self.cfg.passage_field,
            "question_field": self.cfg.question_field,
            "answer_field": self.cfg.answer_field,
            "seed": self.cfg.seed,
        }
        try:
            sample = train_ex[0]
            import hashlib, json as _json
            meta["sample_sha256_first"] = hashlib.sha256(_json.dumps(sample, sort_keys=True).encode("utf-8")).hexdigest()
        except Exception:
            pass

        return {"train": train_ex, "test": test_ex, "meta": meta}


# Register
from ..registry import ADAPTERS
ADAPTERS.register("hf_qa")(HFQADatasetAdapter)
