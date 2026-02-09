from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

from ..core.cache import hash_dict
from ..core.deterministic import deterministic_subsample_indices
from .base import WorldAdapter


@dataclass
class LeRobotAdapterConfig:
    repo_id: str = "lerobot/aloha_mobile_cabinet"
    train_split: str = "train"
    test_split: str = "test"
    n_train: int = 4000
    n_test: int = 2000
    seed: int = 0

    # These are *metadata* here; decoding is recipe-specific
    camera_key: str = "observation.images.cam_high"
    action_key: str = "action"
    timestamp_key: str = "timestamp"


class LeRobotAdapter(WorldAdapter):
    """Adapter for LeRobot datasets hosted on HF.

    v1.0 note:
      - This adapter focuses on loading + stable subsampling + metadata.
      - Video decoding / frame selection is intentionally handled by recipes, because
        it depends heavily on the experiment (camera, delta, action windowing).
    """
    NAME = "lerobot"
    DESCRIPTION = "LeRobot adapter (HF-hosted robotics trajectories)."

    def __init__(self, cfg: LeRobotAdapterConfig):
        self.cfg = cfg

    def fingerprint(self) -> str:
        d = asdict(self.cfg)
        d["__class__"] = self.__class__.__name__
        return hash_dict(d)

    def load(self) -> Dict[str, Any]:
        try:
            from datasets import load_dataset  # type: ignore
        except Exception as e:
            raise RuntimeError("LeRobot adapter requires `datasets`. Install: pip install mezzanine[robotics]") from e

        ds = load_dataset(self.cfg.repo_id)
        train = ds[self.cfg.train_split]
        test = ds[self.cfg.test_split]

        tr_idx = deterministic_subsample_indices(len(train), min(self.cfg.n_train, len(train)), self.cfg.seed)
        te_idx = deterministic_subsample_indices(len(test), min(self.cfg.n_test, len(test)), self.cfg.seed + 1)

        # Return an indexable view rather than decoding videos here.
        meta = {
            "repo_id": self.cfg.repo_id,
            "n_train": len(tr_idx),
            "n_test": len(te_idx),
            "seed": self.cfg.seed,
            "camera_key": self.cfg.camera_key,
            "action_key": self.cfg.action_key,
            "timestamp_key": self.cfg.timestamp_key,
            "train_fingerprint": getattr(train, "_fingerprint", None),
            "test_fingerprint": getattr(test, "_fingerprint", None),
        }
        return {"train_ds": train, "test_ds": test, "train_idx": tr_idx, "test_idx": te_idx, "meta": meta}


# Register
from ..registry import ADAPTERS
ADAPTERS.register("lerobot")(LeRobotAdapter)
