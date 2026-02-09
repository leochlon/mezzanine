from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np


def stable_json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def hash_dict(d: Dict[str, Any]) -> str:
    return sha256_hex(stable_json_dumps(d))


@dataclass
class LatentCacheConfig:
    cache_dir: Path
    enabled: bool = True
    compress: bool = True


class LatentCache:
    """Disk cache for expensive encoder outputs.

    Key design:
      key = H( world_fingerprint || encoder_fingerprint || split || tag )

    Stores:
      - {key}.npz : array + meta
    """

    def __init__(self, cfg: LatentCacheConfig):
        self.cfg = cfg
        self.cfg.cache_dir.mkdir(parents=True, exist_ok=True)

    def make_key(
        self,
        *,
        world_fingerprint: str,
        encoder_fingerprint: str,
        split: str,
        tag: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        payload = {
            "world": world_fingerprint,
            "encoder": encoder_fingerprint,
            "split": split,
            "tag": tag,
            "extra": extra or {},
        }
        return hash_dict(payload)

    def _path(self, key: str) -> Path:
        return self.cfg.cache_dir / f"{key}.npz"

    def get(self, key: str) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        if not self.cfg.enabled:
            return None
        p = self._path(key)
        if not p.exists():
            return None
        data = np.load(p, allow_pickle=True)
        arr = data["arr"].astype(np.float32)
        meta = data["meta"].item()  # type: ignore[assignment]
        return arr, meta

    def put(self, key: str, arr: np.ndarray, meta: Dict[str, Any]) -> Path:
        if not self.cfg.enabled:
            return self._path(key)
        p = self._path(key)
        if self.cfg.compress:
            np.savez_compressed(p, arr=arr.astype(np.float32), meta=np.array(meta, dtype=object))
        else:
            np.savez(p, arr=arr.astype(np.float32), meta=np.array(meta, dtype=object))
        return p

    def get_or_compute(
        self,
        key: str,
        compute_fn: Callable[[], np.ndarray],
        *,
        meta: Optional[Dict[str, Any]] = None,
        force: bool = False,
    ) -> np.ndarray:
        if (not force) and self.cfg.enabled:
            hit = self.get(key)
            if hit is not None:
                return hit[0]
        arr = compute_fn()
        self.put(key, arr, meta or {})
        return arr
