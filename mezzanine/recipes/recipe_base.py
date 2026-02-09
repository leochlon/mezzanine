from __future__ import annotations

import abc
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from ..core import (
    LatentCache,
    LatentCacheConfig,
    deep_update,
    load_config,
    make_logger,
    seed_everything,
)


@dataclass
class RunContext:
    out_dir: Path
    config: Dict[str, Any]
    seed: int
    cache: Optional[LatentCache]
    logger: Any  # BaseLogger (kept Any to avoid import cycles in dataclasses)


class Recipe(abc.ABC):
    """A runnable preset.

    Recipes are intentionally thin: they glue together adapters + encoders + symmetries + pipelines.

    Config precedence (recommended):
      argparse defaults  <  --config file  <  --overrides (CLI)  <  explicit CLI flags

    Practically:
      - `--config` values are applied *only if* the user didn't override the same arg on CLI.
      - `--overrides` is handled by the outer CLI (it becomes `self.config`).
    """

    NAME: str = "recipe"
    DESCRIPTION: str = ""

    def __init__(self, *, out_dir: Path, config: Optional[Dict[str, Any]] = None):
        self.out_dir = out_dir
        self.config = config or {}

    @classmethod
    def add_common_args(cls, p: argparse.ArgumentParser) -> None:
        p.add_argument("--config", type=str, default=None, help="YAML/JSON config file (applies as defaults).")
        p.add_argument("--seed", type=int, default=0, help="Global random seed.")
        p.add_argument("--cache_dir", type=str, default=None, help="Latent cache directory (optional).")
        p.add_argument("--no_cache", action="store_true", help="Disable latent caching even if cache_dir is set.")
        p.add_argument("--log", type=str, default="none", help="Logging: none|tensorboard|wandb")
        p.add_argument("--wandb_project", type=str, default="mezzanine", help="wandb project name")
        p.add_argument("--wandb_name", type=str, default=None, help="wandb run name")
        p.add_argument("--wandb_entity", type=str, default=None, help="wandb entity/org")

    @staticmethod
    def apply_config_defaults(parser: argparse.ArgumentParser, args: argparse.Namespace, cfg: Dict[str, Any]) -> None:
        """Override args with cfg values ONLY if args still equal argparse defaults."""
        for k, v in cfg.items():
            if not hasattr(args, k):
                continue
            default = parser.get_default(k)
            cur = getattr(args, k)
            if cur == default:
                setattr(args, k, v)

    def build_context(self, args: argparse.Namespace) -> RunContext:
        file_cfg = load_config(getattr(args, "config", None))
        cfg = deep_update(file_cfg, self.config)

        seed = int(getattr(args, "seed", 0))
        seed_everything(seed)

        out_dir = Path(getattr(args, "out_dir", None) or self.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        cache: Optional[LatentCache] = None
        cache_dir = getattr(args, "cache_dir", None)
        if cache_dir and (not getattr(args, "no_cache", False)):
            cache = LatentCache(LatentCacheConfig(cache_dir=Path(cache_dir), enabled=True, compress=True))

        logger = make_logger(
            getattr(args, "log", "none"),
            out_dir=out_dir,
            config=cfg,
            wandb_project=getattr(args, "wandb_project", "mezzanine"),
            wandb_name=getattr(args, "wandb_name", None),
            wandb_entity=getattr(args, "wandb_entity", None),
        )
        return RunContext(out_dir=out_dir, config=cfg, seed=seed, cache=cache, logger=logger)

    @abc.abstractmethod
    def run(self, argv: list[str]) -> Dict[str, Any]:
        raise NotImplementedError
