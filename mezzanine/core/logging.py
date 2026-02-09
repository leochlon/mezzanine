from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


class BaseLogger:
    """Optional experiment logger (no-op by default).

    The goal is to make W&B / TensorBoard integration easy, without requiring them.
    """

    def log_metrics(self, metrics: Dict[str, float], *, step: Optional[int] = None, prefix: str = "") -> None:
        return

    def log_text(self, key: str, text: str, *, step: Optional[int] = None) -> None:
        return

    def log_artifact(self, path: str | Path, *, name: Optional[str] = None) -> None:
        return

    def close(self) -> None:
        return


class NullLogger(BaseLogger):
    pass


@dataclass
class TensorBoardLoggerConfig:
    log_dir: Path


class TensorBoardLogger(BaseLogger):
    def __init__(self, cfg: TensorBoardLoggerConfig):
        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore
        except Exception as e:
            raise RuntimeError("TensorBoard requested but not installed. Install: pip install mezzanine[tensorboard]") from e
        self.writer = SummaryWriter(log_dir=str(cfg.log_dir))

    def log_metrics(self, metrics: Dict[str, float], *, step: Optional[int] = None, prefix: str = "") -> None:
        s = 0 if step is None else int(step)
        for k, v in metrics.items():
            self.writer.add_scalar(prefix + k, float(v), s)

    def log_text(self, key: str, text: str, *, step: Optional[int] = None) -> None:
        s = 0 if step is None else int(step)
        self.writer.add_text(key, text, s)

    def log_artifact(self, path: str | Path, *, name: Optional[str] = None) -> None:
        # TensorBoard doesn't have a universal artifact API; we log as text.
        p = Path(path)
        self.writer.add_text(name or p.name, str(p), 0)

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()


@dataclass
class WandbLoggerConfig:
    project: str
    name: Optional[str] = None
    entity: Optional[str] = None
    dir: Optional[Path] = None
    config: Optional[Dict[str, Any]] = None


class WandbLogger(BaseLogger):
    def __init__(self, cfg: WandbLoggerConfig):
        try:
            import wandb  # type: ignore
        except Exception as e:
            raise RuntimeError("wandb requested but not installed. Install: pip install mezzanine[wandb]") from e

        self.wandb = wandb
        init_kwargs: Dict[str, Any] = {"project": cfg.project}
        if cfg.name:
            init_kwargs["name"] = cfg.name
        if cfg.entity:
            init_kwargs["entity"] = cfg.entity
        if cfg.dir:
            init_kwargs["dir"] = str(cfg.dir)
        if cfg.config:
            init_kwargs["config"] = cfg.config
        self.run = wandb.init(**init_kwargs)

    def log_metrics(self, metrics: Dict[str, float], *, step: Optional[int] = None, prefix: str = "") -> None:
        data = {prefix + k: float(v) for k, v in metrics.items()}
        if step is not None:
            data["step"] = int(step)
        self.wandb.log(data)

    def log_text(self, key: str, text: str, *, step: Optional[int] = None) -> None:
        self.wandb.log({key: text, "step": int(step) if step is not None else 0})

    def log_artifact(self, path: str | Path, *, name: Optional[str] = None) -> None:
        p = Path(path)
        art = self.wandb.Artifact(name or p.stem, type="file")
        art.add_file(str(p))
        self.run.log_artifact(art)

    def close(self) -> None:
        self.run.finish()


def make_logger(
    kind: str,
    *,
    out_dir: Path,
    config: Optional[Dict[str, Any]] = None,
    wandb_project: str = "mezzanine",
    wandb_name: Optional[str] = None,
    wandb_entity: Optional[str] = None,
) -> BaseLogger:
    kind = (kind or "none").lower()
    if kind in ["none", "null", "off"]:
        return NullLogger()
    if kind in ["tb", "tensorboard"]:
        return TensorBoardLogger(TensorBoardLoggerConfig(log_dir=out_dir / "tb"))
    if kind in ["wandb", "wb"]:
        return WandbLogger(
            WandbLoggerConfig(
                project=wandb_project,
                name=wandb_name,
                entity=wandb_entity,
                dir=out_dir,
                config=config or {},
            )
        )
    raise ValueError(f"Unknown logger kind: {kind}")
