from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Literal

import numpy as np

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from ..core.cache import hash_dict
from .base import Encoder


EmbedMode = Literal["cls", "mean", "mean_std"]


@dataclass
class HFVisionEncoderConfig:
    model_name: str = "facebook/ijepa_vith14_1k"
    batch_size: int = 32
    fp16: bool = True
    embed_mode: EmbedMode = "mean_std"
    embed_layer: int = -4  # negative=earlier layers


class HFVisionEncoder(Encoder):
    """Frozen HF vision backbone -> vector embeddings suitable for distillation/evaluation.

    Notes:
    - For domains like 2D physics sprites, later layers can be overly invariant (collapse).
      Using patch tokens + earlier layers + mean/std pooling helps.
    """

    NAME = "hf_vision"
    DESCRIPTION = "HuggingFace AutoModel vision backbone (I-JEPA/DINO/ViT/...) with configurable pooling."

    def __init__(self, cfg: HFVisionEncoderConfig, device: str = "cuda"):
        self.cfg = cfg
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(cfg.model_name)
        self.model = AutoModel.from_pretrained(cfg.model_name).to(device)
        self.model.eval()

    def fingerprint(self) -> str:
        d = asdict(self.cfg)
        d["__class__"] = self.__class__.__name__
        return hash_dict(d)

    @staticmethod
    def _l2norm(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
        n = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.clip(n, eps, None)

    def _pool(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B,T,D]
        if self.cfg.embed_mode == "cls":
            z = h[:, 0]
        else:
            patch = h[:, 1:] if h.shape[1] > 1 else h
            if self.cfg.embed_mode == "mean":
                z = patch.mean(dim=1)
            elif self.cfg.embed_mode == "mean_std":
                z = torch.cat([patch.mean(dim=1), patch.std(dim=1)], dim=-1)
            else:
                raise ValueError(f"Unknown embed_mode={self.cfg.embed_mode}")
        return torch.nn.functional.normalize(z, dim=-1)

    def encode(self, imgs: List[np.ndarray]) -> np.ndarray:
        out: List[np.ndarray] = []
        bs = int(self.cfg.batch_size)

        def forward(pil_imgs: List[Image.Image]) -> torch.Tensor:
            inp = self.processor(images=pil_imgs, return_tensors="pt")
            inp = {k: v.to(self.device) for k, v in inp.items()}
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=(self.cfg.fp16 and self.device.startswith("cuda"))):
                    if self.cfg.embed_layer != 0:
                        out = self.model(**inp, output_hidden_states=True)
                        h = out.hidden_states[self.cfg.embed_layer]
                    else:
                        out = self.model(**inp)
                        h = out.last_hidden_state
                    z = self._pool(h)
            return z

        i = 0
        while i < len(imgs):
            chunk = imgs[i:i + bs]
            pil = [Image.fromarray(x) for x in chunk]
            try:
                z = forward(pil)
                out.append(z.detach().cpu().float().numpy())
                i += bs
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and self.device.startswith("cuda") and bs > 1:
                    bs = max(1, bs // 2)
                    torch.cuda.empty_cache()
                    continue
                raise

        Z = np.concatenate(out, axis=0).astype(np.float32)
        return self._l2norm(Z)


# Register
from ..registry import ENCODERS
ENCODERS.register('hf_vision')( HFVisionEncoder )
