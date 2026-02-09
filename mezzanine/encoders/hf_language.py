from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Literal, Optional

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from ..core.cache import hash_dict
from .base import Encoder

TextPool = Literal["cls", "mean"]


@dataclass
class HFLanguageEncoderConfig:
    model_name: str = "bert-base-uncased"
    batch_size: int = 16
    fp16: bool = True
    max_length: int = 256
    layer: int = -1  # hidden_states index (negative allowed)
    pool: TextPool = "mean"


class HFLanguageEncoder(Encoder):
    """Frozen HF language model -> sentence embeddings (for warrant-gap measurement/distillation).

    This is NOT a generative interface; it is a representation interface.
    """

    NAME = "hf_language"
    DESCRIPTION = "HuggingFace AutoModel text encoder with simple pooling."

    def __init__(self, cfg: HFLanguageEncoderConfig, device: str = "cuda"):
        self.cfg = cfg
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
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

    def encode(self, texts: List[str]) -> np.ndarray:
        out: List[np.ndarray] = []
        bs = int(self.cfg.batch_size)

        i = 0
        while i < len(texts):
            chunk = texts[i:i+bs]
            inp = self.tok(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=int(self.cfg.max_length),
            )
            inp = {k: v.to(self.device) for k, v in inp.items()}
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=(self.cfg.fp16 and self.device.startswith("cuda"))):
                    o = self.model(**inp, output_hidden_states=True)
                    h = o.hidden_states[self.cfg.layer]  # [B,T,D]
                    if self.cfg.pool == "cls":
                        z = h[:, 0]
                    elif self.cfg.pool == "mean":
                        attn = inp.get("attention_mask", None)
                        if attn is None:
                            z = h.mean(dim=1)
                        else:
                            mask = attn.unsqueeze(-1).to(h.dtype)  # [B,T,1]
                            z = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
                    else:
                        raise ValueError(f"Unknown pool={self.cfg.pool}")
                    z = torch.nn.functional.normalize(z, dim=-1)
            out.append(z.detach().cpu().float().numpy())
            i += bs

        Z = np.concatenate(out, axis=0).astype(np.float32)
        return self._l2norm(Z)


# Register
from ..registry import ENCODERS
ENCODERS.register('hf_language')( HFLanguageEncoder )
