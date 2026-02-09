from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from ..core.cache import hash_dict
from .base import Encoder


@dataclass
class HFCausalLMEncoderConfig:
    """Extract hidden-state embeddings from a frozen causal LM.

    This is the "stronger LLM" path: we distill symmetry-marginalized beliefs
    into *representations* (hidden states) rather than treating the LLM as a black-box scorer.

    Notes:
      - The LM is used in forward mode with `output_hidden_states=True`.
      - `layer` may be negative (e.g. -1 last, -4 fourth-from-last).
      - `embed_mode` controls pooling over tokens:
          * "last": last non-pad token hidden state (recommended for prompt->answer classifiers)
          * "mean": mean over non-pad tokens
          * "mean_std": concat(mean, std) over non-pad tokens
    """
    model_name: str = "gpt2"
    batch_size: int = 8
    fp16: bool = True
    max_length: int = 512
    device: str = "cuda"

    layer: int = -1
    embed_mode: str = "last"  # last|mean|mean_std
    padding_side: str = "left"
    pad_token: str | None = None
    # Progress bars for long runs
    progress: bool = False


class HFCausalLMEncoder(Encoder):
    NAME = "hf_causal_lm"
    DESCRIPTION = "HF causal LM hidden-state encoder (frozen)."

    def __init__(
        self,
        cfg: HFCausalLMEncoderConfig,
        *,
        model: PreTrainedModel | None = None,
        tok: PreTrainedTokenizerBase | None = None,
    ):
        self.cfg = cfg
        self.tok = tok or AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
        if cfg.pad_token is not None:
            self.tok.pad_token = cfg.pad_token
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = cfg.padding_side

        if model is None:
            self.model = AutoModelForCausalLM.from_pretrained(cfg.model_name)
            self.model.to(cfg.device)
        else:
            self.model = model
        self.model.eval()

    def fingerprint(self) -> str:
        d = asdict(self.cfg)
        # remove device from fingerprint; it's not semantic
        d.pop("device", None)
        return hash_dict(d)

    @torch.no_grad()
    def encode(self, inputs: Sequence[Any]) -> np.ndarray:
        texts = [str(x) for x in inputs]
        device = self.cfg.device
        bs = int(self.cfg.batch_size)
        out_chunks: List[np.ndarray] = []

        from tqdm import tqdm
        it = range(0, len(texts), bs)
        for i in tqdm(it, desc="encode batches", disable=not self.cfg.progress):
            batch = texts[i : i + bs]
            enc = self.tok(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=int(self.cfg.max_length),
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            input_ids = enc["input_ids"]
            attn = enc["attention_mask"]  # [B,T]

            with torch.amp.autocast("cuda", enabled=(self.cfg.fp16 and device.startswith("cuda"))):
                out = self.model(input_ids=input_ids, attention_mask=attn, output_hidden_states=True, use_cache=False)

            hs = out.hidden_states  # tuple(L+1) of [B,T,H]
            layer = int(self.cfg.layer)
            if layer < 0:
                layer = len(hs) + layer
            if layer < 0 or layer >= len(hs):
                raise ValueError(f"layer={self.cfg.layer} out of range for len(hidden_states)={len(hs)}")

            H = hs[layer]  # [B,T,H]
            # build pooled embedding
            if self.cfg.embed_mode == "last":
                lengths = attn.sum(dim=1)  # [B]
                idx = torch.clamp(lengths - 1, min=0).long()  # [B]
                pooled = H[torch.arange(H.shape[0], device=device), idx, :]  # [B,H]
            else:
                mask = attn.unsqueeze(-1).to(H.dtype)  # [B,T,1]
                denom = torch.clamp(mask.sum(dim=1), min=1.0)  # [B,1]
                mean = (H * mask).sum(dim=1) / denom  # [B,H]
                if self.cfg.embed_mode == "mean":
                    pooled = mean
                elif self.cfg.embed_mode == "mean_std":
                    var = ((H - mean.unsqueeze(1)) * mask).pow(2).sum(dim=1) / denom
                    std = torch.sqrt(torch.clamp(var, min=1e-9))
                    pooled = torch.cat([mean, std], dim=1)
                else:
                    raise ValueError(f"Unknown embed_mode={self.cfg.embed_mode}")

            out_chunks.append(pooled.detach().cpu().float().numpy())

        return np.concatenate(out_chunks, axis=0)


# Register
from ..registry import ENCODERS
ENCODERS.register("hf_causal_lm")(HFCausalLMEncoder)
