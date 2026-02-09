from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from ..core.cache import hash_dict
from .base import Predictor


@dataclass
class HFCausalLMChoicePredictorConfig:
    """Score discrete choices by causal LM log-prob.

    This is a lightweight, *model-agnostic* way to turn a generative LM
    into a classifier for warrant-gap measurement.
    """
    model_name: str = "gpt2"
    batch_size: int = 4
    fp16: bool = True
    max_length: int = 512
    device: str = "cuda"
    # Tokenizer / padding knobs
    padding_side: str = "left"
    # If None, will try to infer (or fall back to eos_token as pad_token)
    pad_token: str | None = None
    # Progress bars for long runs
    progress: bool = False


class HFCausalLMChoicePredictor(Predictor):
    """Causal LM choice scoring predictor.

    Usage:
      predictor.predict_proba(prompts, choices=[" yes", " no"])
    """

    def __init__(
        self,
        cfg: HFCausalLMChoicePredictorConfig,
        *,
        model: PreTrainedModel | None = None,
        tok: PreTrainedTokenizerBase | None = None,
    ):
        self.cfg = cfg
        self.tok = tok or AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
        if cfg.pad_token is not None:
            self.tok.pad_token = cfg.pad_token
        if self.tok.pad_token is None:
            # common safe default
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = cfg.padding_side
        if model is None:
            self.model = AutoModelForCausalLM.from_pretrained(cfg.model_name).to(cfg.device)
        else:
            # Assume caller already placed the model on the desired device / device_map.
            self.model = model
        self.model.eval()

    def fingerprint(self) -> str:
        d = asdict(self.cfg)
        d["__class__"] = self.__class__.__name__
        return hash_dict(d)

    @torch.no_grad()
    def _score_batch(self, prompts: List[str], choices: List[str]) -> np.ndarray:
        """Return log-prob scores [B, C] for appending each choice."""
        device = self.cfg.device
        # Tokenize prompts once
        enc = self.tok(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(self.cfg.max_length),
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        input_ids = enc["input_ids"]
        attn = enc["attention_mask"]
        B = input_ids.shape[0]

        # For each choice, append tokens (without BOS) and compute conditional logprobs
        scores = np.zeros((B, len(choices)), dtype=np.float32)

        # Precompute choice token ids
        choice_ids: List[torch.Tensor] = []
        for c in choices:
            ids = self.tok(c, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
            choice_ids.append(ids.to(device))

        # We do *C* forward passes; for small C this is fine and very transparent.
        for j, ids in enumerate(choice_ids):
            # Build extended sequences
            ids_rep = ids.unsqueeze(0).expand(B, -1)  # [B, Lc]
            ext = torch.cat([input_ids, ids_rep], dim=1)
            ext_attn = torch.cat([attn, torch.ones((B, ids_rep.shape[1]), device=device, dtype=attn.dtype)], dim=1)

            with torch.amp.autocast("cuda", enabled=(self.cfg.fp16 and device.startswith("cuda"))):
                out = self.model(ext, attention_mask=ext_attn)
                logits = out.logits  # [B, T, V]

            # We want log p(choice tokens | prompt)
            # logits at position t predict token at t+1, so we align carefully.
            # Choice tokens occupy positions [T0, T0+Lc-1] in ext
            T0 = input_ids.shape[1]
            # For token k in choice (0-index), its logprob comes from logits at position (T0 + k - 1)
            # predicting ext[:, T0+k]
            logp = 0.0
            for k in range(ids.shape[0]):
                pos = T0 + k - 1
                # If k==0, pos=T0-1 uses last prompt token prediction
                l = logits[:, pos, :]
                tok = ext[:, T0 + k]
                lp = torch.log_softmax(l, dim=-1).gather(1, tok.unsqueeze(1)).squeeze(1)
                logp = logp + lp

            scores[:, j] = logp.detach().cpu().float().numpy()

        return scores

    def predict_proba(self, inputs: Sequence[Any], **kwargs: Any) -> np.ndarray:
        prompts = list(inputs)
        choices = kwargs.get("choices", None)
        if choices is None:
            raise ValueError("HFCausalLMChoicePredictor.predict_proba requires choices=[...]")
        choices = list(choices)
        bs = int(self.cfg.batch_size)
        all_scores: List[np.ndarray] = []
        from tqdm import tqdm
        it = range(0, len(prompts), bs)
        for i in tqdm(it, desc="teacher batches", disable=not self.cfg.progress):
            chunk = prompts[i : i + bs]
            s = self._score_batch(chunk, choices)
            all_scores.append(s)
        scores = np.concatenate(all_scores, axis=0)  # logprobs
        # softmax for probabilities
        m = scores.max(axis=1, keepdims=True)
        p = np.exp(scores - m)
        p = p / np.clip(p.sum(axis=1, keepdims=True), 1e-9, None)
        return p.astype(np.float32)
