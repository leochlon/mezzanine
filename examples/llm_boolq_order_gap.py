"""Measure prompt-level order sensitivity for a causal LM on BoolQ.

This is NOT meant to be a leaderboard script. It's a *warrant-gap probe*.

Idea:
  - Keep the question fixed.
  - Treat the passage sentence order as a symmetry.
  - Score answers (yes/no) by log-prob under the LM.
  - Measure belief instability across permutations.
"""

from __future__ import annotations

import argparse
import re
from typing import List

import numpy as np

try:
    from datasets import load_dataset
except Exception as e:  # pragma: no cover
    raise SystemExit("Please `pip install mezzanine[datasets]` (or `pip install datasets`).") from e

from mezzanine.predictors.hf_causal_lm import HFCausalLMChoicePredictor, HFCausalLMChoicePredictorConfig
from mezzanine.pipelines.text_distill import warrant_gap_from_views

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> List[str]:
    parts = _SENT_SPLIT.split((text or "").strip())
    parts = [p.strip() for p in parts if p.strip()]
    return parts if parts else [text]


def join_sentences(parts: List[str]) -> str:
    return " ".join([p.strip() for p in parts if p.strip()])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default="gpt2")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--k", type=int, default=8, help="views per example (includes canonical)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ds = load_dataset("boolq", split="validation")
    ds = ds.shuffle(seed=args.seed).select(range(args.n))

    prompts = []
    passages = []
    for ex in ds:
        passage = ex["passage"]
        question = ex["question"]
        passages.append(passage)
        prompts.append(f"Passage: {passage}\nQuestion: {question}\nAnswer:")

    predictor = HFCausalLMChoicePredictor(HFCausalLMChoicePredictorConfig(model_name=args.model_name))

    # Build views by shuffling passage sentences
    rng = np.random.default_rng(args.seed)
    all_views = []
    for j in range(args.k):
        if j == 0:
            all_views.append(prompts)
            continue
        v = []
        for i, ex in enumerate(ds):
            sents = split_sentences(ex["passage"])
            if len(sents) <= 1:
                p2 = ex["passage"]
            else:
                perm = list(sents)
                rng2 = np.random.default_rng(int((args.seed + 1000*i + 97*j) % (2**32-1)))
                rng2.shuffle(perm)
                p2 = join_sentences(perm)
            v.append(f"Passage: {p2}\nQuestion: {ex['question']}\nAnswer:")
        all_views.append(v)

    # Score yes/no for each view
    choices = [" yes", " no"]
    P_views = []
    for j in range(args.k):
        Pj = predictor.predict_proba(all_views[j], choices=choices)
        P_views.append(Pj)
    P_views = np.stack(P_views, axis=1)  # [N,K,2]

    gap = warrant_gap_from_views(P_views)
    print("warrant_gap:", gap)
    # Optional: show a few examples where belief flips
    flips = np.argmax(P_views[:, 0, :], axis=-1) != np.argmax(P_views[:, 1, :], axis=-1)
    print(f"flip_rate(view0 vs view1) = {flips.mean():.3f}")

if __name__ == "__main__":
    main()
