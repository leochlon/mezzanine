"""Minimal API-only usage: measure embedding stability under an order symmetry.

This example does not train anything; it shows how to call `mezzanine.measure(...)`.
"""

from __future__ import annotations

import re
from typing import List

from mezzanine.api import measure
from mezzanine.encoders.hf_language import HFLanguageEncoder, HFLanguageEncoderConfig
from mezzanine.symmetries.order import OrderSymmetry, OrderSymmetryConfig

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> List[str]:
    parts = _SENT_SPLIT.split((text or "").strip())
    parts = [p.strip() for p in parts if p.strip()]
    return parts if parts else [text]


def join_sentences(parts: List[str]) -> str:
    return " ".join([p.strip() for p in parts if p.strip()])


def make_views(texts, seed: int, k: int):
    symmetry = OrderSymmetry(OrderSymmetryConfig())
    views = [list(texts)]
    for j in range(1, k):
        vj = []
        for i, t in enumerate(texts):
            parts = split_sentences(t)
            if len(parts) <= 1:
                vj.append(t)
            else:
                perm = symmetry.sample(parts, seed=(seed + 1000*i + 97*j))
                vj.append(join_sentences(list(perm)))
        views.append(vj)
    return views


def main():
    texts = [
        "John went to the store. He bought milk. Then he went home.",
        "The cat sat on the mat. It purred loudly. The dog barked.",
    ]
    encoder = HFLanguageEncoder(HFLanguageEncoderConfig(model_name="distilbert-base-uncased", max_length=128, batch_size=8))
    res = measure(inputs=texts, make_views=make_views, encoder=encoder, k=8, seed=0)
    print(res)

if __name__ == "__main__":
    main()
