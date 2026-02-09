from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from ..core.cache import hash_dict
from ..core.deterministic import deterministic_subsample_indices
from ..encoders.hf_language import HFLanguageEncoder, HFLanguageEncoderConfig
from ..symmetries.order import OrderSymmetry, OrderSymmetryConfig
from ..pipelines.text_distill import (
    MLPHeadConfig,
    train_hard_label_head,
    train_soft_label_head,
    predict_proba,
    accuracy,
    warrant_gap_from_views,
)
from ..viz.text_distill import plot_text_order_distill
from ..worlds.hf_dataset import HFDatasetAdapter, HFDatasetAdapterConfig
from .recipe_base import Recipe


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return [""]
    parts = _SENT_SPLIT.split(text)
    parts = [p.strip() for p in parts if p.strip()]
    return parts if parts else [text]


def join_sentences(parts: List[str]) -> str:
    return " ".join([p.strip() for p in parts if p.strip()])


def view_seed(global_seed: int, i: int, j: int) -> int:
    # Deterministic per-example per-view seed, stable across runs.
    return int((global_seed * 1000003 + i * 9176 + j * 7919) % (2**32 - 1))


def _cached_encode(
    *,
    cache,
    world_fp: str,
    enc_fp: str,
    split: str,
    tag: str,
    extra: Dict[str, Any],
    encoder: HFLanguageEncoder,
    texts: List[str],
) -> np.ndarray:
    if cache is None:
        return encoder.encode(texts)

    key = cache.make_key(world_fingerprint=world_fp, encoder_fingerprint=enc_fp, split=split, tag=tag, extra=extra)
    got = cache.get(key)
    if got is not None:
        arr, _meta = got
        return arr
    Z = encoder.encode(texts)
    cache.put(key, Z, meta={"n": len(texts), "tag": tag, "extra": extra})
    return Z


def build_views(
    texts: List[str],
    *,
    symmetry: OrderSymmetry,
    seed: int,
    K: int,
    min_sentences: int = 2,
) -> List[List[str]]:
    """Return list of K view-lists, each view-list is texts for all examples."""
    N = len(texts)
    # View 0: canonical
    views: List[List[str]] = [list(texts)]
    if K <= 1:
        return views
    for j in range(1, K):
        vj: List[str] = []
        for i, t in enumerate(texts):
            parts = split_sentences(t)
            if len(parts) < min_sentences:
                vj.append(t)
                continue
            perm = symmetry.sample(parts, seed=view_seed(seed, i, j))
            vj.append(join_sentences(list(perm)))
        views.append(vj)
    return views


class HFTextOrderDistillRecipe(Recipe):
    NAME = "hf_text_order_distill"
    DESCRIPTION = "HF text: measure sentence-order warrant gap, then distill symmetry-marginalized predictions into a single-pass head."

    def run(self, argv: list[str]) -> Dict[str, Any]:
        p = argparse.ArgumentParser(prog=self.NAME)
        self.add_common_args(p)

        # Dataset
        p.add_argument("--dataset", type=str, default="ag_news")
        p.add_argument("--subset", type=str, default=None)
        p.add_argument("--text_field", type=str, default="text")
        p.add_argument("--label_field", type=str, default="label")
        p.add_argument("--train_split", type=str, default="train")
        p.add_argument("--test_split", type=str, default="test")
        p.add_argument("--n_train", type=int, default=5000)
        p.add_argument("--n_test", type=int, default=2000)

        # Symmetry
        p.add_argument("--k_train", type=int, default=8, help="Number of sentence-order views for teacher expectation (includes canonical).")
        p.add_argument("--k_test", type=int, default=16, help="Number of sentence-order views for evaluation (includes canonical).")
        p.add_argument("--min_sentences", type=int, default=2)

        # Encoder
        p.add_argument("--model_name", type=str, default="distilbert-base-uncased")
        p.add_argument("--pool", type=str, default="cls", choices=["cls", "mean"])
        p.add_argument("--max_length", type=int, default=256)
        p.add_argument("--encode_bs", type=int, default=32)
        p.add_argument("--no_fp16", action="store_true")

        # Head
        p.add_argument("--hidden", type=int, default=512)
        p.add_argument("--depth", type=int, default=1)
        p.add_argument("--dropout", type=float, default=0.0)
        p.add_argument("--base_steps", type=int, default=800)
        p.add_argument("--student_steps", type=int, default=800)
        p.add_argument("--batch_size", type=int, default=256)
        p.add_argument("--lr", type=float, default=1e-3)
        p.add_argument("--hard_label_weight", type=float, default=0.0, help="Optional mix-in of hard labels during distillation (0..1).")

        args = p.parse_args(argv)
        # Apply config file defaults (only where user didn't override)
        file_cfg = {}  # handled by Recipe.build_context
        self.apply_config_defaults(p, args, file_cfg)

        ctx = self.build_context(args)
        device = "cuda" if (hasattr(__import__("torch"), "cuda") and __import__("torch").cuda.is_available()) else "cpu"  # type: ignore[attr-defined]

        # Build adapter
        adapter_cfg = HFDatasetAdapterConfig(
            dataset=args.dataset,
            subset=args.subset,
            text_field=args.text_field,
            label_field=args.label_field,
            train_split=args.train_split,
            test_split=args.test_split,
            n_train=int(args.n_train),
            n_test=int(args.n_test),
            seed=int(args.seed),
        )
        adapter = HFDatasetAdapter(adapter_cfg)
        world = adapter.load()
        world_fp = adapter.fingerprint()

        train = world["train"]
        test = world["test"]
        meta = world.get("meta", {})

        x_train = [ex[args.text_field] for ex in train]
        y_train = np.array([int(ex[args.label_field]) for ex in train], dtype=np.int64)
        x_test = [ex[args.text_field] for ex in test]
        y_test = np.array([int(ex[args.label_field]) for ex in test], dtype=np.int64)

        num_classes = int(max(y_train.max(), y_test.max()) + 1)

        # Encoder
        enc_cfg = HFLanguageEncoderConfig(
            model_name=args.model_name,
            max_length=int(args.max_length),
            batch_size=int(args.encode_bs),
            fp16=not bool(args.no_fp16),
            pool=args.pool,
            device=device,
        )
        encoder = HFLanguageEncoder(enc_cfg)
        enc_fp = encoder.fingerprint()

        # Symmetry (order)
        symmetry = OrderSymmetry(OrderSymmetryConfig())

        # Build canonical embeddings
        Z_train = _cached_encode(
            cache=ctx.cache,
            world_fp=world_fp,
            enc_fp=enc_fp,
            split="train",
            tag="canon",
            extra={"k": 0},
            encoder=encoder,
            texts=x_train,
        )
        Z_test = _cached_encode(
            cache=ctx.cache,
            world_fp=world_fp,
            enc_fp=enc_fp,
            split="test",
            tag="canon",
            extra={"k": 0},
            encoder=encoder,
            texts=x_test,
        )

        # Deterministic train/val split
        n = Z_train.shape[0]
        idx = np.arange(n)
        # 80/20 split
        n_val = max(1, int(0.2 * n))
        idx_val = idx[:n_val]
        idx_tr = idx[n_val:]

        Z_tr, y_tr = Z_train[idx_tr], y_train[idx_tr]
        Z_val, y_val = Z_train[idx_val], y_train[idx_val]

        head_cfg = MLPHeadConfig(in_dim=int(Z_train.shape[1]), num_classes=num_classes, hidden=int(args.hidden), depth=int(args.depth), dropout=float(args.dropout))

        # Train baseline head (hard labels)
        base_head, base_metrics = train_hard_label_head(
            Z_tr, y_tr, Z_val, y_val,
            cfg=head_cfg,
            steps=int(args.base_steps),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            device=device,
            seed=int(args.seed),
        )

        # Baseline evaluation (canonical)
        P_base_test_canon = predict_proba(base_head, Z_test, device=device)
        base_acc = accuracy(P_base_test_canon, y_test)

        # Build test views and measure baseline warrant gap
        test_views = build_views(x_test, symmetry=symmetry, seed=int(args.seed) + 123, K=int(args.k_test), min_sentences=int(args.min_sentences))
        # Encode each view
        Z_test_views: List[np.ndarray] = [Z_test]  # view0 = canon
        for j in range(1, int(args.k_test)):
            Zj = _cached_encode(
                cache=ctx.cache,
                world_fp=world_fp,
                enc_fp=enc_fp,
                split="test",
                tag=f"order_view_{j}",
                extra={"k": int(args.k_test), "j": j, "min_sentences": int(args.min_sentences)},
                encoder=encoder,
                texts=test_views[j],
            )
            Z_test_views.append(Zj)

        # Predict probabilities for each view
        P_views = np.stack([predict_proba(base_head, Zj, device=device) for Zj in Z_test_views], axis=1)  # [N,K,C]
        gap_base = warrant_gap_from_views(P_views)

        # Teacher: symmetry-marginalized probs on train via baseline head
        train_views = build_views(x_train, symmetry=symmetry, seed=int(args.seed) + 999, K=int(args.k_train), min_sentences=int(args.min_sentences))
        Z_train_views: List[np.ndarray] = [Z_train]  # view0
        for j in range(1, int(args.k_train)):
            Zj = _cached_encode(
                cache=ctx.cache,
                world_fp=world_fp,
                enc_fp=enc_fp,
                split="train",
                tag=f"order_view_{j}",
                extra={"k": int(args.k_train), "j": j, "min_sentences": int(args.min_sentences)},
                encoder=encoder,
                texts=train_views[j],
            )
            Z_train_views.append(Zj)
        P_train_views = np.stack([predict_proba(base_head, Zj, device=device) for Zj in Z_train_views], axis=1)
        P_teacher = P_train_views.mean(axis=1)  # [N,C]

        # Train student head on canonical embeddings to match teacher expectation
        stud_head, stud_metrics = train_soft_label_head(
            Z_tr,
            P_teacher[idx_tr],
            Z_val,
            y_val,
            cfg=head_cfg,
            steps=int(args.student_steps),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            device=device,
            seed=int(args.seed) + 1,
            hard_label_weight=float(args.hard_label_weight),
            y_train=y_tr,
        )

        # Student evaluation
        P_stud_test_canon = predict_proba(stud_head, Z_test, device=device)
        stud_acc = accuracy(P_stud_test_canon, y_test)

        # Student gap across the same views (evaluate student on view embeddings too)
        P_stud_views = np.stack([predict_proba(stud_head, Zj, device=device) for Zj in Z_test_views], axis=1)
        gap_stud = warrant_gap_from_views(P_stud_views)

        # Make / break
        base_gap = float(gap_base["mean_tv_to_mean"])
        stud_gap = float(gap_stud["mean_tv_to_mean"])
        tv_rel_improve = float((base_gap - stud_gap) / max(1e-9, base_gap))
        acc_drop = float(base_acc - stud_acc)
        verdict = "MAKE ✅" if (tv_rel_improve >= 0.2 and acc_drop <= 0.05) else "BREAK / INCONCLUSIVE ❌"

        summary = {
            "exp": self.NAME,
            "device": device,
            "data": {
                "dataset": args.dataset,
                "subset": args.subset,
                "n_train": int(args.n_train),
                "n_test": int(args.n_test),
                "text_field": args.text_field,
                "label_field": args.label_field,
                "meta": meta,
            },
            "encoder": asdict(enc_cfg),
            "symmetry": {"name": "order", "k_train": int(args.k_train), "k_test": int(args.k_test), "min_sentences": int(args.min_sentences)},
            "baseline": {
                "acc": base_acc,
                "gap_mean_tv_to_mean": base_gap,
                "gap_mean_pairwise_tv": float(gap_base["mean_pairwise_tv"]),
            },
            "student": {
                "acc": stud_acc,
                "gap_mean_tv_to_mean": stud_gap,
                "gap_mean_pairwise_tv": float(gap_stud["mean_pairwise_tv"]),
            },
            "make_break": {
                "tv_rel_improve": tv_rel_improve,
                "acc_drop": acc_drop,
                "criterion": "tv_rel_improve>=0.2 and acc_drop<=0.05",
                "verdict": verdict,
            },
        }

        # Save artifacts
        (ctx.out_dir / "results.json").write_text(json.dumps(summary, indent=2))
        plot_text_order_distill(summary, ctx.out_dir / "diagnostics.png")

        ctx.logger.log_metrics({
            "baseline/acc": base_acc,
            "baseline/gap_tv": base_gap,
            "student/acc": stud_acc,
            "student/gap_tv": stud_gap,
            "make_break/tv_rel_improve": tv_rel_improve,
            "make_break/acc_drop": acc_drop,
        }, step=0)

        return summary
