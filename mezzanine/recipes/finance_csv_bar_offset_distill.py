from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from typing import Any, Dict, List

import numpy as np
import torch

from ..core.config import deep_update, load_config
from ..pipelines.text_distill import (
    MLPHeadConfig,
    accuracy,
    predict_proba,
    train_hard_label_head,
    train_soft_label_head,
    warrant_gap_from_views,
)
from ..symmetries.market_bar_offset import (
    MarketBarOffsetConfig,
    MarketBarOffsetSymmetry,
)
from ..viz.finance_distill import plot_finance_bar_offset_distill
from ..worlds.finance_csv import FinanceCSVTapeAdapter, FinanceCSVTapeAdapterConfig
from .recipe_base import Recipe


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return "mps"
    return "cpu"


def view_seed(global_seed: int, i: int, j: int) -> int:
    # Deterministic per-example per-view seed, stable across runs.
    return int((global_seed * 1000003 + i * 9176 + j * 7919) % (2**32 - 1))


def build_feature_views(
    examples: List[Dict[str, Any]],
    *,
    symmetry: MarketBarOffsetSymmetry,
    seed: int,
    K: int,
) -> List[np.ndarray]:
    """Return list of K feature matrices, each [N, D]. View 0 is canonical."""
    xs0 = np.stack([np.asarray(ex["x"], dtype=np.float32) for ex in examples], axis=0)
    views: List[np.ndarray] = [xs0]
    if K <= 1:
        return views

    for j in range(1, K):
        xsj = []
        for i, ex in enumerate(examples):
            exj = symmetry.sample(ex, seed=view_seed(seed, i, j))
            xsj.append(np.asarray(exj["x"], dtype=np.float32))
        views.append(np.stack(xsj, axis=0))
    return views


class FinanceCSVBarOffsetDistillRecipe(Recipe):
    NAME = "finance_csv_bar_offset_distill"
    DESCRIPTION = "Finance (CSV): measure bar-offset warrant gap, then distill offset-marginalized predictions into a single-pass tabular head."

    def run(self, argv: list[str]) -> Dict[str, Any]:
        p = argparse.ArgumentParser(prog=self.NAME)
        self.add_common_args(p)

        # CSV / adapter
        p.add_argument(
            "--path",
            type=str,
            required=True,
            help="Path to a CSV with at least a close column.",
        )
        p.add_argument("--close_col", type=str, default="close")
        p.add_argument("--timestamp_col", type=str, default="timestamp")
        p.add_argument(
            "--no_timestamp_col",
            action="store_true",
            help="If set, ignore timestamp_col even if present.",
        )
        p.add_argument("--symbol_col", type=str, default=None)
        p.add_argument("--symbol", type=str, default=None)
        p.add_argument("--delimiter", type=str, default=",")
        p.add_argument("--no_header", action="store_true")
        p.add_argument("--sort_by_timestamp", action="store_true")

        p.add_argument("--lookback", type=int, default=32)
        p.add_argument("--max_offset", type=int, default=1)
        p.add_argument("--trend_lookback", type=int, default=128)
        p.add_argument("--label_horizon", type=int, default=1)
        p.add_argument("--stride", type=int, default=1)
        p.add_argument("--gap", type=int, default=0)
        p.add_argument("--return_type", type=str, default="log", choices=["log", "pct"])

        p.add_argument("--n_train", type=int, default=5000)
        p.add_argument("--n_test", type=int, default=2000)

        # Symmetry
        p.add_argument(
            "--k_train",
            type=int,
            default=8,
            help="Number of bar-offset views for teacher expectation (includes canonical).",
        )
        p.add_argument(
            "--k_test",
            type=int,
            default=16,
            help="Number of bar-offset views for evaluation (includes canonical).",
        )
        p.add_argument(
            "--allow_positive",
            action="store_true",
            help="If set, bar offsets can be positive (may introduce lookahead).",
        )

        # Head / optimization
        p.add_argument("--hidden", type=int, default=256)
        p.add_argument("--depth", type=int, default=2)
        p.add_argument("--dropout", type=float, default=0.0)
        p.add_argument("--weight_decay", type=float, default=0.0)

        p.add_argument("--base_steps", type=int, default=800)
        p.add_argument("--student_steps", type=int, default=800)
        p.add_argument("--batch_size", type=int, default=256)
        p.add_argument("--lr", type=float, default=1e-3)
        p.add_argument("--hard_label_weight", type=float, default=0.0)

        # Validation + make/break
        p.add_argument(
            "--val_frac",
            type=float,
            default=0.2,
            help="Fraction of training set used as validation (taken from the *end* of train).",
        )
        p.add_argument("--min_tv_rel_improve", type=float, default=0.2)
        p.add_argument("--max_acc_drop", type=float, default=0.05)

        args = p.parse_args(argv)

        # Apply config file defaults BEFORE seeding/logger/cache creation
        file_cfg = load_config(getattr(args, "config", None))
        merged_cfg = deep_update(file_cfg, self.config)
        self.apply_config_defaults(p, args, merged_cfg)

        ctx = self.build_context(args)
        out_dir = ctx.out_dir
        device = _device()

        timestamp_col = (
            None
            if bool(args.no_timestamp_col)
            else str(args.timestamp_col or "").strip() or None
        )

        # --- Adapter ---
        adapter_cfg = FinanceCSVTapeAdapterConfig(
            path=str(args.path),
            close_col=str(args.close_col),
            timestamp_col=timestamp_col,
            symbol_col=str(args.symbol_col) if args.symbol_col else None,
            symbol=str(args.symbol) if args.symbol else None,
            delimiter=str(args.delimiter),
            has_header=not bool(args.no_header),
            sort_by_timestamp=bool(args.sort_by_timestamp),
            lookback=int(args.lookback),
            max_offset=int(args.max_offset),
            trend_lookback=int(args.trend_lookback),
            label_horizon=int(args.label_horizon),
            stride=int(args.stride),
            n_train=int(args.n_train),
            n_test=int(args.n_test),
            gap=int(args.gap),
            return_type=str(args.return_type),
            seed=int(args.seed),
        )
        adapter = FinanceCSVTapeAdapter(adapter_cfg)
        world = adapter.load()
        world_fp = adapter.fingerprint()

        train = world["train"]
        test = world["test"]
        meta = world.get("meta", {})

        Z_train = np.stack(
            [np.asarray(ex["x"], dtype=np.float32) for ex in train], axis=0
        )
        y_train = np.asarray([int(ex["label"]) for ex in train], dtype=np.int64)
        Z_test = np.stack(
            [np.asarray(ex["x"], dtype=np.float32) for ex in test], axis=0
        )
        y_test = np.asarray([int(ex["label"]) for ex in test], dtype=np.int64)

        # --- Split train/val (time-like: val is last val_frac) ---
        n = int(Z_train.shape[0])
        n_val = max(1, int(float(args.val_frac) * n))
        n_val = min(n - 1, n_val)
        idx_tr = np.arange(0, n - n_val, dtype=np.int64)
        idx_val = np.arange(n - n_val, n, dtype=np.int64)

        Z_tr, y_tr = Z_train[idx_tr], y_train[idx_tr]
        Z_val, y_val = Z_train[idx_val], y_train[idx_val]

        head_cfg = MLPHeadConfig(
            in_dim=int(Z_train.shape[1]),
            num_classes=2,
            hidden=int(args.hidden),
            depth=int(args.depth),
            dropout=float(args.dropout),
        )

        # --- Baseline head (hard labels) ---
        base_head, base_metrics = train_hard_label_head(
            Z_tr,
            y_tr,
            Z_val,
            y_val,
            cfg=head_cfg,
            steps=int(args.base_steps),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            device=device,
            seed=int(args.seed),
        )

        P_base_test_canon = predict_proba(base_head, Z_test, device=device)
        base_acc = accuracy(P_base_test_canon, y_test)

        # --- Symmetry views ---
        symmetry = MarketBarOffsetSymmetry(
            MarketBarOffsetConfig(
                max_offset=int(args.max_offset),
                allow_positive=bool(args.allow_positive),
            )
        )

        Z_test_views = build_feature_views(
            test, symmetry=symmetry, seed=int(args.seed) + 123, K=int(args.k_test)
        )
        P_base_views = np.stack(
            [predict_proba(base_head, Zj, device=device) for Zj in Z_test_views], axis=1
        )  # [N,K,C]
        gap_base = warrant_gap_from_views(P_base_views)

        # --- Teacher on train: average predictions across k_train views ---
        Z_train_views = build_feature_views(
            train, symmetry=symmetry, seed=int(args.seed) + 999, K=int(args.k_train)
        )
        P_train_views = np.stack(
            [predict_proba(base_head, Zj, device=device) for Zj in Z_train_views],
            axis=1,
        )
        P_teacher = P_train_views.mean(axis=1).astype(np.float32, copy=False)

        # --- Student head: distill onto canonical features ---
        stud_head, stud_metrics = train_soft_label_head(
            Z_tr,
            P_teacher[idx_tr],
            Z_val,
            y_val,
            cfg=head_cfg,
            steps=int(args.student_steps),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            device=device,
            seed=int(args.seed) + 1,
            hard_label_weight=float(args.hard_label_weight),
            y_train=y_tr,
        )

        P_stud_test_canon = predict_proba(stud_head, Z_test, device=device)
        stud_acc = accuracy(P_stud_test_canon, y_test)

        P_stud_views = np.stack(
            [predict_proba(stud_head, Zj, device=device) for Zj in Z_test_views], axis=1
        )
        gap_stud = warrant_gap_from_views(P_stud_views)

        # --- Make/break ---
        base_gap = float(gap_base["mean_tv_to_mean"])
        stud_gap = float(gap_stud["mean_tv_to_mean"])
        tv_rel_improve = float((base_gap - stud_gap) / max(1e-9, base_gap))
        acc_drop = float(base_acc - stud_acc)
        verdict = (
            "MAKE OK"
            if (
                tv_rel_improve >= float(args.min_tv_rel_improve)
                and acc_drop <= float(args.max_acc_drop)
            )
            else "BREAK / INCONCLUSIVE X"
        )

        summary: Dict[str, Any] = {
            "exp": self.NAME,
            "device": device,
            "world": {
                "adapter": "finance_csv",
                "fingerprint": world_fp,
                "config": asdict(adapter_cfg),
                "meta": meta,
            },
            "symmetry": {
                "name": "market_bar_offset",
                "k_train": int(args.k_train),
                "k_test": int(args.k_test),
                "max_offset": int(args.max_offset),
                "allow_positive": bool(args.allow_positive),
            },
            "baseline": {
                "acc": base_acc,
                "gap_mean_tv_to_mean": base_gap,
                "gap_mean_pairwise_tv": float(gap_base["mean_pairwise_tv"]),
                "val_acc": float(base_metrics.get("val_acc", float("nan"))),
            },
            "student": {
                "acc": stud_acc,
                "gap_mean_tv_to_mean": stud_gap,
                "gap_mean_pairwise_tv": float(gap_stud["mean_pairwise_tv"]),
                "val_acc": float(stud_metrics.get("val_acc", float("nan"))),
            },
            "make_break": {
                "tv_rel_improve": tv_rel_improve,
                "acc_drop": acc_drop,
                "criterion": f"tv_rel_improve>={float(args.min_tv_rel_improve)} and acc_drop<={float(args.max_acc_drop)}",
                "verdict": verdict,
            },
        }

        (out_dir / "results.json").write_text(json.dumps(summary, indent=2))
        plot_finance_bar_offset_distill(summary, out_dir / "diagnostics.png")

        ctx.logger.log_metrics(
            {
                "baseline/acc": base_acc,
                "baseline/gap_tv": base_gap,
                "student/acc": stud_acc,
                "student/gap_tv": stud_gap,
                "make_break/tv_rel_improve": tv_rel_improve,
                "make_break/acc_drop": acc_drop,
            },
            step=0,
        )

        return summary
