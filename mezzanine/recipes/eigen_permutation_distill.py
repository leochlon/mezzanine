from __future__ import annotations

import argparse
from dataclasses import asdict
from typing import Any, Dict, List

import numpy as np

from ..pipelines.regression_distill import (
    MLPRegressorConfig,
    predict,
    train_regressor,
    train_regressor_distill,
    warrant_gap_regression,
)
from ..symmetries.node_permutation import NodePermutationConfig, NodePermutationSymmetry
from ..worlds.eigen_npz import EigenNPZAdapter, EigenNPZAdapterConfig
from .recipe_base import Recipe


def _view_seed(global_seed: int, i: int, j: int) -> int:
    return int((global_seed * 1000003 + i * 9176 + j * 7919) % (2**32 - 1))


def _featurize(xs: List[Dict[str, Any]]) -> np.ndarray:
    return np.stack(
        [np.asarray(ex["A"], dtype=np.float32).reshape(-1) for ex in xs], axis=0
    )


def _targets(xs: List[Dict[str, Any]]) -> np.ndarray:
    return np.stack(
        [np.asarray(ex["eval"], dtype=np.float32).reshape(-1) for ex in xs], axis=0
    )


class EigenPermutationDistillRecipe(Recipe):
    NAME = "eigen_permutation_distill"
    DESCRIPTION = (
        "Eigenvalues: measure permutation similarity (A->PAP^T) warrant gap and distill orbit-averaged teacher "
        "predictions into a single-pass student."
    )

    def run(self, argv: list[str]) -> Dict[str, Any]:
        p = argparse.ArgumentParser(prog=self.NAME)
        self.add_common_args(p)

        # Data
        p.add_argument(
            "--dataset",
            "--data",
            dest="dataset",
            type=str,
            required=True,
            help="Path to eigen .npz",
        )
        p.add_argument("--n_train", type=int, default=50000)
        p.add_argument("--n_test", type=int, default=10000)

        # Symmetry
        p.add_argument("--k_train", type=int, default=4)
        p.add_argument("--k_test", type=int, default=8)

        # Model/training
        p.add_argument("--hidden", type=int, default=1024)
        p.add_argument("--depth", type=int, default=2)
        p.add_argument("--dropout", type=float, default=0.0)
        p.add_argument("--base_steps", type=int, default=2000)
        p.add_argument("--student_steps", type=int, default=2000)
        p.add_argument("--batch_size", type=int, default=128)
        p.add_argument("--lr", type=float, default=1e-3)
        p.add_argument("--wd", type=float, default=1e-4)
        p.add_argument("--hard_label_weight", type=float, default=0.2)

        args = p.parse_args(argv)
        ctx = self.build_context(args)

        import torch  # type: ignore

        device = "cuda" if torch.cuda.is_available() else "cpu"

        adapter_cfg = EigenNPZAdapterConfig(
            path=str(args.dataset),
            n_train=int(args.n_train),
            n_test=int(args.n_test),
            seed=int(args.seed),
        )
        world = EigenNPZAdapter(adapter_cfg).load()
        train = world["train"]
        test = world["test"]

        X_train = _featurize(train)
        y_train = _targets(train)
        X_test = _featurize(test)
        y_test = _targets(test)
        out_dim = int(y_train.shape[1])

        # Train/val split
        n = X_train.shape[0]
        idx = np.arange(n)
        n_val = max(1, int(0.2 * n))
        idx_val = idx[:n_val]
        idx_tr = idx[n_val:]

        cfg = MLPRegressorConfig(
            in_dim=int(X_train.shape[1]),
            out_dim=out_dim,
            hidden=int(args.hidden),
            depth=int(args.depth),
            dropout=float(args.dropout),
        )

        teacher, teacher_metrics = train_regressor(
            X_train[idx_tr],
            y_train[idx_tr],
            X_train[idx_val],
            y_train[idx_val],
            cfg=cfg,
            steps=int(args.base_steps),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            wd=float(args.wd),
            device=device,
            seed=int(args.seed),
        )

        yhat_test = predict(teacher, X_test, device=device)
        mse_test = float(np.mean((yhat_test - y_test) ** 2))

        sym = NodePermutationSymmetry(
            NodePermutationConfig(key_A="A", key_b="b", key_x="x")
        )

        # Teacher gap: eigenvalues should be invariant under permutation similarity
        preds_views: List[np.ndarray] = []
        for j in range(int(args.k_test)):
            if j == 0:
                view = test
            else:
                view = [
                    sym.sample(ex, seed=_view_seed(int(args.seed) + 123, i, j))
                    for i, ex in enumerate(test)
                ]
            preds_views.append(predict(teacher, _featurize(view), device=device))
        teacher_gap = warrant_gap_regression(np.stack(preds_views, axis=0))

        # Distill: orbit-average teacher predictions
        preds_train_views: List[np.ndarray] = []
        for j in range(int(args.k_train)):
            if j == 0:
                view = train
            else:
                view = [
                    sym.sample(ex, seed=_view_seed(int(args.seed) + 999, i, j))
                    for i, ex in enumerate(train)
                ]
            preds_train_views.append(predict(teacher, _featurize(view), device=device))
        y_soft = np.stack(preds_train_views, axis=0).mean(axis=0)

        student, student_metrics = train_regressor_distill(
            X_train[idx_tr],
            y_soft[idx_tr],
            y_train[idx_tr],
            X_train[idx_val],
            y_soft[idx_val],
            y_train[idx_val],
            cfg=cfg,
            steps=int(args.student_steps),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            wd=float(args.wd),
            device=device,
            seed=int(args.seed) + 1,
            hard_label_weight=float(args.hard_label_weight),
        )

        yhat_student_test = predict(student, X_test, device=device)
        student_mse_test = float(np.mean((yhat_student_test - y_test) ** 2))

        preds_student_views: List[np.ndarray] = []
        for j in range(int(args.k_test)):
            if j == 0:
                view = test
            else:
                view = [
                    sym.sample(ex, seed=_view_seed(int(args.seed) + 123, i, j))
                    for i, ex in enumerate(test)
                ]
            preds_student_views.append(
                predict(student, _featurize(view), device=device)
            )
        student_gap = warrant_gap_regression(np.stack(preds_student_views, axis=0))

        gap_improves = student_gap["gap_mse"] <= 0.8 * teacher_gap["gap_mse"]
        mse_ok = student_mse_test <= 1.5 * mse_test
        verdict = "MAKE OK" if (gap_improves and mse_ok) else "BREAK / INCONCLUSIVE X"

        results = {
            "world": {
                "adapter": "eigen_npz",
                "adapter_config": asdict(adapter_cfg),
                "n_train": int(len(train)),
                "n_test": int(len(test)),
            },
            "symmetry": {"name": "node_permutation"},
            "teacher": {
                "metrics": {**teacher_metrics, "mse_test": mse_test, **teacher_gap}
            },
            "student": {
                "metrics": {
                    **student_metrics,
                    "mse_test": student_mse_test,
                    **student_gap,
                }
            },
            "distill": {"k_train": int(args.k_train), "k_test": int(args.k_test)},
            "make_break": {
                "criterion": {
                    "gap_mse reduces by >=20%": bool(gap_improves),
                    "student mse not worse by >50%": bool(mse_ok),
                },
                "verdict": verdict,
            },
        }

        self.save_json(results, ctx)
        return results
