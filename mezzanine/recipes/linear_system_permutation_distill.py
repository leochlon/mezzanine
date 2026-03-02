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
from ..worlds.linear_system_npz import (
    LinearSystemNPZAdapter,
    LinearSystemNPZAdapterConfig,
)
from .recipe_base import Recipe


def _view_seed(global_seed: int, i: int, j: int) -> int:
    return int((global_seed * 1000003 + i * 9176 + j * 7919) % (2**32 - 1))


def _featurize(xs: List[Dict[str, Any]]) -> np.ndarray:
    A = np.stack(
        [np.asarray(ex["A"], dtype=np.float32).reshape(-1) for ex in xs], axis=0
    )
    b = np.stack(
        [np.asarray(ex["b"], dtype=np.float32).reshape(-1) for ex in xs], axis=0
    )
    return np.concatenate([A, b], axis=1)


def _targets(xs: List[Dict[str, Any]]) -> np.ndarray:
    return np.stack(
        [np.asarray(ex["x"], dtype=np.float32).reshape(-1) for ex in xs], axis=0
    )


def _unpermute(vec: np.ndarray, perm: np.ndarray) -> np.ndarray:
    # vec is [n]; perm maps canonical -> permuted indices.
    inv = np.argsort(perm)
    return vec[inv]


def _canonicalize_preds(pred: np.ndarray, views: List[Dict[str, Any]]) -> np.ndarray:
    """Map predictions from each view back into canonical node order.

    Args:
      pred: [N,n]
      views: list of N examples; each may contain _perm.
    """
    out = np.empty_like(pred)
    for i, ex in enumerate(views):
        perm = ex.get("_perm", None)
        if perm is None:
            out[i] = pred[i]
        else:
            out[i] = _unpermute(pred[i], np.asarray(perm, dtype=np.int64))
    return out


class LinearSystemPermutationDistillRecipe(Recipe):
    NAME = "linear_system_permutation_distill"
    DESCRIPTION = (
        "Linear systems: measure node-relabeling (permutation similarity) warrant gap and distill a "
        "permutation-canonicalized teacher into a single-pass student."
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
            help="Path to linear system .npz",
        )
        p.add_argument("--n_train", type=int, default=50000)
        p.add_argument("--n_test", type=int, default=10000)

        # Symmetry
        p.add_argument("--k_train", type=int, default=4)
        p.add_argument("--k_test", type=int, default=8)
        p.add_argument(
            "--student_train_on_views",
            action="store_true",
            help="Train student on permuted views (equivariant)",
        )

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

        adapter_cfg = LinearSystemNPZAdapterConfig(
            path=str(args.dataset),
            n_train=int(args.n_train),
            n_test=int(args.n_test),
            seed=int(args.seed),
        )
        world = LinearSystemNPZAdapter(adapter_cfg).load()
        train = world["train"]
        test = world["test"]

        X_train = _featurize(train)
        y_train = _targets(train)
        X_test = _featurize(test)
        y_test = _targets(test)
        n_out = int(y_train.shape[1])

        # Train/val split
        n = X_train.shape[0]
        idx = np.arange(n)
        n_val = max(1, int(0.2 * n))
        idx_val = idx[:n_val]
        idx_tr = idx[n_val:]

        cfg = MLPRegressorConfig(
            in_dim=int(X_train.shape[1]),
            out_dim=n_out,
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

        # Residual norm ||Ax-b|| on test (gold physics check)
        A_test = np.stack(
            [np.asarray(ex["A"], dtype=np.float32) for ex in test], axis=0
        )
        b_test = np.stack(
            [np.asarray(ex["b"], dtype=np.float32) for ex in test], axis=0
        )
        resid = A_test @ yhat_test[..., None]
        resid = resid[..., 0] - b_test
        resid_l2 = float(np.mean(np.linalg.norm(resid, axis=1)))

        sym = NodePermutationSymmetry(NodePermutationConfig())

        # Teacher symmetry gap on test
        teacher_view_preds: List[np.ndarray] = []
        for j in range(int(args.k_test)):
            if j == 0:
                view = test
            else:
                view = [
                    sym.sample(ex, seed=_view_seed(int(args.seed) + 123, i, j))
                    for i, ex in enumerate(test)
                ]
            pred = predict(teacher, _featurize(view), device=device)
            pred_canon = _canonicalize_preds(pred, view)
            teacher_view_preds.append(pred_canon)
        teacher_gap = warrant_gap_regression(np.stack(teacher_view_preds, axis=0))

        # Distillation: orbit-average teacher predictions in canonical order
        train_view_preds: List[np.ndarray] = []
        train_views: List[List[Dict[str, Any]]] = []
        for j in range(int(args.k_train)):
            if j == 0:
                view = train
            else:
                view = [
                    sym.sample(ex, seed=_view_seed(int(args.seed) + 999, i, j))
                    for i, ex in enumerate(train)
                ]
            train_views.append(view)
            pred = predict(teacher, _featurize(view), device=device)
            pred_canon = _canonicalize_preds(pred, view)
            train_view_preds.append(pred_canon)
        mu = np.stack(train_view_preds, axis=0).mean(axis=0)  # [N,n] canonical

        # Build student training set
        if bool(args.student_train_on_views):
            Xs: List[np.ndarray] = []
            ysofts: List[np.ndarray] = []
            yhard: List[np.ndarray] = []
            # For each view, the correct output is permuted(mu)
            for view in train_views:
                Xs.append(_featurize(view))
                ys = np.empty_like(mu)
                yh = np.empty_like(y_train)
                for i, ex in enumerate(view):
                    perm = ex.get("_perm", None)
                    if perm is None:
                        ys[i] = mu[i]
                        yh[i] = y_train[i]
                    else:
                        perm_arr = np.asarray(perm, dtype=np.int64)
                        ys[i] = mu[i][perm_arr]
                        yh[i] = y_train[i][perm_arr]
                ysofts.append(ys)
                yhard.append(yh)
            X_student = np.concatenate(Xs, axis=0)
            y_soft_student = np.concatenate(ysofts, axis=0)
            y_hard_student = np.concatenate(yhard, axis=0)
        else:
            X_student = X_train
            y_soft_student = mu
            y_hard_student = y_train

        # Split student train/val consistently (use indices from canonical set)
        # If training on views, we replicate the split across concatenated blocks.
        if bool(args.student_train_on_views):
            k = int(args.k_train)

            # replicate indices into each block
            def _replicate(idxs: np.ndarray) -> np.ndarray:
                out = []
                for j in range(k):
                    out.append(idxs + j * n)
                return np.concatenate(out, axis=0)

            idx_tr_s = _replicate(idx_tr)
            idx_val_s = _replicate(idx_val)
        else:
            idx_tr_s = idx_tr
            idx_val_s = idx_val

        student, student_metrics = train_regressor_distill(
            X_student[idx_tr_s],
            y_soft_student[idx_tr_s],
            y_hard_student[idx_tr_s],
            X_student[idx_val_s],
            y_soft_student[idx_val_s],
            y_hard_student[idx_val_s],
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
        resid_s = A_test @ yhat_student_test[..., None]
        resid_s = resid_s[..., 0] - b_test
        student_resid_l2 = float(np.mean(np.linalg.norm(resid_s, axis=1)))

        # Student gap on test
        student_view_preds: List[np.ndarray] = []
        for j in range(int(args.k_test)):
            if j == 0:
                view = test
            else:
                view = [
                    sym.sample(ex, seed=_view_seed(int(args.seed) + 123, i, j))
                    for i, ex in enumerate(test)
                ]
            pred = predict(student, _featurize(view), device=device)
            pred_canon = _canonicalize_preds(pred, view)
            student_view_preds.append(pred_canon)
        student_gap = warrant_gap_regression(np.stack(student_view_preds, axis=0))

        gap_improves = student_gap["gap_mse"] <= 0.8 * teacher_gap["gap_mse"]
        mse_ok = student_mse_test <= 1.5 * mse_test
        verdict = "MAKE OK" if (gap_improves and mse_ok) else "BREAK / INCONCLUSIVE X"

        results = {
            "world": {
                "adapter": "linear_system_npz",
                "adapter_config": asdict(adapter_cfg),
                "n_train": int(len(train)),
                "n_test": int(len(test)),
            },
            "symmetry": {"name": "node_permutation"},
            "teacher": {
                "metrics": {
                    **teacher_metrics,
                    "mse_test": mse_test,
                    "resid_l2": resid_l2,
                    **teacher_gap,
                }
            },
            "student": {
                "metrics": {
                    **student_metrics,
                    "mse_test": student_mse_test,
                    "resid_l2": student_resid_l2,
                    **student_gap,
                }
            },
            "distill": {
                "k_train": int(args.k_train),
                "k_test": int(args.k_test),
                "student_train_on_views": bool(args.student_train_on_views),
                "hard_label_weight": float(args.hard_label_weight),
            },
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
