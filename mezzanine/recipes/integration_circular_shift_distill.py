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
from ..symmetries.circular_shift import CircularShiftConfig, CircularShiftSymmetry
from ..worlds.integration_npz import IntegrationNPZAdapter, IntegrationNPZAdapterConfig
from .recipe_base import Recipe


def _view_seed(global_seed: int, i: int, j: int) -> int:
    return int((global_seed * 1000003 + i * 9176 + j * 7919) % (2**32 - 1))


def _featurize_raw(xs: List[Dict[str, Any]]) -> np.ndarray:
    return np.stack(
        [np.asarray(ex["f"], dtype=np.float32).reshape(-1) for ex in xs], axis=0
    )


def _featurize_invariant(xs: List[Dict[str, Any]]) -> np.ndarray:
    """Cheap shift-invariant summary features for periodic 1D fields.

    These are invariant under circular shifts and make it much easier for the
    student to close the circular-shift warrant gap.
    """

    f = np.stack(
        [np.asarray(ex["f"], dtype=np.float32).reshape(-1) for ex in xs], axis=0
    )  # [N,L]
    m1 = f.mean(axis=1, keepdims=True)
    m2 = (f * f).mean(axis=1, keepdims=True)
    return np.concatenate([m1, m2], axis=1).astype(np.float32)


def _targets(xs: List[Dict[str, Any]]) -> np.ndarray:
    return np.stack(
        [np.asarray(ex["y"], dtype=np.float32).reshape(1) for ex in xs], axis=0
    )


class IntegrationCircularShiftDistillRecipe(Recipe):
    NAME = "integration_circular_shift_distill"
    DESCRIPTION = (
        "1D periodic integration: measure circular-shift invariance gap in an integral regressor, "
        "then distill orbit-averaged teacher predictions."
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
            help="Path to integration .npz",
        )
        p.add_argument("--n_train", type=int, default=50000)
        p.add_argument("--n_test", type=int, default=10000)

        # Symmetry
        p.add_argument("--k_train", type=int, default=8)
        p.add_argument("--k_test", type=int, default=16)
        p.add_argument("--max_shift", type=int, default=32)
        # For invariance tests on raw grid inputs, training the student on shifted
        # views (data augmentation) is usually required; otherwise the student can
        # match the orbit-averaged target only on the canonical orientation while
        # still violating invariance off-orbit.
        p.add_argument(
            "--student_train_on_views",
            dest="student_train_on_views",
            action="store_true",
            help="Train the student on shifted views (optional; redundant with invariant featurization).",
        )
        p.add_argument(
            "--no_student_train_on_views",
            dest="student_train_on_views",
            action="store_false",
            help="Disable view augmentation for the student.",
        )
        p.set_defaults(student_train_on_views=False)

        # Model/training
        p.add_argument("--hidden", type=int, default=512)
        p.add_argument("--depth", type=int, default=2)
        p.add_argument("--dropout", type=float, default=0.0)
        p.add_argument("--base_steps", type=int, default=1500)
        p.add_argument("--student_steps", type=int, default=1500)
        p.add_argument("--batch_size", type=int, default=256)
        p.add_argument("--lr", type=float, default=1e-3)
        p.add_argument("--wd", type=float, default=1e-4)
        p.add_argument("--hard_label_weight", type=float, default=0.2)

        args = p.parse_args(argv)
        ctx = self.build_context(args)

        import torch  # type: ignore

        device = "cuda" if torch.cuda.is_available() else "cpu"

        adapter_cfg = IntegrationNPZAdapterConfig(
            path=str(args.dataset),
            n_train=int(args.n_train),
            n_test=int(args.n_test),
            seed=int(args.seed),
        )
        world = IntegrationNPZAdapter(adapter_cfg).load()
        train = world["train"]
        test = world["test"]

        X_train = _featurize_raw(train)
        y_train = _targets(train)
        X_test = _featurize_raw(test)
        y_test = _targets(test)

        # Train/val split
        n = X_train.shape[0]
        idx = np.arange(n)
        n_val = max(1, int(0.2 * n))
        idx_val = idx[:n_val]
        idx_tr = idx[n_val:]

        cfg_teacher = MLPRegressorConfig(
            in_dim=int(X_train.shape[1]),
            out_dim=1,
            hidden=int(args.hidden),
            depth=int(args.depth),
            dropout=float(args.dropout),
        )

        teacher, teacher_metrics = train_regressor(
            X_train[idx_tr],
            y_train[idx_tr],
            X_train[idx_val],
            y_train[idx_val],
            cfg=cfg_teacher,
            steps=int(args.base_steps),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            wd=float(args.wd),
            device=device,
            seed=int(args.seed),
        )

        yhat_test = predict(teacher, X_test, device=device)
        mse_test = float(np.mean((yhat_test - y_test) ** 2))

        sym = CircularShiftSymmetry(
            CircularShiftConfig(key="f", max_shift=int(args.max_shift), axis=0)
        )

        # Teacher gap
        preds_views: List[np.ndarray] = []
        for j in range(int(args.k_test)):
            if j == 0:
                view = test
            else:
                view = [
                    sym.sample(ex, seed=_view_seed(int(args.seed) + 123, i, j))
                    for i, ex in enumerate(test)
                ]
            preds_views.append(predict(teacher, _featurize_raw(view), device=device))
        teacher_gap = warrant_gap_regression(np.stack(preds_views, axis=0))

        # Distill
        train_views: List[List[Dict[str, Any]]] = []
        preds_train_views: List[np.ndarray] = []
        for j in range(int(args.k_train)):
            if j == 0:
                view = train
            else:
                view = [
                    sym.sample(ex, seed=_view_seed(int(args.seed) + 999, i, j))
                    for i, ex in enumerate(train)
                ]
            train_views.append(view)
            preds_train_views.append(
                predict(teacher, _featurize_raw(view), device=device)
            )
        y_soft = np.stack(preds_train_views, axis=0).mean(axis=0)

        # Student uses shift-invariant summary features.
        X_train_student = _featurize_invariant(train)
        X_test_student = _featurize_invariant(test)

        cfg_student = MLPRegressorConfig(
            in_dim=int(X_train_student.shape[1]),
            out_dim=1,
            hidden=int(args.hidden),
            depth=int(args.depth),
            dropout=float(args.dropout),
        )

        if bool(args.student_train_on_views):
            k = int(args.k_train)
            Xs: List[np.ndarray] = []
            ysofts: List[np.ndarray] = []
            yhard: List[np.ndarray] = []
            for view in train_views:
                Xs.append(_featurize_invariant(view))
                # Scalar target is invariant: labels are identical for all shifts.
                ysofts.append(y_soft)
                yhard.append(y_train)
            X_student = np.concatenate(Xs, axis=0)
            y_soft_student = np.concatenate(ysofts, axis=0)
            y_hard_student = np.concatenate(yhard, axis=0)

            def _replicate(idxs: np.ndarray) -> np.ndarray:
                out = []
                for j in range(k):
                    out.append(idxs + j * n)
                return np.concatenate(out, axis=0)

            idx_tr_s = _replicate(idx_tr)
            idx_val_s = _replicate(idx_val)
        else:
            X_student = X_train_student
            y_soft_student = y_soft
            y_hard_student = y_train
            idx_tr_s = idx_tr
            idx_val_s = idx_val

        student, student_metrics = train_regressor_distill(
            X_student[idx_tr_s],
            y_soft_student[idx_tr_s],
            y_hard_student[idx_tr_s],
            X_student[idx_val_s],
            y_soft_student[idx_val_s],
            y_hard_student[idx_val_s],
            cfg=cfg_student,
            steps=int(args.student_steps),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            wd=float(args.wd),
            device=device,
            seed=int(args.seed) + 1,
            hard_label_weight=float(args.hard_label_weight),
        )

        yhat_student_test = predict(student, X_test_student, device=device)
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
                predict(student, _featurize_invariant(view), device=device)
            )
        student_gap = warrant_gap_regression(np.stack(preds_student_views, axis=0))

        gap_improves = student_gap["gap_mse"] <= 0.8 * teacher_gap["gap_mse"]
        mse_ok = student_mse_test <= 1.5 * mse_test
        verdict = "MAKE OK" if (gap_improves and mse_ok) else "BREAK / INCONCLUSIVE X"

        results = {
            "world": {
                "adapter": "integration_npz",
                "adapter_config": asdict(adapter_cfg),
                "n_train": int(len(train)),
                "n_test": int(len(test)),
            },
            "symmetry": {"name": "circular_shift", "max_shift": int(args.max_shift)},
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
            "distill": {
                "k_train": int(args.k_train),
                "k_test": int(args.k_test),
                "student_train_on_views": bool(args.student_train_on_views),
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
