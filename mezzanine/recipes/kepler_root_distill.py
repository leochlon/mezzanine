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
from ..symmetries.angle_wrap import AngleWrapConfig, AngleWrapSymmetry
from ..worlds.kepler_root_npz import KeplerRootNPZAdapter, KeplerRootNPZAdapterConfig
from .recipe_base import Recipe


def _view_seed(global_seed: int, i: int, j: int) -> int:
    return int((global_seed * 1000003 + i * 9176 + j * 7919) % (2**32 - 1))


def _featurize(xs: List[Dict[str, Any]]) -> np.ndarray:
    # Use a periodic embedding for the mean anomaly M.
    #
    # If we featurize M as a raw scalar, the symmetry action M -> M + 2πk
    # sends us out-of-distribution (since the generator samples M in [0,2π)),
    # and the teacher can extrapolate catastrophically. Using (sin M, cos M)
    # makes the representation itself invariant to 2π wraps.
    e = np.array([float(ex["e"]) for ex in xs], dtype=np.float32)
    M64 = np.array([float(ex["M"]) for ex in xs], dtype=np.float64)
    # Canonicalize the angle before trig to avoid tiny numerical differences
    # between M and M + 2πk from trig argument reduction.
    period = float(2.0 * np.pi)
    M_wrap = np.remainder(M64, period)
    s = np.sin(M_wrap).astype(np.float32)
    c = np.cos(M_wrap).astype(np.float32)
    return np.stack([e, s, c], axis=1)


def _targets(xs: List[Dict[str, Any]]) -> np.ndarray:
    return np.stack(
        [np.asarray(ex["y"], dtype=np.float32).reshape(2) for ex in xs], axis=0
    )


def _wrap_to_pi(a: np.ndarray) -> np.ndarray:
    # wrap to [-pi, pi]
    return (a + np.pi) % (2.0 * np.pi) - np.pi


class KeplerRootDistillRecipe(Recipe):
    NAME = "kepler_root_distill"
    DESCRIPTION = (
        "Kepler root-finding: measure 2π angle-wrap warrant gap and distill orbit-averaged teacher predictions "
        "into a single-pass student."
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
            help="Path to kepler .npz",
        )
        p.add_argument("--n_train", type=int, default=50000)
        p.add_argument("--n_test", type=int, default=10000)

        # Symmetry (M -> M + 2πk)
        p.add_argument("--max_k", type=int, default=2)
        p.add_argument("--k_train", type=int, default=8)
        p.add_argument("--k_test", type=int, default=16)

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

        adapter_cfg = KeplerRootNPZAdapterConfig(
            path=str(args.dataset),
            n_train=int(args.n_train),
            n_test=int(args.n_test),
            seed=int(args.seed),
        )
        world = KeplerRootNPZAdapter(adapter_cfg).load()
        train = world["train"]
        test = world["test"]

        X_train = _featurize(train)
        y_train = _targets(train)
        X_test = _featurize(test)
        y_test = _targets(test)

        # Train/val split
        n = X_train.shape[0]
        idx = np.arange(n)
        n_val = max(1, int(0.2 * n))
        idx_val = idx[:n_val]
        idx_tr = idx[n_val:]

        X_tr, y_tr = X_train[idx_tr], y_train[idx_tr]
        X_val, y_val = X_train[idx_val], y_train[idx_val]

        cfg = MLPRegressorConfig(
            in_dim=int(X_train.shape[1]),
            out_dim=int(y_train.shape[1]),
            hidden=int(args.hidden),
            depth=int(args.depth),
            dropout=float(args.dropout),
        )

        teacher, teacher_metrics = train_regressor(
            X_tr,
            y_tr,
            X_val,
            y_val,
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

        # Symmetry views on test
        sym = AngleWrapSymmetry(
            AngleWrapConfig(field="M", period=float(2.0 * np.pi), max_k=int(args.max_k))
        )
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

        # Distillation: orbit-average teacher predictions across angle wraps
        train_views = []
        preds_train_views = []
        for j in range(int(args.k_train)):
            if j == 0:
                view = train
            else:
                view = [
                    sym.sample(ex, seed=_view_seed(int(args.seed) + 999, i, j))
                    for i, ex in enumerate(train)
                ]
            train_views.append(view)
            preds_train_views.append(predict(teacher, _featurize(view), device=device))

        preds_train_views_arr = np.stack(preds_train_views, axis=0)  # [K,N,2]
        y_soft = preds_train_views_arr.mean(axis=0)
        y_soft_tr, y_soft_val = y_soft[idx_tr], y_soft[idx_val]

        student, student_metrics = train_regressor_distill(
            X_train[idx_tr],
            y_soft_tr,
            y_train[idx_tr],
            X_train[idx_val],
            y_soft_val,
            y_val,
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

        # Student gap on same test views
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

        # Kepler residual diagnostics (uses principal-value E from atan2)
        e_test = np.array([float(ex["e"]) for ex in test], dtype=np.float32)
        M_test = np.array([float(ex["M"]) for ex in test], dtype=np.float32)

        def residual_from_sincos(pred: np.ndarray) -> float:
            s = pred[:, 0]
            c = pred[:, 1]
            E = np.arctan2(s, c)
            # bring to [0,2pi)
            E = (E + 2.0 * np.pi) % (2.0 * np.pi)
            r = E - e_test * np.sin(E) - M_test
            r = _wrap_to_pi(r)
            return float(np.mean(np.abs(r)))

        teacher_kepler_resid = residual_from_sincos(yhat_test)
        student_kepler_resid = residual_from_sincos(yhat_student_test)

        gap_improves = student_gap["gap_mse"] <= 0.8 * teacher_gap["gap_mse"]
        mse_ok = student_mse_test <= 1.5 * mse_test

        verdict = "MAKE OK" if (gap_improves and mse_ok) else "BREAK / INCONCLUSIVE X"

        results = {
            "world": {
                "adapter": "kepler_root_npz",
                "adapter_config": asdict(adapter_cfg),
                "n_train": int(len(train)),
                "n_test": int(len(test)),
            },
            "symmetry": {"name": "angle_wrap", "max_k": int(args.max_k)},
            "teacher": {
                "metrics": {
                    **teacher_metrics,
                    "mse_test": mse_test,
                    **teacher_gap,
                    "kepler_abs_residual": teacher_kepler_resid,
                }
            },
            "student": {
                "metrics": {
                    **student_metrics,
                    "mse_test": student_mse_test,
                    **student_gap,
                    "kepler_abs_residual": student_kepler_resid,
                }
            },
            "distill": {
                "k_train": int(args.k_train),
                "k_test": int(args.k_test),
                "max_k": int(args.max_k),
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
