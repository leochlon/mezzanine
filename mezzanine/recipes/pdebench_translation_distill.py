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
from ..symmetries.periodic_translation import (
    PeriodicTranslationConfig,
    PeriodicTranslationSymmetry,
)
from ..worlds.pdebench_h5 import PDEBenchH5Adapter, PDEBenchH5AdapterConfig
from .recipe_base import Recipe


def _view_seed(global_seed: int, i: int, j: int) -> int:
    return int((global_seed * 1000003 + i * 9176 + j * 7919) % (2**32 - 1))


def _flatten_field(a: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=np.float32).reshape(-1)


def _featurize(xs: List[Dict[str, Any]]) -> np.ndarray:
    return np.stack([_flatten_field(ex["x"]) for ex in xs], axis=0)


def _targets(xs: List[Dict[str, Any]]) -> np.ndarray:
    return np.stack([_flatten_field(ex["y"]) for ex in xs], axis=0)


def _parse_group_key(path: str, *, default_group: str) -> tuple[str, str]:
    # Accept 'train/u0', '/train/u0', or just 'u0' (relative to default_group).
    s = str(path).strip()
    if s == "":
        raise ValueError("empty HDF5 key")
    s = s[1:] if s.startswith("/") else s
    if "/" not in s:
        return str(default_group), s
    group, key = s.split("/", 1)
    if group.strip() == "" or key.strip() == "":
        raise ValueError(f"expected key like 'train/u0', got {path!r}")
    return group, key


def _apply_legacy_u0_u1_keys(
    args: argparse.Namespace, p: argparse.ArgumentParser
) -> None:
    """Map legacy flags (--train_u0_key, etc.) onto (train_group/test_group, x_key/y_key)."""

    train_u0 = getattr(args, "train_u0_key", None)
    train_u1 = getattr(args, "train_u1_key", None)
    test_u0 = getattr(args, "test_u0_key", None)
    test_u1 = getattr(args, "test_u1_key", None)

    any_legacy = any(
        v not in (None, "", "None") for v in (train_u0, train_u1, test_u0, test_u1)
    )
    if not any_legacy:
        return

    if any(v in (None, "", "None") for v in (train_u0, train_u1)):
        p.error("Legacy flags require both --train_u0_key and --train_u1_key.")
    if any(v in (None, "", "None") for v in (test_u0, test_u1)):
        p.error("Legacy flags require both --test_u0_key and --test_u1_key.")

    if args.x_key not in (None, "", "None") or args.y_key not in (None, "", "None"):
        p.error("Do not mix legacy --{train,test}_u{0,1}_key with --x_key/--y_key.")

    train_group0, x_key_train = _parse_group_key(
        str(train_u0), default_group=str(args.train_group)
    )
    train_group1, y_key_train = _parse_group_key(
        str(train_u1), default_group=str(args.train_group)
    )
    if train_group0 != train_group1:
        p.error(
            f"train_u0_key and train_u1_key must share a group (got {train_group0!r} vs {train_group1!r})."
        )

    test_group0, x_key_test = _parse_group_key(
        str(test_u0), default_group=str(args.test_group)
    )
    test_group1, y_key_test = _parse_group_key(
        str(test_u1), default_group=str(args.test_group)
    )
    if test_group0 != test_group1:
        p.error(
            f"test_u0_key and test_u1_key must share a group (got {test_group0!r} vs {test_group1!r})."
        )

    if x_key_train != x_key_test or y_key_train != y_key_test:
        p.error(
            "Legacy u0/u1 keys must match across train/test splits. "
            "If your file uses different dataset names per split, use --train_group/--test_group with --x_key/--y_key."
        )

    args.train_group = str(train_group0)
    args.test_group = str(test_group0)
    args.x_key = str(x_key_train)
    args.y_key = str(y_key_train)


def _inverse_translate(
    arr: np.ndarray, shifts: List[int], axes: List[int], out_shape: tuple[int, ...]
) -> np.ndarray:
    # arr is flat; reshape, roll back, flatten
    y = arr.reshape(out_shape)
    for ax, sh in zip(axes, shifts):
        y = np.roll(y, shift=-int(sh), axis=int(ax))
    return y.reshape(-1)


class PDEBenchTranslationDistillRecipe(Recipe):
    NAME = "pdebench_translation_distill"
    DESCRIPTION = (
        "PDEBench operator learning: measure periodic translation equivariance gap and distill an orbit-averaged "
        "teacher into a single-pass student."
    )

    def _build_arg_parser(self) -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(prog=self.NAME)
        self.add_common_args(p)

        # Data
        p.add_argument(
            "--dataset",
            "--data",
            dest="dataset",
            type=str,
            required=True,
            help="Path to PDEBench HDF5",
        )
        p.add_argument("--n_train", type=int, default=20000)
        p.add_argument("--n_test", type=int, default=5000)
        p.add_argument("--train_group", type=str, default="train")
        p.add_argument("--test_group", type=str, default="test")
        p.add_argument("--u_key", type=str, default="u")
        p.add_argument("--x_key", type=str, default=None)
        p.add_argument("--y_key", type=str, default=None)
        p.add_argument("--t0", type=int, default=0)
        p.add_argument("--t1", type=int, default=-1)
        p.add_argument("--spatial_slice", type=str, default=None)

        # Legacy (older notebooks/scripts): explicit split paths like 'train/u0' and 'test/u1'.
        p.add_argument(
            "--train_u0_key",
            type=str,
            default=None,
            help="Legacy: HDF5 path for train inputs (u0).",
        )
        p.add_argument(
            "--train_u1_key",
            type=str,
            default=None,
            help="Legacy: HDF5 path for train targets (u1).",
        )
        p.add_argument(
            "--test_u0_key",
            type=str,
            default=None,
            help="Legacy: HDF5 path for test inputs (u0).",
        )
        p.add_argument(
            "--test_u1_key",
            type=str,
            default=None,
            help="Legacy: HDF5 path for test targets (u1).",
        )

        # Symmetry
        p.add_argument(
            "--axes",
            type=str,
            default="0",
            help="Comma-separated axes to roll (e.g., '0' or '1,2')",
        )
        p.add_argument("--max_shift", type=int, default=8)
        p.add_argument("--k_train", type=int, default=4)
        p.add_argument("--k_test", type=int, default=8)
        p.add_argument("--student_train_on_views", action="store_true")

        # Model/training
        p.add_argument("--hidden", type=int, default=1024)
        p.add_argument("--depth", type=int, default=2)
        p.add_argument("--dropout", type=float, default=0.0)
        p.add_argument("--base_steps", type=int, default=2000)
        p.add_argument("--student_steps", type=int, default=2000)
        p.add_argument("--batch_size", type=int, default=64)
        p.add_argument("--lr", type=float, default=1e-3)
        p.add_argument("--wd", type=float, default=1e-4)
        p.add_argument("--hard_label_weight", type=float, default=0.2)
        return p

    def run(self, argv: list[str]) -> Dict[str, Any]:
        p = self._build_arg_parser()
        args = p.parse_args(argv)
        _apply_legacy_u0_u1_keys(args, p)
        ctx = self.build_context(args)

        import torch  # type: ignore

        device = "cuda" if torch.cuda.is_available() else "cpu"

        axes = [int(x.strip()) for x in str(args.axes).split(",") if x.strip() != ""]

        adapter_cfg = PDEBenchH5AdapterConfig(
            path=str(args.dataset),
            n_train=int(args.n_train),
            n_test=int(args.n_test),
            seed=int(args.seed),
            train_group=str(args.train_group),
            test_group=str(args.test_group),
            u_key=str(args.u_key),
            x_key=(None if args.x_key in (None, "None", "") else str(args.x_key)),
            y_key=(None if args.y_key in (None, "None", "") else str(args.y_key)),
            t0=int(args.t0),
            t1=int(args.t1),
            spatial_slice=(
                None
                if args.spatial_slice in (None, "None", "")
                else str(args.spatial_slice)
            ),
        )
        world = PDEBenchH5Adapter(adapter_cfg).load()
        train = world["train"]
        test = world["test"]

        X_train = _featurize(train)
        y_train = _targets(train)
        X_test = _featurize(test)
        y_test = _targets(test)

        out_shape = (
            tuple(np.asarray(train[0]["y"]).shape)
            if train
            else tuple(np.asarray(test[0]["y"]).shape)
        )
        out_dim = int(np.prod(out_shape))

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

        sym = PeriodicTranslationSymmetry(
            PeriodicTranslationConfig(
                keys=["x", "y"], axes=axes, max_shift=int(args.max_shift)
            )
        )

        # Teacher gap: translate inputs, predict, inverse-translate predictions back
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
            # canonicalize
            canon = np.empty_like(pred)
            for i, ex in enumerate(view):
                shifts = ex.get("_shifts", [0] * len(axes))
                canon[i] = _inverse_translate(pred[i], list(shifts), axes, out_shape)
            teacher_view_preds.append(canon)
        teacher_gap = warrant_gap_regression(np.stack(teacher_view_preds, axis=0))

        # Distillation: orbit-average teacher predictions (canonical)
        train_views: List[List[Dict[str, Any]]] = []
        train_view_preds: List[np.ndarray] = []
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
            canon = np.empty_like(pred)
            for i, ex in enumerate(view):
                shifts = ex.get("_shifts", [0] * len(axes))
                canon[i] = _inverse_translate(pred[i], list(shifts), axes, out_shape)
            train_view_preds.append(canon)
        mu = np.stack(train_view_preds, axis=0).mean(axis=0)  # [N,out_dim] canonical

        # Student training set: optionally train on views with equivariant targets
        if bool(args.student_train_on_views):
            Xs: List[np.ndarray] = []
            ysofts: List[np.ndarray] = []
            yhard: List[np.ndarray] = []
            for view in train_views:
                Xs.append(_featurize(view))
                ys = np.empty_like(mu)
                yh = np.empty_like(y_train)
                for i, ex in enumerate(view):
                    shifts = list(ex.get("_shifts", [0] * len(axes)))
                    # forward translate canonical mu into view coordinates
                    y_mu = mu[i].reshape(out_shape)
                    y_gt = y_train[i].reshape(out_shape)
                    for ax, sh in zip(axes, shifts):
                        y_mu = np.roll(y_mu, shift=int(sh), axis=int(ax))
                        y_gt = np.roll(y_gt, shift=int(sh), axis=int(ax))
                    ys[i] = y_mu.reshape(-1)
                    yh[i] = y_gt.reshape(-1)
                ysofts.append(ys)
                yhard.append(yh)
            X_student = np.concatenate(Xs, axis=0)
            y_soft_student = np.concatenate(ysofts, axis=0)
            y_hard_student = np.concatenate(yhard, axis=0)
        else:
            X_student = X_train
            y_soft_student = mu
            y_hard_student = y_train

        # Split student train/val
        if bool(args.student_train_on_views):
            k = int(args.k_train)

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

        # Student gap
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
            canon = np.empty_like(pred)
            for i, ex in enumerate(view):
                shifts = ex.get("_shifts", [0] * len(axes))
                canon[i] = _inverse_translate(pred[i], list(shifts), axes, out_shape)
            student_view_preds.append(canon)
        student_gap = warrant_gap_regression(np.stack(student_view_preds, axis=0))

        gap_improves = student_gap["gap_mse"] <= 0.8 * teacher_gap["gap_mse"]
        mse_ok = student_mse_test <= 1.5 * mse_test
        verdict = "MAKE OK" if (gap_improves and mse_ok) else "BREAK / INCONCLUSIVE X"

        results = {
            "world": {
                "adapter": "pdebench_h5",
                "adapter_config": asdict(adapter_cfg),
                "n_train": int(len(train)),
                "n_test": int(len(test)),
            },
            "symmetry": {
                "name": "periodic_translation",
                "axes": axes,
                "max_shift": int(args.max_shift),
            },
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
