from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..pipelines.regression_distill import (
    MLPRegressorConfig,
    predict,
    train_regressor,
    train_regressor_distill,
    warrant_gap_regression,
)
from ..symmetries.time_origin_shift import (
    TimeOriginShiftConfig,
    TimeOriginShiftSymmetry,
)
from ..worlds.ode_npz import ODENPZAdapter, ODENPZAdapterConfig
from .recipe_base import Recipe


def _view_seed(global_seed: int, i: int, j: int) -> int:
    return int((global_seed * 1000003 + i * 9176 + j * 7919) % (2**32 - 1))


def _featurize(xs: List[Dict[str, Any]], *, include_time: bool) -> np.ndarray:
    states = np.stack(
        [np.asarray(ex["state"], dtype=np.float32).reshape(-1) for ex in xs], axis=0
    )
    if not include_time:
        return states
    t = np.array([float(ex.get("t", 0.0)) for ex in xs], dtype=np.float32)[:, None]
    return np.concatenate([states, t], axis=1)


def _targets(xs: List[Dict[str, Any]]) -> np.ndarray:
    return np.stack(
        [np.asarray(ex["next_state"], dtype=np.float32).reshape(-1) for ex in xs],
        axis=0,
    )


def _load_trajs_npz(path: str, split: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    p = Path(path)
    with np.load(p, allow_pickle=False) as z:
        x = z[f"{split}_x"].astype(np.float32)
        t = z.get(f"{split}_t", None)
        if t is not None:
            t = np.asarray(t).astype(np.float32)
        return x, t


def _rollout_mse(
    model: Any,
    *,
    examples: List[Dict[str, Any]],
    trajs: np.ndarray,
    include_time: bool,
    t_arr: Optional[np.ndarray],
    horizon: int,
    device: str,
) -> float:
    """Compute rollout MSE over a small set of seeded transitions.

    We roll out from x_{t} for `horizon` steps using the learned one-step model,
    and compare to the ground-truth trajectory stored in trajs.
    """

    if horizon <= 1:
        return float("nan")
    errs: List[float] = []

    # Use up to 256 rollouts for speed
    n_eval = min(256, len(examples))
    for ex in examples[:n_eval]:
        if "traj_idx" not in ex or "t_idx" not in ex:
            continue
        traj_idx = int(ex["traj_idx"])
        t0 = int(ex["t_idx"])
        # ensure we have enough future states
        if t0 + horizon >= trajs.shape[1]:
            continue

        x_cur = np.asarray(ex["state"], dtype=np.float32).reshape(1, -1)
        for h in range(horizon):
            # build feature with optional time
            if include_time:
                if t_arr is None:
                    tt = np.array([[float(t0 + h)]], dtype=np.float32)
                else:
                    if t_arr.ndim == 1:
                        tt = np.array([[float(t_arr[t0 + h])]], dtype=np.float32)
                    else:
                        tt = np.array(
                            [[float(t_arr[traj_idx, t0 + h])]], dtype=np.float32
                        )
                X_in = np.concatenate([x_cur, tt], axis=1)
            else:
                X_in = x_cur
            x_next = predict(model, X_in, device=device)
            x_cur = x_next

        gt = trajs[traj_idx, t0 + horizon].reshape(1, -1)
        errs.append(float(np.mean((x_cur - gt) ** 2)))

    return float(np.mean(errs)) if errs else float("nan")


class ODETimeOriginDistillRecipe(Recipe):
    NAME = "ode_time_origin_distill"
    DESCRIPTION = (
        "ODE one-step surrogate: measure time-origin shift (t->t+c) warrant gap for time-conditioned teachers, "
        "then distill orbit-averaged teacher predictions into a time-agnostic student."
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
            help="Path to ODE .npz",
        )
        p.add_argument("--n_train", type=int, default=50000)
        p.add_argument("--n_test", type=int, default=10000)
        p.add_argument("--teacher_include_time", action="store_true")
        p.add_argument("--rollout_horizon", type=int, default=25)

        # Symmetry
        p.add_argument("--k_train", type=int, default=8)
        p.add_argument("--k_test", type=int, default=16)
        p.add_argument("--max_shift", type=float, default=5.0)

        # Model/training
        p.add_argument("--hidden", type=int, default=512)
        p.add_argument("--depth", type=int, default=2)
        p.add_argument("--dropout", type=float, default=0.0)
        p.add_argument("--base_steps", type=int, default=3000)
        p.add_argument("--student_steps", type=int, default=3000)
        p.add_argument("--batch_size", type=int, default=256)
        p.add_argument("--lr", type=float, default=1e-3)
        p.add_argument("--wd", type=float, default=1e-4)
        p.add_argument("--hard_label_weight", type=float, default=0.2)

        args = p.parse_args(argv)
        ctx = self.build_context(args)

        import torch  # type: ignore

        device = "cuda" if torch.cuda.is_available() else "cpu"

        include_time_teacher = bool(args.teacher_include_time)

        adapter_cfg = ODENPZAdapterConfig(
            path=str(args.dataset),
            n_train=int(args.n_train),
            n_test=int(args.n_test),
            seed=int(args.seed),
            include_time=True,  # keep time in examples; we decide whether to use it
            include_indices=True,
        )
        world = ODENPZAdapter(adapter_cfg).load()
        train = world["train"]
        test = world["test"]

        X_train = _featurize(train, include_time=include_time_teacher)
        y_train = _targets(train)
        X_test = _featurize(test, include_time=include_time_teacher)
        y_test = _targets(test)

        # Train/val split
        n = X_train.shape[0]
        idx = np.arange(n)
        n_val = max(1, int(0.2 * n))
        idx_val = idx[:n_val]
        idx_tr = idx[n_val:]

        cfg = MLPRegressorConfig(
            in_dim=int(X_train.shape[1]),
            out_dim=int(y_train.shape[1]),
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

        # Time-origin shift symmetry views
        # If the dataset provides time bounds, enforce them to avoid OOD time values.
        t_vals = np.array(
            [float(ex.get("t", 0.0)) for ex in train + test], dtype=np.float32
        )
        t_min = float(np.min(t_vals))
        t_max = float(np.max(t_vals))

        sym = TimeOriginShiftSymmetry(
            TimeOriginShiftConfig(
                max_shift=float(args.max_shift), t_min=t_min, t_max=t_max
            )
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
            Xv = _featurize(view, include_time=include_time_teacher)
            preds_views.append(predict(teacher, Xv, device=device))
        teacher_gap = warrant_gap_regression(np.stack(preds_views, axis=0))

        # Rollout MSE against stored trajectories
        trajs_test, t_arr = _load_trajs_npz(str(args.dataset), "test")
        teacher_roll = _rollout_mse(
            teacher,
            examples=test,
            trajs=trajs_test,
            include_time=include_time_teacher,
            t_arr=t_arr,
            horizon=int(args.rollout_horizon),
            device=device,
        )

        # Distill: orbit-average teacher predictions
        train_preds_views: List[np.ndarray] = []
        for j in range(int(args.k_train)):
            if j == 0:
                view = train
            else:
                view = [
                    sym.sample(ex, seed=_view_seed(int(args.seed) + 999, i, j))
                    for i, ex in enumerate(train)
                ]
            Xv = _featurize(view, include_time=include_time_teacher)
            train_preds_views.append(predict(teacher, Xv, device=device))
        y_soft = np.stack(train_preds_views, axis=0).mean(axis=0)

        # Student is time-agnostic
        X_train_student = _featurize(train, include_time=False)
        X_test_student = _featurize(test, include_time=False)

        student_cfg = MLPRegressorConfig(
            in_dim=int(X_train_student.shape[1]),
            out_dim=int(y_train.shape[1]),
            hidden=int(args.hidden),
            depth=int(args.depth),
            dropout=float(args.dropout),
        )

        student, student_metrics = train_regressor_distill(
            X_train_student[idx_tr],
            y_soft[idx_tr],
            y_train[idx_tr],
            X_train_student[idx_val],
            y_soft[idx_val],
            y_train[idx_val],
            cfg=student_cfg,
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

        # Student gap: evaluate student on test views, but student doesn't take time.
        preds_student_views: List[np.ndarray] = []
        for j in range(int(args.k_test)):
            if j == 0:
                view = test
            else:
                view = [
                    sym.sample(ex, seed=_view_seed(int(args.seed) + 123, i, j))
                    for i, ex in enumerate(test)
                ]
            Xv = _featurize(view, include_time=False)
            preds_student_views.append(predict(student, Xv, device=device))
        student_gap = warrant_gap_regression(np.stack(preds_student_views, axis=0))

        student_roll = _rollout_mse(
            student,
            examples=test,
            trajs=trajs_test,
            include_time=False,
            t_arr=t_arr,
            horizon=int(args.rollout_horizon),
            device=device,
        )

        gap_improves = student_gap["gap_mse"] <= 0.8 * teacher_gap["gap_mse"]
        mse_ok = student_mse_test <= 1.5 * mse_test
        verdict = "MAKE OK" if (gap_improves and mse_ok) else "BREAK / INCONCLUSIVE X"

        results = {
            "world": {
                "adapter": "ode_npz",
                "adapter_config": asdict(adapter_cfg),
                **{
                    k: world["meta"].get(k)
                    for k in ["dt", "n_train", "n_test"]
                    if k in world.get("meta", {})
                },
            },
            "symmetry": {
                "name": "time_origin_shift",
                "max_shift": float(args.max_shift),
            },
            "teacher": {
                "include_time": include_time_teacher,
                "metrics": {
                    **teacher_metrics,
                    "mse_test": mse_test,
                    "rollout_mse": teacher_roll,
                    **teacher_gap,
                },
            },
            "student": {
                "include_time": False,
                "metrics": {
                    **student_metrics,
                    "mse_test": student_mse_test,
                    "rollout_mse": student_roll,
                    **student_gap,
                },
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
