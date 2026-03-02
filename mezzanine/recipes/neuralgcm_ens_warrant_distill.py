from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from ..core.config import deep_update, load_config
from ..core.deterministic import deterministic_subsample_indices
from ..pipelines.regression_distill import (
    MLPRegHeadConfig,
    mse,
    predict_regression,
    train_soft_regression_head,
    warrant_gap_l2_from_views,
)
from ..symmetries.ens_member import EnsembleMemberSymmetry, EnsembleMemberSymmetryConfig
from ..symmetries.field_codec import FieldCodecConfig, FieldCodecSymmetry
from .recipe_base import Recipe


def _device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
            return "mps"
    except Exception:
        pass
    return "cpu"


def _is_gcs_zarr(path: str) -> bool:
    return str(path).startswith("gs://")


def _require_neuralgcm_zarr_deps(*, members_zarr: str, mean_zarr: str) -> None:
    need_gcsfs = _is_gcs_zarr(members_zarr) or _is_gcs_zarr(mean_zarr)
    missing: list[str] = []

    if importlib.util.find_spec("xarray") is None:
        missing.append("xarray")
    if importlib.util.find_spec("zarr") is None:
        missing.append("zarr")
    if need_gcsfs and importlib.util.find_spec("gcsfs") is None:
        missing.append("gcsfs")

    if missing:  # pragma: no cover
        hint = "pip install xarray zarr" + (" gcsfs" if need_gcsfs else "")
        raise ImportError(
            "Missing optional dependencies for `neuralgcm_ens_warrant_distill`: "
            + ", ".join(missing)
            + f". Install with: {hint}"
        )


def _open_zarr_compat(
    xr: Any, path: str, *, storage_options: Dict[str, Any] | None
) -> Any:
    kwargs: Dict[str, Any] = {"chunks": None}
    if storage_options is not None:
        kwargs["storage_options"] = storage_options
    try:
        return xr.open_zarr(path, decode_timedelta=True, **kwargs)
    except TypeError:
        return xr.open_zarr(path, **kwargs)


def _seed32(*parts: Any) -> int:
    x = 2166136261
    for p in parts:
        b = str(p).encode("utf-8")
        for bb in b:
            x ^= bb
            x = (x * 16777619) & 0xFFFFFFFF
    return int(x)


def _fit_mu_sig(Y: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    mu = Y.mean(axis=0).astype(np.float32)
    sig = np.maximum(Y.std(axis=0).astype(np.float32), eps)
    return mu, sig


def _std(X: np.ndarray, mu: np.ndarray, sig: np.ndarray) -> np.ndarray:
    return ((X - mu) / sig).astype(np.float32)


def _stack_mean(
    ds: Any,
    variables: Sequence[str],
    *,
    spatial_order: Sequence[str] = ("level", "longitude", "latitude"),
    level_idx: int | None = None,
) -> np.ndarray:
    arrs: list[np.ndarray] = []
    for v in variables:
        da = ds[v]
        if level_idx is not None and "level" in getattr(da, "dims", ()):
            da = da.isel(level=int(level_idx))
        dims = [d for d in spatial_order if d in getattr(da, "dims", ())]
        if tuple(dims) != tuple(getattr(da, "dims", ())):
            da = da.transpose(*dims)
        arrs.append(np.asarray(da.values, dtype=np.float32).reshape(-1))
    return np.stack(arrs, axis=1)  # [P,V]


def _stack_members(
    ds: Any,
    variables: Sequence[str],
    *,
    mems: List[int] | None = None,
    spatial_order: Sequence[str] = ("level", "longitude", "latitude"),
    level_idx: int | None = None,
) -> np.ndarray:
    ds2 = ds[variables] if mems is None else ds[variables].isel(realization=mems)
    mats: list[np.ndarray] = []
    for v in variables:
        da = ds2[v]
        if level_idx is not None and "level" in getattr(da, "dims", ()):
            da = da.isel(level=int(level_idx))
        dims = [d for d in spatial_order if d in getattr(da, "dims", ())]
        da = da.transpose("realization", *dims)
        a = np.asarray(da.values, dtype=np.float32)
        mats.append(a.reshape(a.shape[0], -1))  # [K,P]
    return np.stack(mats, axis=-1)  # [K,P,V]


def _subsample_points(
    Xv: np.ndarray, Y: np.ndarray, n: int, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    if n <= 0 or Xv.shape[1] <= n:
        return Xv, Y
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(Xv.shape[1], size=int(n), replace=False)
    return Xv[:, idx, :], Y[idx, :]


def _choose_level_idx(ds: Any, level_hpa: float | None) -> int | None:
    if level_hpa is None:
        return None
    if ("level" not in getattr(ds, "coords", {})) and (
        "level" not in getattr(ds, "dims", {})
    ):
        return None
    lv = np.asarray(ds["level"].values, dtype=np.float64)
    target = float(level_hpa)
    if np.nanmax(lv) > 2000:
        target *= 100.0
    return int(np.argmin(np.abs(lv - target)))


def _plot_diagnostics(
    out_dir: Path, *, metrics: Dict[str, Dict[str, float]], verdict: Dict[str, Any]
) -> Path:
    import matplotlib.pyplot as plt

    methods = list(metrics.keys())
    mse_view = [float(metrics[m]["mse_to_teacher_norm_view"]) for m in methods]
    gap = [float(metrics[m]["gap_l2_to_mean"]) for m in methods]

    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    ax1.bar(methods, mse_view)
    ax1.set_title("MSE to teacher (normalized)")
    ax1.set_ylabel("mse")
    ax1.tick_params(axis="x", rotation=20)

    ax2.bar(methods, gap)
    ax2.set_title("Warrant gap (L2 to mean)")
    ax2.set_ylabel("gap")
    ax2.tick_params(axis="x", rotation=20)

    v = verdict.get("verdict", False)
    fig.suptitle(
        f"neuralgcm_ens_warrant_distill — {'MAKE OK' if v else 'BREAK / INCONCLUSIVE X'}"
    )
    fig.tight_layout()

    path = out_dir / "diagnostics.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


class NeuralGCMEnsWarrantDistillRecipe(Recipe):
    NAME = "neuralgcm_ens_warrant_distill"
    DESCRIPTION = "NeuralGCM-ENS: compose symmetries (ens_member × field_codec), close regression warrant gap, emit hero GIF."

    def run(self, argv: list[str]) -> Dict[str, Any]:
        p = argparse.ArgumentParser(prog=self.NAME)
        self.add_common_args(p)

        p.add_argument(
            "--members_zarr",
            type=str,
            default="gs://weatherbench2/datasets/neuralgcm_ens/2020-64x32_equiangular_conservative.zarr",
        )
        p.add_argument(
            "--mean_zarr",
            type=str,
            default="gs://weatherbench2/datasets/neuralgcm_ens/2020-64x32_equiangular_conservative_mean.zarr",
        )
        p.add_argument("--lead_hours", type=int, default=24)
        p.add_argument(
            "--variables",
            type=str,
            default=(
                "u_component_of_wind,v_component_of_wind,geopotential,temperature,specific_humidity,"
                "specific_cloud_ice_water_content,specific_cloud_liquid_water_content"
            ),
        )

        # Symmetry (member)
        p.add_argument("--num_members", type=int, default=50)
        p.add_argument("--canonical_member", type=int, default=0)
        p.add_argument("--k_train", type=int, default=8)
        p.add_argument("--k_test", type=int, default=16)
        p.add_argument("--train_on_views", action="store_true")

        # Symmetry (codec)
        p.add_argument("--use_codec", action="store_true")
        p.add_argument("--codec_clip", type=float, default=6.0)
        p.add_argument("--codec_fp16_prob", type=float, default=0.5)
        p.add_argument("--codec_bits_min", type=int, default=8)
        p.add_argument("--codec_bits_max", type=int, default=10)
        p.add_argument("--codec_noise_std_min", type=float, default=0.00)
        p.add_argument("--codec_noise_std_max", type=float, default=0.05)

        # Data sampling
        p.add_argument("--n_train_times", type=int, default=32)
        p.add_argument("--n_val_times", type=int, default=4)
        p.add_argument("--n_test_times", type=int, default=8)
        p.add_argument("--max_points_per_time", type=int, default=40000)

        # Head
        p.add_argument("--steps", type=int, default=6000)
        p.add_argument("--batch", type=int, default=8192)
        p.add_argument("--lr", type=float, default=2e-4)
        p.add_argument("--wd", type=float, default=1e-4)
        p.add_argument("--hidden", type=int, default=256)
        p.add_argument("--depth", type=int, default=3)
        p.add_argument("--dropout", type=float, default=0.0)
        p.add_argument("--grad_clip", type=float, default=1.0)
        p.add_argument("--eval_every", type=int, default=200)

        # Hero (optional)
        p.add_argument("--hero", action="store_true")
        p.add_argument("--hero_time_index", type=int, default=-1)
        p.add_argument("--hero_var", type=str, default="temperature")
        p.add_argument("--hero_level_hpa", type=float, default=500.0)
        p.add_argument("--hero_leads", type=str, default="6,12,24,48,72")
        p.add_argument("--hero_k", type=int, default=16)
        p.add_argument("--hero_fps", type=int, default=2)

        args = p.parse_args(argv)

        # Apply config file defaults BEFORE seeding/logger/cache creation
        file_cfg = load_config(getattr(args, "config", None))
        merged_cfg = deep_update(file_cfg, self.config)
        self.apply_config_defaults(p, args, merged_cfg)

        ctx = self.build_context(args)
        out_dir = ctx.out_dir

        members_zarr = str(args.members_zarr)
        mean_zarr = str(args.mean_zarr)
        _require_neuralgcm_zarr_deps(members_zarr=members_zarr, mean_zarr=mean_zarr)

        import xarray as xr

        t0 = time.time()
        print(f"[io] opening members_zarr={members_zarr}")
        members = _open_zarr_compat(
            xr,
            members_zarr,
            storage_options={"token": "anon"} if _is_gcs_zarr(members_zarr) else None,
        )
        print(f"[io] opening mean_zarr={mean_zarr}")
        mean = _open_zarr_compat(
            xr,
            mean_zarr,
            storage_options={"token": "anon"} if _is_gcs_zarr(mean_zarr) else None,
        )
        print(f"[io] opened zarr stores in {time.time() - t0:.1f}s")

        import torch

        device = _device()
        print(f"[io] device={device}")

        variables = [v.strip() for v in str(args.variables).split(",") if v.strip()]
        V = len(variables)
        if V == 0:
            raise ValueError("No variables provided.")

        # Lead index
        lead = np.timedelta64(int(args.lead_hours), "h")
        deltas = mean["prediction_timedelta"].values
        lead_idx = int(np.argmin(np.abs(deltas - lead)))
        lead_sel = deltas[lead_idx]

        # Symmetry objects
        member_sym = EnsembleMemberSymmetry(
            EnsembleMemberSymmetryConfig(
                num_members=int(args.num_members),
                without_replacement=True,
            )
        )
        codec_sym = FieldCodecSymmetry(
            FieldCodecConfig(
                clip=float(args.codec_clip),
                fp16_prob=float(args.codec_fp16_prob),
                quant_bits_min=int(args.codec_bits_min),
                quant_bits_max=int(args.codec_bits_max),
                noise_std_min=float(args.codec_noise_std_min),
                noise_std_max=float(args.codec_noise_std_max),
            )
        )

        # Splits
        n_total = int(mean.sizes["time"])
        n_train = int(args.n_train_times)
        n_val = int(args.n_val_times)
        n_test = int(args.n_test_times)
        idx = deterministic_subsample_indices(
            n_total, n_train + n_val + n_test, seed=int(ctx.seed)
        )
        train_idx = idx[:n_train]
        val_idx = idx[n_train : n_train + n_val]
        test_idx = idx[n_train + n_val :]

        def sample_mems(K: int, t: int, tag: str) -> List[int]:
            canonical = int(args.canonical_member)
            chosen = [canonical]
            used = {canonical}

            base_seed = _seed32(ctx.seed, tag, t, K)
            cands = member_sym.batch(None, k=max(16, K * 3), seed=base_seed)
            for r in cands:
                if len(chosen) >= K:
                    break
                r = int(r)
                if r not in used:
                    chosen.append(r)
                    used.add(r)

            bump = 1
            while len(chosen) < K:
                r = int(member_sym.sample(None, seed=base_seed + bump))
                bump += 1
                if r not in used:
                    chosen.append(r)
                    used.add(r)
            return chosen

        def load_block(
            t: int, K: int, *, level_idx: int | None
        ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
            mems = sample_mems(K, t, "mems")
            print(f"[io] t={t} loading mean + {len(mems)} members")
            t0 = time.time()

            ds_mean = (
                mean[variables].isel(time=int(t), prediction_timedelta=lead_idx).load()
            )
            ds_mem = (
                members[variables]
                .isel(
                    time=int(t),
                    prediction_timedelta=lead_idx,
                    realization=mems,
                )
                .load()
            )

            Y = _stack_mean(ds_mean, variables, level_idx=level_idx)
            Xv = _stack_members(ds_mem, variables, mems=None, level_idx=level_idx)
            Xv, Y = _subsample_points(
                Xv, Y, int(args.max_points_per_time), seed=_seed32(ctx.seed, "sub", t)
            )
            print(f"[io] t={t} loaded in {time.time() - t0:.1f}s")
            return Xv, Y, mems

        # --- Build train (plain vs sym) ---
        X_plain, Y_plain = [], []
        X_sym, Y_sym = [], []

        for t in train_idx:
            Xv, Y, _mems = load_block(int(t), int(args.k_train), level_idx=None)
            X_plain.append(Xv[0])
            Y_plain.append(Y)

            if bool(args.train_on_views):
                K, P, _ = Xv.shape
                X_sym.append(Xv.reshape(K * P, V))
                Y_sym.append(np.repeat(Y[None, :, :], K, axis=0).reshape(K * P, V))
            else:
                X_sym.append(Xv[0])
                Y_sym.append(Y)

        X_train_plain = np.concatenate(X_plain, axis=0)
        Y_train_plain = np.concatenate(Y_plain, axis=0)
        X_train_sym = np.concatenate(X_sym, axis=0)
        Y_train_sym = np.concatenate(Y_sym, axis=0)

        mu, sig = _fit_mu_sig(Y_train_plain)

        Xn_train_plain = _std(X_train_plain, mu, sig)
        Yn_train_plain = _std(Y_train_plain, mu, sig)
        Xn_train_sym = _std(X_train_sym, mu, sig)
        Yn_train_sym = _std(Y_train_sym, mu, sig)

        # Codec only in sym training => composed symmetry = member × codec
        if bool(args.use_codec):
            chunk = 200000
            for i in range(0, int(Xn_train_sym.shape[0]), chunk):
                Xn_train_sym[i : i + chunk] = codec_sym.sample(
                    Xn_train_sym[i : i + chunk],
                    seed=_seed32(ctx.seed, "codec_train", i),
                )

        # --- Val (optional): canonical-only ---
        Xn_val_plain = Yn_val_plain = None
        Xn_val_sym = Yn_val_sym = None
        if len(val_idx) > 0:
            X_plain, Y_plain = [], []
            X_sym, Y_sym = [], []
            for t in val_idx:
                Xv, Y, _mems = load_block(int(t), int(args.k_train), level_idx=None)
                X_plain.append(Xv[0])
                Y_plain.append(Y)
                if bool(args.train_on_views):
                    K, P, _ = Xv.shape
                    X_sym.append(Xv.reshape(K * P, V))
                    Y_sym.append(np.repeat(Y[None, :, :], K, axis=0).reshape(K * P, V))
                else:
                    X_sym.append(Xv[0])
                    Y_sym.append(Y)

            X_val_plain = np.concatenate(X_plain, axis=0)
            Y_val_plain = np.concatenate(Y_plain, axis=0)
            X_val_sym = np.concatenate(X_sym, axis=0)
            Y_val_sym = np.concatenate(Y_sym, axis=0)

            Xn_val_plain = _std(X_val_plain, mu, sig)
            Yn_val_plain = _std(Y_val_plain, mu, sig)
            Xn_val_sym = _std(X_val_sym, mu, sig)
            Yn_val_sym = _std(Y_val_sym, mu, sig)

            if bool(args.use_codec):
                Xn_val_sym = codec_sym.sample(
                    Xn_val_sym, seed=_seed32(ctx.seed, "codec_val")
                )

        head_cfg = MLPRegHeadConfig(
            in_dim=int(V),
            out_dim=int(V),
            hidden=int(args.hidden),
            depth=int(args.depth),
            dropout=float(args.dropout),
            residual=True,
        )

        print("[train] plain head (no symmetry)...")
        plain_head, plain_train_metrics, _ = train_soft_regression_head(
            Xn_train_plain,
            Yn_train_plain,
            X_val=Xn_val_plain,
            Y_val=Yn_val_plain,
            steps=int(args.steps),
            batch_size=int(args.batch),
            lr=float(args.lr),
            wd=float(args.wd),
            grad_clip=float(args.grad_clip),
            eval_every=int(args.eval_every),
            seed=int(ctx.seed),
            device=device,
            cfg=head_cfg,
        )

        print("[train] symmetry head (member × codec views)...")
        sym_head, sym_train_metrics, _ = train_soft_regression_head(
            Xn_train_sym,
            Yn_train_sym,
            X_val=Xn_val_sym,
            Y_val=Yn_val_sym,
            steps=int(args.steps),
            batch_size=int(args.batch),
            lr=float(args.lr),
            wd=float(args.wd),
            grad_clip=float(args.grad_clip),
            eval_every=int(args.eval_every),
            seed=int(ctx.seed) + 17,
            device=device,
            cfg=head_cfg,
        )

        # --- Evaluate under composed symmetry ---
        def apply_codec_per_view(Xn: np.ndarray, t: int) -> np.ndarray:
            if not bool(args.use_codec):
                return Xn
            Xn2 = Xn.copy()
            for j in range(int(Xn2.shape[0])):
                Xn2[j] = codec_sym.sample(
                    Xn2[j], seed=_seed32(ctx.seed, "codec_test", t, j)
                )
            return Xn2

        def pred_views(method: str, Xn: np.ndarray, Yn: np.ndarray) -> np.ndarray:
            K, P, _ = Xn.shape
            if method == "base":
                return Xn
            if method == "kmean":
                m = Xn.mean(axis=0, keepdims=True)
                return np.repeat(m, K, axis=0)
            if method == "plain":
                return predict_regression(
                    plain_head, Xn.reshape(K * P, V), device=device
                ).reshape(K, P, V)
            if method == "sym":
                return predict_regression(
                    sym_head, Xn.reshape(K * P, V), device=device
                ).reshape(K, P, V)
            if method == "teacher":
                m = Yn[None, :, :]
                return np.repeat(m, K, axis=0)
            raise ValueError(method)

        methods = ["base", "plain", "sym", "kmean", "teacher"]
        agg: Dict[str, Dict[str, float]] = {
            m: {"mse_view": 0.0, "mse_canon": 0.0, "gap": 0.0, "pair": 0.0}
            for m in methods
        }
        n_blocks = 0

        for t in test_idx:
            Xv, Y, _mems = load_block(int(t), int(args.k_test), level_idx=None)
            Xn = _std(Xv, mu, sig)
            Yn = _std(Y, mu, sig)
            Xn = apply_codec_per_view(Xn, int(t))

            for m in methods:
                Ypred = pred_views(m, Xn, Yn)  # [K,P,V]
                mse_view = float(
                    np.mean([mse(Ypred[j], Yn) for j in range(int(Ypred.shape[0]))])
                )
                mse_canon = float(mse(Ypred[0], Yn))
                gaps = warrant_gap_l2_from_views(
                    np.transpose(Ypred, (1, 0, 2))
                )  # [P,K,V]
                agg[m]["mse_view"] += mse_view
                agg[m]["mse_canon"] += mse_canon
                agg[m]["gap"] += float(gaps["mean_l2_to_mean"])
                agg[m]["pair"] += float(gaps["mean_pairwise_l2"])

            n_blocks += 1

        for m in methods:
            for k in agg[m]:
                agg[m][k] /= max(1, n_blocks)

        make_break = {
            "gap_reduced_vs_base": bool(agg["sym"]["gap"] < agg["base"]["gap"]),
            "gap_reduced_vs_plain": bool(agg["sym"]["gap"] < agg["plain"]["gap"]),
            "mse_improved_vs_base": bool(
                agg["sym"]["mse_view"] < agg["base"]["mse_view"]
            ),
            "mse_improved_vs_plain": bool(
                agg["sym"]["mse_view"] < agg["plain"]["mse_view"]
            ),
            "verdict": bool(
                (agg["sym"]["gap"] < agg["base"]["gap"])
                and (agg["sym"]["mse_view"] <= agg["plain"]["mse_view"] * 1.01)
            ),
        }

        out: Dict[str, Any] = {
            "exp": self.NAME,
            "members_zarr": members_zarr,
            "mean_zarr": mean_zarr,
            "lead_hours_requested": int(args.lead_hours),
            "lead_timedelta_selected": str(lead_sel),
            "variables": variables,
            "symmetry": ["ens_member"]
            + (["field_codec"] if bool(args.use_codec) else []),
            "symmetry_cfg": {
                "ens_member": asdict(
                    EnsembleMemberSymmetryConfig(
                        num_members=int(args.num_members),
                        without_replacement=True,
                    )
                ),
                "field_codec": asdict(codec_sym.cfg) if bool(args.use_codec) else None,
            },
            "k_train": int(args.k_train),
            "k_test": int(args.k_test),
            "canonical_member": int(args.canonical_member),
            "train_on_views": bool(args.train_on_views),
            "splits": {
                "n_train_times": n_train,
                "n_val_times": n_val,
                "n_test_times": n_test,
            },
            "train_rows": int(X_train_sym.shape[0]),
            "val_rows": int(0 if Xn_val_sym is None else Xn_val_sym.shape[0]),
            "metrics": {
                "base": {
                    "mse_to_teacher_norm_view": agg["base"]["mse_view"],
                    "mse_to_teacher_norm_canon": agg["base"]["mse_canon"],
                    "gap_l2_to_mean": agg["base"]["gap"],
                    "pairwise_l2": agg["base"]["pair"],
                    "cost_members": 1,
                },
                "plain": {
                    "mse_to_teacher_norm_view": agg["plain"]["mse_view"],
                    "mse_to_teacher_norm_canon": agg["plain"]["mse_canon"],
                    "gap_l2_to_mean": agg["plain"]["gap"],
                    "pairwise_l2": agg["plain"]["pair"],
                    "cost_members": 1,
                },
                "sym": {
                    "mse_to_teacher_norm_view": agg["sym"]["mse_view"],
                    "mse_to_teacher_norm_canon": agg["sym"]["mse_canon"],
                    "gap_l2_to_mean": agg["sym"]["gap"],
                    "pairwise_l2": agg["sym"]["pair"],
                    "cost_members": 1,
                },
                "kmean": {
                    "mse_to_teacher_norm_view": agg["kmean"]["mse_view"],
                    "mse_to_teacher_norm_canon": agg["kmean"]["mse_canon"],
                    "gap_l2_to_mean": agg["kmean"]["gap"],
                    "pairwise_l2": agg["kmean"]["pair"],
                    "cost_members": int(args.k_test),
                },
                "teacher": {
                    "mse_to_teacher_norm_view": 0.0,
                    "mse_to_teacher_norm_canon": 0.0,
                    "gap_l2_to_mean": 0.0,
                    "pairwise_l2": 0.0,
                    "cost_members": int(args.num_members),
                },
                "train_plain": plain_train_metrics,
                "train_sym": sym_train_metrics,
            },
            "make_break": make_break,
            "head_cfg": asdict(head_cfg),
        }

        # --- Artifacts ---
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "results.json").write_text(
            json.dumps(out, indent=2, sort_keys=True), encoding="utf-8"
        )
        # Back-compat with older patch scripts.
        (out_dir / "metrics.json").write_text(
            json.dumps(out, indent=2, sort_keys=True), encoding="utf-8"
        )

        plot_metrics = {
            k: out["metrics"][k] for k in ["base", "plain", "sym", "kmean", "teacher"]
        }
        _plot_diagnostics(out_dir, metrics=plot_metrics, verdict=make_break)

        torch.save(
            {
                "state_dict": plain_head.state_dict(),
                "mu": mu,
                "sig": sig,
                "variables": variables,
                "cfg": asdict(head_cfg),
            },
            out_dir / "head_plain.pt",
        )
        torch.save(
            {
                "state_dict": sym_head.state_dict(),
                "mu": mu,
                "sig": sig,
                "variables": variables,
                "cfg": asdict(head_cfg),
            },
            out_dir / "head_sym.pt",
        )

        if bool(args.hero):
            self._hero(
                out_dir=out_dir,
                members=members,
                mean=mean,
                variables=variables,
                mu=mu,
                sig=sig,
                member_sym=member_sym,
                codec_sym=codec_sym if bool(args.use_codec) else None,
                plain_head=plain_head,
                sym_head=sym_head,
                device=device,
                hero_var=str(args.hero_var),
                hero_time_index=int(args.hero_time_index),
                hero_level_hpa=float(args.hero_level_hpa)
                if args.hero_level_hpa is not None
                else None,
                hero_leads=str(args.hero_leads),
                hero_k=int(args.hero_k),
                hero_fps=int(args.hero_fps),
                seed=int(ctx.seed),
                canonical_member=int(args.canonical_member),
            )

        return out

    def _hero(
        self,
        *,
        out_dir: Path,
        members: Any,
        mean: Any,
        variables: Sequence[str],
        mu: np.ndarray,
        sig: np.ndarray,
        member_sym: EnsembleMemberSymmetry,
        codec_sym: FieldCodecSymmetry | None,
        plain_head: Any,
        sym_head: Any,
        device: str,
        hero_var: str,
        hero_time_index: int,
        hero_level_hpa: float | None,
        hero_leads: str,
        hero_k: int,
        hero_fps: int,
        seed: int,
        canonical_member: int,
    ) -> None:
        import matplotlib.pyplot as plt
        from PIL import Image

        def unnorm(Yn: np.ndarray) -> np.ndarray:
            return (Yn * sig[None, :] + mu[None, :]).astype(np.float32)

        if hero_var not in variables:
            print(f"[hero] {hero_var} not in variables; using {variables[0]}")
            hero_var = variables[0]
        v_idx = list(variables).index(hero_var)

        level_idx = _choose_level_idx(members, hero_level_hpa)

        t_index = (
            int(mean.sizes["time"] // 2)
            if int(hero_time_index) < 0
            else int(hero_time_index)
        )
        t_index = max(0, min(t_index, int(mean.sizes["time"]) - 1))

        leads = [int(x) for x in hero_leads.split(",") if x.strip()]
        if not leads:
            leads = [24]
        deltas = mean["prediction_timedelta"].values
        lead_idxs = [
            (h, int(np.argmin(np.abs(deltas - np.timedelta64(int(h), "h")))))
            for h in leads
        ]

        def sample_mems(K: int) -> List[int]:
            chosen = [int(canonical_member)]
            used = {int(canonical_member)}
            base_seed = _seed32(seed, "hero_mems", t_index, K)
            cands = member_sym.batch(None, k=max(16, K * 3), seed=base_seed)
            for r in cands:
                if len(chosen) >= K:
                    break
                r = int(r)
                if r not in used:
                    chosen.append(r)
                    used.add(r)
            bump = 1
            while len(chosen) < K:
                r = int(member_sym.sample(None, seed=base_seed + bump))
                bump += 1
                if r not in used:
                    chosen.append(r)
                    used.add(r)
            return chosen

        def load_block_for_lead(lead_idx: int, K: int) -> Tuple[np.ndarray, np.ndarray]:
            mems = sample_mems(K)
            ds_mean = (
                mean[variables]
                .isel(time=int(t_index), prediction_timedelta=int(lead_idx))
                .load()
            )
            ds_mem = (
                members[variables]
                .isel(
                    time=int(t_index),
                    prediction_timedelta=int(lead_idx),
                    realization=mems,
                )
                .load()
            )
            Y = _stack_mean(ds_mean, variables, level_idx=level_idx)
            Xv = _stack_members(ds_mem, variables, mems=None, level_idx=level_idx)
            return Xv, Y

        frames: list[Image.Image] = []
        lead_rows: list[dict[str, Any]] = []

        for h, li in lead_idxs:
            Xv, Y = load_block_for_lead(int(li), int(hero_k))
            Xn = _std(Xv, mu, sig)
            Yn = _std(Y, mu, sig)

            if codec_sym is not None:
                Xn2 = Xn.copy()
                for j in range(int(Xn2.shape[0])):
                    Xn2[j] = codec_sym.sample(
                        Xn2[j], seed=_seed32(seed, "hero_codec", t_index, h, j)
                    )
                Xn = Xn2

            K, P, V = Xn.shape

            Yb = Xn  # base: raw member
            Yp = predict_regression(
                plain_head, Xn.reshape(K * P, V), device=device
            ).reshape(K, P, V)
            Ys = predict_regression(
                sym_head, Xn.reshape(K * P, V), device=device
            ).reshape(K, P, V)

            if "longitude" in getattr(mean, "coords", {}) and "latitude" in getattr(
                mean, "coords", {}
            ):
                W = int(mean.sizes.get("longitude", 1))
                H = int(mean.sizes.get("latitude", 1))
            else:
                W = int(np.sqrt(P))
                H = int(np.sqrt(P))

            T = unnorm(Yn)[:, v_idx].reshape(W, H).T
            B = unnorm(Yb[0])[:, v_idx].reshape(W, H).T
            Pld = unnorm(Yp[0])[:, v_idx].reshape(W, H).T
            S = unnorm(Ys[0])[:, v_idx].reshape(W, H).T

            eB = np.abs(B - T)
            eP = np.abs(Pld - T)
            eS = np.abs(S - T)

            gB = np.std(unnorm(Yb)[:, :, v_idx], axis=0).reshape(W, H).T
            gP = np.std(unnorm(Yp)[:, :, v_idx], axis=0).reshape(W, H).T
            gS = np.std(unnorm(Ys)[:, :, v_idx], axis=0).reshape(W, H).T

            mseB = float(np.mean((B - T) ** 2))
            mseP = float(np.mean((Pld - T) ** 2))
            mseS = float(np.mean((S - T) ** 2))
            gapB = float(np.mean(gB))
            gapP = float(np.mean(gP))
            gapS = float(np.mean(gS))

            lead_rows.append(
                {
                    "lead_h": int(h),
                    "mse_base": mseB,
                    "mse_plain": mseP,
                    "mse_sym": mseS,
                    "gap_base": gapB,
                    "gap_plain": gapP,
                    "gap_sym": gapS,
                }
            )

            fig = plt.figure(figsize=(14, 8))
            gs = fig.add_gridspec(3, 4)

            def put(ax, img, title: str) -> None:
                im = ax.imshow(img)
                ax.set_title(title)
                ax.axis("off")
                fig.colorbar(im, ax=ax, fraction=0.046)

            put(fig.add_subplot(gs[0, 0]), T, f"Teacher mean\n{hero_var}")
            put(fig.add_subplot(gs[0, 1]), B, "Base (1 member)")
            put(fig.add_subplot(gs[0, 2]), Pld, "Plain distilled")
            put(fig.add_subplot(gs[0, 3]), S, "Mezzanine (sym)")

            put(fig.add_subplot(gs[1, 0]), eB, f"|Base-Teacher|\nMSE={mseB:.3g}")
            put(fig.add_subplot(gs[1, 1]), eP, f"|Plain-Teacher|\nMSE={mseP:.3g}")
            put(fig.add_subplot(gs[1, 2]), eS, f"|Sym-Teacher|\nMSE={mseS:.3g}")
            ax = fig.add_subplot(gs[1, 3])
            ax.axis("off")
            ax.text(
                0,
                0.95,
                f"t={t_index}\nlead={h}h\nK={K} views\ncodec={'on' if codec_sym is not None else 'off'}",
                va="top",
                fontsize=12,
            )
            ax.text(
                0,
                0.55,
                f"Gap(mean std across views)\nBase:  {gapB:.3g}\nPlain: {gapP:.3g}\nSym:   {gapS:.3g}",
                va="top",
                fontsize=12,
            )

            put(fig.add_subplot(gs[2, 0]), gB, f"Gap map (Base)\nmean std={gapB:.3g}")
            put(fig.add_subplot(gs[2, 1]), gP, f"Gap map (Plain)\nmean std={gapP:.3g}")
            put(fig.add_subplot(gs[2, 2]), gS, f"Gap map (Sym)\nmean std={gapS:.3g}")
            fig.add_subplot(gs[2, 3]).axis("off")

            fig.suptitle(
                "Mezzanine symmetry distillation closes the warrant gap", fontsize=16
            )
            fig.tight_layout()

            frame_path = out_dir / f"hero_lead_{int(h):03d}h.png"
            fig.savefig(frame_path, dpi=140)
            plt.close(fig)
            frames.append(Image.open(frame_path))

        if frames:
            gif_path = out_dir / "hero.gif"
            duration_ms = int(1000 / max(1, int(hero_fps)))
            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:],
                duration=duration_ms,
                loop=0,
            )
            print("[hero] wrote:", gif_path)

        if lead_rows:
            csv_path = out_dir / "hero_lead_metrics.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                w = csv.DictWriter(handle, fieldnames=list(lead_rows[0].keys()))
                w.writeheader()
                w.writerows(lead_rows)
            print("[hero] wrote:", csv_path)

            leads2 = [r["lead_h"] for r in lead_rows]
            for key, outp, title in [
                (
                    "mse",
                    out_dir / "hero_mse_vs_lead.png",
                    "MSE vs Teacher (hero var-level)",
                ),
                (
                    "gap",
                    out_dir / "hero_gap_vs_lead.png",
                    "Warrant gap proxy (mean std across views)",
                ),
            ]:
                fig = plt.figure(figsize=(8, 4))
                if key == "mse":
                    plt.plot(
                        leads2,
                        [r["mse_base"] for r in lead_rows],
                        marker="o",
                        label="base",
                    )
                    plt.plot(
                        leads2,
                        [r["mse_plain"] for r in lead_rows],
                        marker="o",
                        label="plain",
                    )
                    plt.plot(
                        leads2,
                        [r["mse_sym"] for r in lead_rows],
                        marker="o",
                        label="sym",
                    )
                    plt.ylabel("mse")
                else:
                    plt.plot(
                        leads2,
                        [r["gap_base"] for r in lead_rows],
                        marker="o",
                        label="base",
                    )
                    plt.plot(
                        leads2,
                        [r["gap_plain"] for r in lead_rows],
                        marker="o",
                        label="plain",
                    )
                    plt.plot(
                        leads2,
                        [r["gap_sym"] for r in lead_rows],
                        marker="o",
                        label="sym",
                    )
                    plt.ylabel("gap")
                plt.xlabel("Lead (hours)")
                plt.title(title)
                plt.legend()
                plt.tight_layout()
                fig.savefig(outp, dpi=160)
                plt.close(fig)
                print("[hero] wrote:", outp)
