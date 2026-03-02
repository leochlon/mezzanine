from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from typing import Any, Dict, List

import numpy as np

from ..encoders.qg import (
    QGEEC2Encoder,
    QGEEC2EncoderConfig,
    QGFlattenEncoder,
    QGFlattenEncoderConfig,
)
from ..symmetries.qg import (
    QGCoordNoiseConfig,
    QGCoordNoiseSymmetry,
    QGPermutationSymmetry,
    QGReflectionSymmetry,
    QGSO2RotateConfig,
    QGSO2RotateSymmetry,
)
from ..pipelines.text_distill import (
    MLPHeadConfig,
    accuracy,
    predict_proba,
    train_hard_label_head,
    train_soft_label_head,
    warrant_gap_from_views,
)
from ..viz.qg_gif import JetGifConfig, write_jet_nuisance_gif
from ..worlds.qg_jets import QGJetsAdapter, QGJetsAdapterConfig
from .recipe_base import Recipe


def view_seed(global_seed: int, i: int, j: int, k: int = 0) -> int:
    """Deterministic per-example per-view per-symmetry seed."""
    return int((global_seed * 1000003 + i * 9176 + j * 7919 + k * 104729) % (2**32 - 1))


def build_views(
    xs: List[Dict[str, Any]],
    *,
    symmetries: List[Any],
    seed: int,
    K: int,
) -> List[List[Dict[str, Any]]]:
    """Return list of K view-lists, each view-list is examples for all items."""
    views: List[List[Dict[str, Any]]] = [list(xs)]
    if K <= 1:
        return views
    for j in range(1, K):
        vj: List[Dict[str, Any]] = []
        for i, x in enumerate(xs):
            xj = x
            for k, sym in enumerate(symmetries):
                xj = sym.sample(xj, seed=view_seed(seed, i, j, k))
            vj.append(xj)
        views.append(vj)
    return views


def _cached_encode(
    *,
    cache,
    world_fp: str,
    enc_fp: str,
    split: str,
    tag: str,
    extra: Dict[str, Any],
    encoder,
    batch: List[Any],
) -> np.ndarray:
    if cache is None:
        return encoder.encode(batch)
    key = cache.make_key(
        world_fingerprint=world_fp,
        encoder_fingerprint=enc_fp,
        split=split,
        tag=tag,
        extra=extra,
    )
    got = cache.get(key)
    if got is not None:
        arr, _meta = got
        return arr
    Z = encoder.encode(batch)
    cache.put(key, Z, meta={"n": len(batch), "tag": tag, "extra": extra})
    return Z


class QGJetsDistillRecipe(Recipe):
    NAME = "qg_jets_distill"
    DESCRIPTION = (
        "Quark/gluon jets: measure warrant gap under particle-permutation + internal SO(2) symmetries, "
        "then distill symmetry-marginalized predictions into a single-pass student head."
    )

    def run(self, argv: list[str]) -> Dict[str, Any]:
        p = argparse.ArgumentParser(prog=self.NAME)
        self.add_common_args(p)

        # Dataset
        p.add_argument(
            "--num_data",
            type=int,
            default=50000,
            help="Number of jets to load from EnergyFlow cache.",
        )
        p.add_argument(
            "--generator", type=str, default="pythia", choices=["pythia", "herwig"]
        )
        p.add_argument(
            "--with_bc",
            action="store_true",
            help="Include b/c quark jets (different dataset).",
        )
        p.add_argument(
            "--ef_cache_dir",
            type=str,
            default="~/.energyflow",
            help="EnergyFlow dataset cache directory.",
        )
        p.add_argument("--n_train", type=int, default=20000)
        p.add_argument("--n_test", type=int, default=5000)
        p.add_argument(
            "--label_field",
            type=str,
            default="label",
            choices=["label"],
            help="Kept for schema-compatibility.",
        )

        # Symmetries / views
        p.add_argument("--k_train", type=int, default=8)
        p.add_argument("--k_test", type=int, default=16)
        p.add_argument("--no_perm", action="store_true")
        p.add_argument("--no_rotate", action="store_true")
        p.add_argument("--no_reflect", action="store_true")
        p.add_argument(
            "--theta_max",
            type=float,
            default=float(2 * np.pi),
            help="Rotation range: θ ~ U(-theta_max, theta_max)",
        )
        p.add_argument("--noise_y", type=float, default=0.0)
        p.add_argument("--noise_phi", type=float, default=0.0)
        p.add_argument(
            "--noise_pt", type=float, default=0.0, help="Multiplicative pt noise sigma"
        )

        # Encoder
        p.add_argument(
            "--encoder",
            type=str,
            default="qg_flatten",
            choices=["qg_flatten", "qg_eec2"],
        )
        p.add_argument("--max_particles", type=int, default=64)
        p.add_argument("--no_pid", action="store_true")
        p.add_argument("--pid_scale", type=float, default=0.01)

        # EEC2
        p.add_argument("--eec_bins", type=int, default=64)
        p.add_argument("--eec_rmax", type=float, default=1.0)
        p.add_argument("--no_log_mult", action="store_true")

        # Head
        p.add_argument("--hidden", type=int, default=512)
        p.add_argument("--depth", type=int, default=2)
        p.add_argument("--dropout", type=float, default=0.0)
        p.add_argument("--base_steps", type=int, default=800)
        p.add_argument("--student_steps", type=int, default=800)
        p.add_argument("--batch_size", type=int, default=256)
        p.add_argument("--lr", type=float, default=1e-3)
        p.add_argument("--hard_label_weight", type=float, default=0.1)

        # Demo gif
        p.add_argument(
            "--make_gif",
            action="store_true",
            help="Write a jet symmetry GIF to the run directory.",
        )
        p.add_argument("--gif_bins", type=int, default=160)
        p.add_argument("--gif_extent", type=float, default=0.8)
        p.add_argument("--gif_ms", type=int, default=90)

        args = p.parse_args(argv)

        ctx = self.build_context(args)
        import torch  # type: ignore

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Adapter
        adapter_cfg = QGJetsAdapterConfig(
            num_data=int(args.num_data),
            generator=str(args.generator),
            with_bc=bool(args.with_bc),
            cache_dir=str(args.ef_cache_dir),
            n_train=int(args.n_train),
            n_test=int(args.n_test),
            seed=int(args.seed),
        )
        adapter = QGJetsAdapter(adapter_cfg)
        world = adapter.load()
        world_fp = adapter.fingerprint()

        train = world["train"]
        test = world["test"]
        meta = world.get("meta", {})

        y_train = np.array([int(ex["label"]) for ex in train], dtype=np.int64)
        y_test = np.array([int(ex["label"]) for ex in test], dtype=np.int64)
        num_classes = 2

        # Encoder
        if str(args.encoder) == "qg_flatten":
            enc_cfg = QGFlattenEncoderConfig(
                max_particles=int(args.max_particles),
                include_pid=not bool(args.no_pid),
                pid_scale=float(args.pid_scale),
                pt_normalize=True,
            )
            encoder = QGFlattenEncoder(enc_cfg)
        else:
            enc_cfg = QGEEC2EncoderConfig(
                n_bins=int(args.eec_bins),
                r_max=float(args.eec_rmax),
                max_particles=int(args.max_particles),
                pt_normalize=True,
                include_log_mult=not bool(args.no_log_mult),
            )
            encoder = QGEEC2Encoder(enc_cfg)

        enc_fp = encoder.fingerprint()

        # Symmetries
        symmetries: List[Any] = []
        if not bool(args.no_perm):
            symmetries.append(QGPermutationSymmetry())
        if not bool(args.no_rotate):
            symmetries.append(
                QGSO2RotateSymmetry(QGSO2RotateConfig(theta_max=float(args.theta_max)))
            )
        if not bool(args.no_reflect):
            symmetries.append(QGReflectionSymmetry())
        if (
            float(args.noise_y) > 0
            or float(args.noise_phi) > 0
            or float(args.noise_pt) > 0
        ):
            symmetries.append(
                QGCoordNoiseSymmetry(
                    QGCoordNoiseConfig(
                        sigma_y=float(args.noise_y),
                        sigma_phi=float(args.noise_phi),
                        sigma_pt_frac=float(args.noise_pt),
                    )
                )
            )

        # Canonical embeddings
        Z_train = _cached_encode(
            cache=ctx.cache,
            world_fp=world_fp,
            enc_fp=enc_fp,
            split="train",
            tag="canon",
            extra={"k": 0},
            encoder=encoder,
            batch=train,
        )
        Z_test = _cached_encode(
            cache=ctx.cache,
            world_fp=world_fp,
            enc_fp=enc_fp,
            split="test",
            tag="canon",
            extra={"k": 0},
            encoder=encoder,
            batch=test,
        )

        # Deterministic train/val split (80/20)
        n = Z_train.shape[0]
        idx = np.arange(n)
        n_val = max(1, int(0.2 * n))
        idx_val = idx[:n_val]
        idx_tr = idx[n_val:]
        Z_tr, y_tr = Z_train[idx_tr], y_train[idx_tr]
        Z_val, y_val = Z_train[idx_val], y_train[idx_val]

        head_cfg = MLPHeadConfig(
            in_dim=int(Z_train.shape[1]),
            num_classes=num_classes,
            hidden=int(args.hidden),
            depth=int(args.depth),
            dropout=float(args.dropout),
        )

        # Train baseline
        base_head, base_metrics = train_hard_label_head(
            Z_tr,
            y_tr,
            Z_val,
            y_val,
            cfg=head_cfg,
            steps=int(args.base_steps),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            device=device,
            seed=int(args.seed),
        )

        P_base_test_canon = predict_proba(base_head, Z_test, device=device)
        base_acc = accuracy(P_base_test_canon, y_test)

        # Test views for warrant gap
        test_views = build_views(
            test, symmetries=symmetries, seed=int(args.seed) + 123, K=int(args.k_test)
        )
        Z_test_views: List[np.ndarray] = [Z_test]
        for j in range(1, int(args.k_test)):
            Zj = _cached_encode(
                cache=ctx.cache,
                world_fp=world_fp,
                enc_fp=enc_fp,
                split="test",
                tag=f"view_{j}",
                extra={
                    "k": int(args.k_test),
                    "j": j,
                    "symmetries": [
                        getattr(sym, "NAME", str(sym)) for sym in symmetries
                    ],
                },
                encoder=encoder,
                batch=test_views[j],
            )
            Z_test_views.append(Zj)
        P_views_base = np.stack(
            [predict_proba(base_head, Zj, device=device) for Zj in Z_test_views], axis=1
        )
        gap_base = warrant_gap_from_views(P_views_base)

        # Teacher expectation on train views
        train_views = build_views(
            train, symmetries=symmetries, seed=int(args.seed) + 999, K=int(args.k_train)
        )
        Z_train_views: List[np.ndarray] = [Z_train]
        for j in range(1, int(args.k_train)):
            Zj = _cached_encode(
                cache=ctx.cache,
                world_fp=world_fp,
                enc_fp=enc_fp,
                split="train",
                tag=f"view_{j}",
                extra={
                    "k": int(args.k_train),
                    "j": j,
                    "symmetries": [
                        getattr(sym, "NAME", str(sym)) for sym in symmetries
                    ],
                },
                encoder=encoder,
                batch=train_views[j],
            )
            Z_train_views.append(Zj)
        P_train_views = np.stack(
            [predict_proba(base_head, Zj, device=device) for Zj in Z_train_views],
            axis=1,
        )
        P_teacher = P_train_views.mean(axis=1)

        # Student
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

        P_stud_test_canon = predict_proba(stud_head, Z_test, device=device)
        stud_acc = accuracy(P_stud_test_canon, y_test)

        P_views_stud = np.stack(
            [predict_proba(stud_head, Zj, device=device) for Zj in Z_test_views], axis=1
        )
        gap_stud = warrant_gap_from_views(P_views_stud)

        base_gap = float(gap_base["mean_tv_to_mean"])
        stud_gap = float(gap_stud["mean_tv_to_mean"])

        base_pw = float(gap_base["mean_pairwise_tv"])
        stud_pw = float(gap_stud["mean_pairwise_tv"])

        tv_rel_improve = float((base_gap - stud_gap) / max(1e-9, base_gap))
        pw_rel_improve = float((base_pw - stud_pw) / max(1e-9, base_pw))
        acc_drop = float(base_acc - stud_acc)

        # Make/break logic: avoid "BREAK" when baseline is already stable.
        if base_gap < 0.02 and base_pw < 0.02:
            criterion = "(baseline already stable: base_gap<0.02 and base_pw<0.02) and acc_drop<=0.05"
            verdict = "MAKE OK"
        else:
            rel = max(tv_rel_improve, pw_rel_improve)
            criterion = "max(tv_rel_improve,pw_rel_improve)>=0.2 and acc_drop<=0.05"
            verdict = (
                "MAKE OK"
                if (rel >= 0.2 and acc_drop <= 0.05)
                else "BREAK / INCONCLUSIVE X"
            )

        # Persist probabilities for downstream visualisation (small)
        np.savez_compressed(
            ctx.out_dir / "probs_test_views.npz",
            y=y_test,
            base=P_views_base,
            student=P_views_stud,
        )

        # Optional GIF for a maximally unstable jet (more dramatic)
        gif_path = None
        demo = {}
        if bool(args.make_gif):
            # pick example with largest baseline pairwise TV between view0 and view1
            if P_views_base.shape[1] >= 2:
                tv01 = 0.5 * np.abs(P_views_base[:, 0, :] - P_views_base[:, 1, :]).sum(
                    axis=-1
                )
            else:
                tv01 = np.zeros((P_views_base.shape[0],), dtype=np.float32)
            i0 = int(np.argmax(tv01))

            parts_views = [
                test_views[j][i0]["particles"] for j in range(len(test_views))
            ]
            pb = P_views_base[i0]
            ps = P_views_stud[i0]
            lab = int(y_test[i0])

            gif_path = ctx.out_dir / "jet_nuisance.gif"
            write_jet_nuisance_gif(
                particles_views=parts_views,
                p_base_views=pb,
                p_student_views=ps,
                label=lab,
                out_path=gif_path,
                cfg=JetGifConfig(
                    extent=float(args.gif_extent),
                    bins=int(args.gif_bins),
                    duration_ms=int(args.gif_ms),
                    loop=0,
                ),
            )

            demo = {
                "example_index": i0,
                "baseline_pairwise_tv01": float(tv01[i0]),
                "gif": str(gif_path.name),
            }

        summary = {
            "exp": self.NAME,
            "device": device,
            "data": {
                "num_data": int(args.num_data),
                "generator": str(args.generator),
                "with_bc": bool(args.with_bc),
                "ef_cache_dir": str(args.ef_cache_dir),
                "n_train": int(args.n_train),
                "n_test": int(args.n_test),
                "meta": meta,
            },
            "encoder": {"name": str(args.encoder), "cfg": asdict(enc_cfg)},
            "symmetries": {
                "k_train": int(args.k_train),
                "k_test": int(args.k_test),
                "active": [getattr(sym, "NAME", str(sym)) for sym in symmetries],
                "theta_max": float(args.theta_max),
                "noise": {
                    "y": float(args.noise_y),
                    "phi": float(args.noise_phi),
                    "pt": float(args.noise_pt),
                },
            },
            "baseline": {
                "acc": base_acc,
                "gap_mean_tv_to_mean": base_gap,
                "gap_mean_pairwise_tv": base_pw,
                "val_acc": float(base_metrics.get("val_acc", np.nan)),
            },
            "student": {
                "acc": stud_acc,
                "gap_mean_tv_to_mean": stud_gap,
                "gap_mean_pairwise_tv": stud_pw,
                "val_acc": float(stud_metrics.get("val_acc", np.nan)),
            },
            "make_break": {
                "tv_rel_improve": tv_rel_improve,
                "pw_rel_improve": pw_rel_improve,
                "acc_drop": acc_drop,
                "criterion": criterion,
                "verdict": verdict,
            },
            "artifacts": {
                "probs_test_views": "probs_test_views.npz",
                **({"demo": demo} if demo else {}),
            },
        }

        (ctx.out_dir / "results.json").write_text(json.dumps(summary, indent=2))

        ctx.logger.log_metrics(
            {
                "baseline/acc": base_acc,
                "baseline/gap_tv": base_gap,
                "baseline/gap_pw": base_pw,
                "student/acc": stud_acc,
                "student/gap_tv": stud_gap,
                "student/gap_pw": stud_pw,
                "make_break/tv_rel_improve": tv_rel_improve,
                "make_break/pw_rel_improve": pw_rel_improve,
                "make_break/acc_drop": acc_drop,
            },
            step=0,
        )

        return summary
