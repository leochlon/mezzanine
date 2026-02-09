from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from typing import Any, Dict, List, Tuple

import numpy as np

from ..encoders.lj import (
    LJFlattenEncoder, LJFlattenEncoderConfig,
    LJRDFEncoder, LJRDFEncoderConfig,
)
from ..symmetries.lj import (
    LJSE3Symmetry, LJSE3Config,
    LJPermutationSymmetry, LJPermutationConfig,
    LJImageChoiceSymmetry, LJImageChoiceConfig,
    LJCoordinateNoiseSymmetry, LJCoordinateNoiseConfig,
)
from ..pipelines.text_distill import (
    MLPHeadConfig,
    train_hard_label_head,
    train_soft_label_head,
    predict_proba,
    accuracy,
    warrant_gap_from_views,
)
from ..viz.lj_distill import plot_lj_distill
from ..worlds.lj_fluid import LJFluidH5Adapter, LJFluidH5AdapterConfig
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
    N = len(xs)
    views: List[List[Dict[str, Any]]] = [list(xs)]
    if K <= 1:
        return views
    for j in range(1, K):
        vj: List[Dict[str, Any]] = []
        for i, x in enumerate(xs):
            xj = x
            for k, nu in enumerate(symmetries):
                xj = nu.sample(xj, seed=view_seed(seed, i, j, k))
            vj.append(xj)
        views.append(vj)
    return views


def _cached_encode(*, cache, world_fp: str, enc_fp: str, split: str, tag: str, extra: Dict[str, Any], encoder, batch: List[Any]) -> np.ndarray:
    if cache is None:
        return encoder.encode(batch)
    key = cache.make_key(world_fingerprint=world_fp, encoder_fingerprint=enc_fp, split=split, tag=tag, extra=extra)
    got = cache.get(key)
    if got is not None:
        arr, _meta = got
        return arr
    Z = encoder.encode(batch)
    cache.put(key, Z, meta={"n": len(batch), "tag": tag, "extra": extra})
    return Z


class LJFluidDistillRecipe(Recipe):
    NAME = "lj_fluid_distill"
    DESCRIPTION = "LJ fluid/droplet: measure warrant gaps under MD symmetries and distill symmetry-marginalized predictions into a single-pass head."

    def run(self, argv: list[str]) -> Dict[str, Any]:
        p = argparse.ArgumentParser(prog=self.NAME)
        self.add_common_args(p)

        # Dataset
        p.add_argument("--dataset", type=str, required=True, help="Path to HDF5 dataset (from examples/lj_generate_dataset.py)")
        p.add_argument("--label_field", type=str, default="state_id", choices=["state_id", "phase"])
        p.add_argument("--n_train", type=int, default=20000)
        p.add_argument("--n_test", type=int, default=5000)
        p.add_argument("--state_ids", type=int, nargs="+", default=None, help="Optional subset of state_id values")

        # Symmetries / views
        p.add_argument("--k_train", type=int, default=8)
        p.add_argument("--k_test", type=int, default=16)
        p.add_argument("--no_se3", action="store_true", help="Disable global rotation+translation symmetry")
        p.add_argument("--no_perm", action="store_true", help="Disable particle permutation symmetry")
        p.add_argument("--no_image", action="store_true", help="Disable per-particle periodic image-choice symmetry")
        p.add_argument("--noise_sigma", type=float, default=0.0, help="Stddev of coordinate noise symmetry (0 disables)")
        p.add_argument("--image_max", type=int, default=1, help="Max integer image shift per axis (if image symmetry enabled)")

        # Encoder
        p.add_argument("--encoder", type=str, default="lj_flatten", choices=["lj_flatten", "lj_rdf"])
        # Flatten
        p.add_argument("--center", type=str, default="box", choices=["none", "box", "mean"])
        p.add_argument("--scale_by_box", action="store_true")
        # RDF
        p.add_argument("--rdf_bins", type=int, default=128)
        p.add_argument("--rdf_rmax", type=float, default=3.5)
        p.add_argument("--rdf_counts_only", action="store_true")
        p.add_argument("--rdf_include_log_rho", action="store_true")

        # Head
        p.add_argument("--hidden", type=int, default=512)
        p.add_argument("--depth", type=int, default=1)
        p.add_argument("--dropout", type=float, default=0.0)
        p.add_argument("--base_steps", type=int, default=800)
        p.add_argument("--student_steps", type=int, default=800)
        p.add_argument("--batch_size", type=int, default=256)
        p.add_argument("--lr", type=float, default=1e-3)
        p.add_argument("--hard_label_weight", type=float, default=0.0)

        args = p.parse_args(argv)

        ctx = self.build_context(args)
        # Torch is optional at import time, but required here.
        import torch  # type: ignore
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Adapter
        adapter_cfg = LJFluidH5AdapterConfig(
            path=str(args.dataset),
            label_field=str(args.label_field),
            n_train=int(args.n_train),
            n_test=int(args.n_test),
            seed=int(args.seed),
            state_ids=list(args.state_ids) if args.state_ids is not None else None,
        )
        adapter = LJFluidH5Adapter(adapter_cfg)
        world = adapter.load()
        world_fp = adapter.fingerprint()

        train = world["train"]
        test = world["test"]
        meta = world.get("meta", {})

        y_train = np.array([int(ex["label"]) for ex in train], dtype=np.int64)
        y_test = np.array([int(ex["label"]) for ex in test], dtype=np.int64)
        num_classes = int(max(y_train.max(initial=0), y_test.max(initial=0)) + 1)

        # Encoder
        if str(args.encoder) == "lj_flatten":
            enc_cfg = LJFlattenEncoderConfig(center=str(args.center), scale_by_box=bool(args.scale_by_box))
            encoder = LJFlattenEncoder(enc_cfg)
        else:
            enc_cfg = LJRDFEncoderConfig(
                n_bins=int(args.rdf_bins),
                r_max=float(args.rdf_rmax),
                include_log_rho=bool(args.rdf_include_log_rho),
                counts_only=bool(args.rdf_counts_only),
            )
            encoder = LJRDFEncoder(enc_cfg)
        enc_fp = encoder.fingerprint()

        # Symmetries
        symmetries: List[Any] = []
        if not bool(args.no_se3):
            symmetries.append(LJSE3Symmetry(LJSE3Config(rotate=True, translate=True, wrap=True)))
        if not bool(args.no_perm):
            symmetries.append(LJPermutationSymmetry(LJPermutationConfig()))
        if not bool(args.no_image):
            symmetries.append(LJImageChoiceSymmetry(LJImageChoiceConfig(max_image=int(args.image_max), wrap_after=False)))
        if float(args.noise_sigma) > 0:
            # If we also use image-choice, do NOT wrap, else it cancels the image shift.
            wrap = bool(args.no_image)
            symmetries.append(LJCoordinateNoiseSymmetry(LJCoordinateNoiseConfig(sigma=float(args.noise_sigma), wrap=wrap)))

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

        # Train baseline head
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

        # Baseline accuracy on canonical test
        P_base_test_canon = predict_proba(base_head, Z_test, device=device)
        base_acc = accuracy(P_base_test_canon, y_test)

        # Build symmetry views for test and compute warrant gap
        test_views = build_views(test, symmetries=symmetries, seed=int(args.seed) + 123, K=int(args.k_test))
        Z_test_views: List[np.ndarray] = [Z_test]
        for j in range(1, int(args.k_test)):
            Zj = _cached_encode(
                cache=ctx.cache,
                world_fp=world_fp,
                enc_fp=enc_fp,
                split="test",
                tag=f"view_{j}",
                extra={"k": int(args.k_test), "j": j, "symmetries": [getattr(nu, "NAME", str(nu)) for nu in symmetries]},
                encoder=encoder,
                batch=test_views[j],
            )
            Z_test_views.append(Zj)
        P_views = np.stack([predict_proba(base_head, Zj, device=device) for Zj in Z_test_views], axis=1)
        gap_base = warrant_gap_from_views(P_views)

        # Teacher expectation on train views
        train_views = build_views(train, symmetries=symmetries, seed=int(args.seed) + 999, K=int(args.k_train))
        Z_train_views: List[np.ndarray] = [Z_train]
        for j in range(1, int(args.k_train)):
            Zj = _cached_encode(
                cache=ctx.cache,
                world_fp=world_fp,
                enc_fp=enc_fp,
                split="train",
                tag=f"view_{j}",
                extra={"k": int(args.k_train), "j": j, "symmetries": [getattr(nu, "NAME", str(nu)) for nu in symmetries]},
                encoder=encoder,
                batch=train_views[j],
            )
            Z_train_views.append(Zj)
        P_train_views = np.stack([predict_proba(base_head, Zj, device=device) for Zj in Z_train_views], axis=1)
        P_teacher = P_train_views.mean(axis=1)

        # Train student head to match teacher on canonical embeddings
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

        # Student gap across same view embeddings
        P_stud_views = np.stack([predict_proba(stud_head, Zj, device=device) for Zj in Z_test_views], axis=1)
        gap_stud = warrant_gap_from_views(P_stud_views)

        # Save a tiny payload for GIF/HTML viz (one example across symmetry views)
        try:
            i0 = 0
            y0 = int(y_test[i0])
            box0 = float(test_views[0][i0]["box"])
            pos0 = np.stack([np.asarray(test_views[j][i0]["pos"], dtype=np.float32) for j in range(len(test_views))], axis=0)
            p_base0 = np.asarray(P_views[i0], dtype=np.float32)
            p_stud0 = np.asarray(P_stud_views[i0], dtype=np.float32)
            np.savez_compressed(ctx.out_dir / "viz_payload.npz",
                                pos=pos0,
                                box=np.array([box0], dtype=np.float32),
                                y=np.array([y0], dtype=np.int64),
                                p_base=p_base0,
                                p_stud=p_stud0)
        except Exception:
            pass

        base_gap = float(gap_base["mean_tv_to_mean"])
        stud_gap = float(gap_stud["mean_tv_to_mean"])
        base_pw  = float(gap_base["mean_pairwise_tv"])
        stud_pw  = float(gap_stud["mean_pairwise_tv"])

        tv_rel_improve = float((base_gap - stud_gap) / max(1e-9, base_gap))
        pw_rel_improve = float((base_pw  - stud_pw)  / max(1e-9, base_pw))
        rel_improve = float(max(tv_rel_improve, pw_rel_improve))

        acc_drop = float(base_acc - stud_acc)

        # Conclusive MD gate:
        # - don't demand huge relative improvement when you're already good (RDF case)
        # - demand: accuracy doesn't crater AND student is stable under symmetries
        verdict = "MAKE ✅" if (acc_drop <= 0.05 and stud_pw <= 0.10) else "CHECK ⚠️"

        summary = {
            "exp": self.NAME,
            "device": device,
            "data": {
                "dataset": str(args.dataset),
                "label_field": str(args.label_field),
                "n_train": int(args.n_train),
                "n_test": int(args.n_test),
                "state_ids": list(args.state_ids) if args.state_ids is not None else None,
                "meta": meta,
            },
            "encoder": {
                "name": str(args.encoder),
                "cfg": asdict(enc_cfg),
            },
            "symmetries": {
                "k_train": int(args.k_train),
                "k_test": int(args.k_test),
                "active": [getattr(nu, "NAME", str(nu)) for nu in symmetries],
                "noise_sigma": float(args.noise_sigma),
                "image_max": int(args.image_max),
            },
            "baseline": {
                "acc": base_acc,
                "gap_mean_tv_to_mean": base_gap,
                "gap_mean_pairwise_tv": float(gap_base["mean_pairwise_tv"]),
                "val_acc": float(base_metrics.get("val_acc", np.nan)),
            },
            "student": {
                "acc": stud_acc,
                "gap_mean_tv_to_mean": stud_gap,
                "gap_mean_pairwise_tv": float(gap_stud["mean_pairwise_tv"]),
                "val_acc": float(stud_metrics.get("val_acc", np.nan)),
            },
            "make_break": {
                "tv_rel_improve": tv_rel_improve,
                "pairwise_rel_improve": pw_rel_improve,
                "rel_improve": rel_improve,
                "acc_drop": acc_drop,
                "criterion": "acc_drop<=0.05 and student_pairwise_tv<=0.10 (plus rel improvements reported)",
                "verdict": verdict,
            },
        }

        (ctx.out_dir / "results.json").write_text(json.dumps(summary, indent=2))
        plot_lj_distill(summary, ctx.out_dir / "diagnostics.png")

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
