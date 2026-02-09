from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from ..core.config import load_config, deep_update
from ..encoders.hf_vision import HFVisionEncoder, HFVisionEncoderConfig
from ..pipelines.latent_dynamics import (
    LatentDynamicsTrainConfig,
    train_latent_dynamics,
    eval_latent_dynamics,
)
from ..viz.latent_dynamics import plot_diagnostics, save_montage
from ..worlds.iphyre import IPhyreCollectConfig, IPhyreAdapter
from .recipe_base import Recipe


class IPhyreLatentDynamicsRecipe(Recipe):
    NAME = "iphyre_latent_dynamics"
    DESCRIPTION = "I-PHYRE: distill action-conditioned latent dynamics in JEPA-like latents (with action-shuffle counterfactual)."

    def run(self, argv: list[str]) -> Dict[str, Any]:
        p = argparse.ArgumentParser(prog=self.NAME)
        self.add_common_args(p)

        # World / data
        p.add_argument("--games", type=str, default="hole,seesaw", help="Comma-separated I-PHYRE games")
        p.add_argument("--n_train", type=int, default=3000)
        p.add_argument("--n_test", type=int, default=768)
        p.add_argument("--max_steps", type=int, default=260)
        p.add_argument("--delta_seconds", type=float, default=4.0)
        p.add_argument("--sim_dt", type=float, default=0.1)

        # Action / symmetry knobs
        p.add_argument("--action_scheme", type=str, default="single", choices=["single", "window"],
                       help="single click vs windowed action representation (window uses per_episode_samples > 1).")
        p.add_argument("--action_prob", type=float, default=0.08, help="Probability of a random click per step (data generation).")
        p.add_argument("--p_no_action", type=float, default=0.35, help="Fraction of training examples forced to have no action.")

        # Encoder knobs
        p.add_argument("--ijepa_model", type=str, default="facebook/ijepa_vith14_1k")
        p.add_argument("--encode_bs", type=int, default=32)
        p.add_argument("--no_fp16", action="store_true")
        p.add_argument("--embed_mode", type=str, default="mean_std", choices=["cls", "mean", "mean_std"])
        p.add_argument("--embed_layer", type=int, default=-4)

        # Training knobs
        p.add_argument("--train_steps", type=int, default=800)
        p.add_argument("--train_bs", type=int, default=256)
        p.add_argument("--train_lr", type=float, default=1e-3)
        p.add_argument("--hidden", type=int, default=1024)
        p.add_argument("--depth", type=int, default=2)

        # Output
        p.add_argument("--out_dir", type=str, default=str(self.out_dir))

        args = p.parse_args(argv)

        # Apply config file defaults BEFORE seeding/logger/cache creation
        file_cfg = load_config(getattr(args, "config", None))
        merged_cfg = deep_update(file_cfg, self.config)
        self.apply_config_defaults(p, args, merged_cfg)

        ctx = self.build_context(args)
        out_dir = ctx.out_dir
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # --- World adapter ---
        w_cfg = IPhyreCollectConfig(
            games=[g.strip() for g in str(args.games).split(",") if g.strip()],
            n_train=int(args.n_train),
            n_test=int(args.n_test),
            max_steps=int(args.max_steps),
            delta_seconds=float(args.delta_seconds),
            sim_dt=float(args.sim_dt),
            action_prob=float(args.action_prob),
            p_no_action=float(args.p_no_action),
            per_episode_samples=4 if str(args.action_scheme) == "window" else 1,
            seed=int(args.seed),
        )
        world = IPhyreAdapter(w_cfg)
        data = world.load()
        world_fp = world.fingerprint()

        # --- Encoder ---
        enc_cfg = HFVisionEncoderConfig(
            model_name=str(args.ijepa_model),
            batch_size=int(args.encode_bs),
            fp16=not bool(args.no_fp16),
            embed_mode=str(args.embed_mode),  # type: ignore[arg-type]
            embed_layer=int(args.embed_layer),
        )
        enc = HFVisionEncoder(enc_cfg, device=device)
        enc_fp = enc.fingerprint()

        # --- Cache-aware encoding ---
        train = data["train"]
        test = data["test"]

        imgs_train_t = [ex["img_t"] for ex in train]
        imgs_train_tp = [ex["img_tp"] for ex in train]
        a_train = np.stack([ex["action_feat"] for ex in train], axis=0).astype(np.float32)

        imgs_test_t = [ex["img_t"] for ex in test]
        imgs_test_tp = [ex["img_tp"] for ex in test]
        a_test = np.stack([ex["action_feat"] for ex in test], axis=0).astype(np.float32)

        if ctx.cache is not None:
            z_train_t = ctx.cache.get_or_compute(
                ctx.cache.make_key(
                    world_fingerprint=world_fp,
                    encoder_fingerprint=enc_fp,
                    split="train",
                    tag="z_t",
                    extra={"embed_mode": args.embed_mode, "embed_layer": args.embed_layer},
                ),
                lambda: enc.encode(imgs_train_t),
                meta={"n": len(imgs_train_t), "world_fp": world_fp, "enc_fp": enc_fp},
            )
            z_train_tp = ctx.cache.get_or_compute(
                ctx.cache.make_key(
                    world_fingerprint=world_fp,
                    encoder_fingerprint=enc_fp,
                    split="train",
                    tag="z_tp",
                    extra={"embed_mode": args.embed_mode, "embed_layer": args.embed_layer, "delta_seconds": args.delta_seconds},
                ),
                lambda: enc.encode(imgs_train_tp),
                meta={"n": len(imgs_train_tp), "world_fp": world_fp, "enc_fp": enc_fp},
            )
            z_test_t = ctx.cache.get_or_compute(
                ctx.cache.make_key(
                    world_fingerprint=world_fp,
                    encoder_fingerprint=enc_fp,
                    split="test",
                    tag="z_t",
                    extra={"embed_mode": args.embed_mode, "embed_layer": args.embed_layer},
                ),
                lambda: enc.encode(imgs_test_t),
                meta={"n": len(imgs_test_t), "world_fp": world_fp, "enc_fp": enc_fp},
            )
            z_test_tp = ctx.cache.get_or_compute(
                ctx.cache.make_key(
                    world_fingerprint=world_fp,
                    encoder_fingerprint=enc_fp,
                    split="test",
                    tag="z_tp",
                    extra={"embed_mode": args.embed_mode, "embed_layer": args.embed_layer, "delta_seconds": args.delta_seconds},
                ),
                lambda: enc.encode(imgs_test_tp),
                meta={"n": len(imgs_test_tp), "world_fp": world_fp, "enc_fp": enc_fp},
            )
        else:
            z_train_t = enc.encode(imgs_train_t)
            z_train_tp = enc.encode(imgs_train_tp)
            z_test_t = enc.encode(imgs_test_t)
            z_test_tp = enc.encode(imgs_test_tp)


        # --- Predictors + evaluation ---
        cfg = LatentDynamicsTrainConfig(
            steps=int(args.train_steps),
            batch_size=int(args.train_bs),
            lr=float(args.train_lr),
            hidden=int(args.hidden),
            depth=int(args.depth),
        )

        train_out = train_latent_dynamics(z_train_t, z_train_tp, a_train, cfg, device=device)

        ev = eval_latent_dynamics(train_out, z_test_t, z_test_tp, a_test, device=device, seed=int(args.seed))

        # --- Visuals ---
        save_montage(data["test"], out_dir / "montage.png", n_rows=8, thumb=192)
        plot_diagnostics(ev, out_dir / "diagnostics.png")

        
        def _flatten(prefix: str, obj: Any, out: Dict[str, float]) -> None:
            if isinstance(obj, dict):
                for kk, vv in obj.items():
                    _flatten(prefix + str(kk) + "/", vv, out)
            else:
                try:
                    out[prefix[:-1]] = float(obj)
                except Exception:
                    pass

        result = {
            "exp": "iphyre_latent_dynamics",
            "device": device,
            "world": {"adapter": "iphyre", "fingerprint": world_fp, "config": w_cfg.__dict__, "meta": data["meta"]},
            "encoder": {"name": "hf_vision", "fingerprint": enc_fp, "config": enc_cfg.__dict__},
            "metrics": ev["metrics"],
            "deltas": ev["deltas"],
            "make_break": ev["make_break"],
            "artifacts": {
                "results_json": str(out_dir / "results.json"),
                "diagnostics_png": str(out_dir / "diagnostics.png"),
                "montage_png": str(out_dir / "montage.png"),
            },
        }

        (out_dir / "results.json").write_text(json.dumps(result, indent=2))

        # Optional logging hooks
        try:
            flat: Dict[str, float] = {}
            _flatten("metrics/", result["metrics"], flat)
            _flatten("deltas/", result["deltas"], flat)
            ctx.logger.log_metrics(flat)
            ctx.logger.log_text("make_break/verdict", str(result["make_break"].get("verdict", "")))
            ctx.logger.log_artifact(out_dir / "results.json", name="results.json")
            ctx.logger.log_artifact(out_dir / "diagnostics.png", name="diagnostics.png")
            ctx.logger.log_artifact(out_dir / "montage.png", name="montage.png")
        finally:
            try:
                ctx.logger.close()
            except Exception:
                pass

        return result
