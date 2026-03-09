#!/usr/bin/env python3
"""Distill D4 orbit-averaged depth into Depth Anything via fine-tuning.

The Mezzanine pattern for depth equivariance:
  1. For each training image, compute D4 orbit-averaged depth (teacher signal)
  2. Fine-tune the full model to predict this orbit average in a single pass
  3. Where model is already equivariant → target ≈ original → no change
  4. Where viewpoint bias exists → target is corrected → model learns the fix

Usage:
    # Quick test (default: 50 epochs, 30 COCO images)
    python scripts/distill_orbit_average.py --device cuda

    # Longer run for better results
    python scripts/distill_orbit_average.py --epochs 200 --n-train 100 --device cuda

    # Resume from checkpoint
    python scripts/distill_orbit_average.py --resume out_distill/student_model --epochs 100
"""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# ── Data ─────────────────────────────────────────────────────────────────


def download_coco_images(
    n: int, cache_dir: Path, seed: int = 42
) -> list[tuple[str, np.ndarray]]:
    """Download n random COCO val2017 images. Returns (name, HWC uint8) pairs."""
    import requests

    cache_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)

    # Known valid COCO val2017 IDs
    known_ids = [
        39769,
        397133,
        37777,
        87470,
        131131,
        181859,
        87875,
        146457,
        459467,
        284991,
        579635,
        226111,
        348881,
        80340,
        227765,
        295316,
        360661,
        2587,
        174004,
        279541,
        180135,
        84270,
        226592,
        349860,
        434230,
        532493,
        173091,
        153529,
        176446,
        87038,
        56350,
        404484,
        289393,
        581929,
        578489,
        574769,
        572678,
        565778,
        562150,
        558854,
        554291,
        551215,
        547816,
        545129,
        544565,
        542145,
        540763,
        539715,
        537991,
        536947,
        534827,
        531968,
        530854,
        528399,
        527029,
        525155,
        522713,
        520264,
        518770,
        517687,
        515579,
        514508,
        512194,
        511321,
        509735,
        508602,
        506656,
        504711,
        502737,
        501523,
        499313,
        498286,
        496861,
        494913,
        493286,
        491757,
        490171,
        488592,
        487583,
        486479,
        484792,
        483108,
        481404,
        479248,
        477118,
        474854,
        473237,
        471087,
        469067,
        466567,
        464522,
        462565,
        460347,
        458790,
        456559,
        455483,
        453860,
        451478,
        449406,
        447313,
        445365,
        443303,
        441286,
        439180,
        437313,
        435081,
        433068,
        430961,
        428454,
        426329,
        424551,
        422886,
        421023,
        419096,
        417779,
        416343,
        414673,
        412894,
        410650,
        409475,
        407614,
        405249,
        403385,
        401244,
        399462,
        397327,
    ]
    rng.shuffle(known_ids)

    pairs: list[tuple[str, np.ndarray]] = []
    for img_id in known_ids:
        if len(pairs) >= n:
            break
        fname = f"{img_id:012d}.jpg"
        cache_path = cache_dir / fname

        if cache_path.exists():
            try:
                img = Image.open(cache_path).convert("RGB")
                pairs.append((f"coco_{img_id}", np.asarray(img)))
                continue
            except Exception:
                cache_path.unlink(missing_ok=True)

        url = f"http://images.cocodataset.org/val2017/{fname}"
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            img = Image.open(io.BytesIO(r.content)).convert("RGB")
            img.save(cache_path)
            pairs.append((f"coco_{img_id}", np.asarray(img)))
        except Exception:
            continue

    print(f"  Loaded {len(pairs)} images (cached in {cache_dir})")
    return pairs


def load_local_images(directory: Path) -> list[tuple[str, np.ndarray]]:
    """Load local test images."""
    pairs = []
    for ext in ("*.jpg", "*.png"):
        for p in sorted(directory.glob(ext)):
            img = np.asarray(Image.open(p).convert("RGB"))
            pairs.append((p.stem, img))
    return pairs


# ── Metrics ──────────────────────────────────────────────────────────────


def vertical_gradient_strength(depth_map: np.ndarray) -> float:
    """Pearson correlation between row index and row-mean depth."""
    h = depth_map.shape[0]
    row_means = depth_map.mean(axis=1).astype(np.float64)
    if row_means.std() < 1e-10:
        return 0.0
    return float(np.corrcoef(np.arange(h, dtype=np.float64), row_means)[0, 1])


def evaluate_model(encoder, sym, images, label=""):
    """Evaluate warrant gap and vertical bias for a set of images."""
    results = []
    for name, img_np in images:
        # Original prediction
        orig_depth = encoder.predict(img_np)
        orig_vbias = vertical_gradient_strength(orig_depth)

        # Orbit-averaged prediction
        views = sym.batch(img_np)
        depths = [encoder.predict(v) for v in views]
        aligned = [sym.inverse(d, i) for i, d in enumerate(depths)]
        stack = np.stack(aligned)
        orbit_avg = stack.mean(axis=0)
        orbit_vbias = vertical_gradient_strength(orbit_avg)

        # Warrant gap (coefficient of variation)
        pixel_mean = stack.mean(axis=0)
        pixel_std = stack.std(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            cv = np.where(pixel_mean > 1e-8, pixel_std / pixel_mean, 0.0)

        results.append(
            {
                "image": name,
                "vbias": float(orig_vbias),
                "vbias_orbit": float(orbit_vbias),
                "mean_cv": float(cv.mean()),
            }
        )

    if results and label:
        mean_vbias = np.mean([abs(r["vbias"]) for r in results])
        mean_cv = np.mean([r["mean_cv"] for r in results])
        print(f"  {label}: mean |vbias|={mean_vbias:.4f}, mean CV={mean_cv:.4f}")
        for r in results:
            is_td = "robot" in r["image"] or "_td" in r["image"]
            marker = "TD" if is_td else "  "
            print(
                f"    {marker} {r['image']:<20} vbias={r['vbias']:+.4f}  CV={r['mean_cv']:.4f}"
            )

    return results


# ── Training ─────────────────────────────────────────────────────────────


def compute_orbit_targets(encoder, sym, images):
    """Pre-compute orbit-averaged depth targets. Returns dict name → ndarray."""
    targets = {}
    n = len(images)
    t0 = time.time()
    for i, (name, img_np) in enumerate(images):
        views = sym.batch(img_np)
        depths = [encoder.predict(v) for v in views]
        aligned = [sym.inverse(d, j) for j, d in enumerate(depths)]
        targets[name] = np.stack(aligned).mean(axis=0)

        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate
            print(f"  [{i + 1}/{n}] {rate:.1f} img/s, ETA {eta:.0f}s")

    return targets


def train(
    model,
    processor,
    train_imgs,
    targets,
    *,
    epochs: int,
    lr: float,
    batch_size: int,
    device: str,
    eval_imgs=None,
    eval_encoder=None,
    eval_sym=None,
    out_dir=None,
):
    """Fine-tune model to predict orbit-averaged targets."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_names = [name for name, _ in train_imgs]
    train_images = {name: img for name, img in train_imgs}
    N = len(train_names)

    logs = []
    for epoch in range(1, epochs + 1):
        indices = np.random.permutation(N)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, N, batch_size):
            batch_idx = indices[start : start + batch_size]
            batch_pils = []
            batch_targets = []

            for idx in batch_idx:
                name = train_names[idx]
                pil = Image.fromarray(train_images[name])
                batch_pils.append(pil)
                batch_targets.append(targets[name])

            target_sizes = [(p.height, p.width) for p in batch_pils]
            inputs = processor(
                images=batch_pils, return_tensors="pt", keep_aspect_ratio=False
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)

            # Post-process to get per-image depth at original resolution
            post = processor.post_process_depth_estimation(
                outputs, target_sizes=target_sizes
            )

            loss = torch.tensor(0.0, device=device, requires_grad=True)
            for b in range(len(batch_targets)):
                pred = post[b]["predicted_depth"].to(device)  # (H, W)
                tgt = torch.from_numpy(batch_targets[b]).float().to(device)

                # Ensure same shape (should match via target_sizes)
                if pred.shape != tgt.shape:
                    tgt = F.interpolate(
                        tgt.unsqueeze(0).unsqueeze(0),
                        size=pred.shape,
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze()

                loss = loss + F.mse_loss(pred, tgt)

            loss = loss / len(batch_targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}: loss={avg_loss:.6f}")
            log_entry = {"epoch": epoch, "loss": avg_loss}

            # Periodic evaluation
            if epoch % 25 == 0 and eval_imgs and eval_encoder and eval_sym:
                # Update encoder's internal model reference to current weights
                eval_encoder.model = model
                model.eval()
                eval_results = evaluate_model(
                    eval_encoder, eval_sym, eval_imgs, f"Epoch {epoch}"
                )
                log_entry["eval"] = eval_results
                model.train()

            logs.append(log_entry)

        # Save checkpoint periodically
        if out_dir and epoch % 50 == 0:
            model.eval()
            model.save_pretrained(out_dir / "student_model")
            processor.save_pretrained(out_dir / "student_model")
            model.train()

    return logs


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Distill D4 orbit-averaged depth into Depth Anything"
    )
    parser.add_argument("--model", default="depth-anything/Depth-Anything-V2-Small-hf")
    parser.add_argument("--out", default="out_distill", help="Output directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--n-train", type=int, default=30, help="COCO training images")
    parser.add_argument(
        "--subgroup", default="d4", choices=["d4", "vflip", "flips", "rotations"]
    )
    parser.add_argument(
        "--resume", default=None, help="Path to student_model dir to resume from"
    )
    parser.add_argument(
        "--test-images", default="test_images", help="Directory with eval images"
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "image_cache"

    # Load model
    print(f"Loading {args.resume or args.model}...")
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    model_path = args.resume or args.model
    processor = AutoImageProcessor.from_pretrained(
        args.model if not args.resume else args.resume
    )
    model = AutoModelForDepthEstimation.from_pretrained(model_path).to(args.device)

    # Encoder wrapper (for orbit averaging + evaluation)
    from mezzanine.encoders.depth_anything import (
        DepthAnythingEncoder,
        DepthAnythingEncoderConfig,
    )
    from mezzanine.symmetries.depth_geometric import (
        DepthGeometricSymmetry,
        DepthGeometricSymmetryConfig,
    )

    encoder = DepthAnythingEncoder(
        DepthAnythingEncoderConfig(model_name=args.model), device=args.device
    )
    sym = DepthGeometricSymmetry(DepthGeometricSymmetryConfig(subgroup=args.subgroup))
    print(f"Subgroup: {args.subgroup} ({sym.k} elements)")

    # Load images
    print(f"\nDownloading {args.n_train} COCO training images...")
    train_imgs = download_coco_images(args.n_train, cache_dir)

    eval_imgs = []
    test_dir = Path(args.test_images)
    if test_dir.exists():
        eval_imgs = load_local_images(test_dir)
        print(f"Eval images: {len(eval_imgs)}")

    # Evaluate baseline
    print("\n--- Baseline ---")
    pre_results = evaluate_model(encoder, sym, eval_imgs, "Pre-train")

    # Pre-compute orbit-averaged targets
    print(f"\nComputing orbit targets ({sym.k} passes per image)...")
    targets = compute_orbit_targets(encoder, sym, train_imgs)

    # Train
    print(
        f"\nTraining: {args.epochs} epochs, lr={args.lr}, batch_size={args.batch_size}"
    )
    logs = train(
        model,
        processor,
        train_imgs,
        targets,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=args.device,
        eval_imgs=eval_imgs,
        eval_encoder=encoder,
        eval_sym=sym,
        out_dir=out_dir,
    )

    # Final evaluation
    print("\n--- After training ---")
    encoder.model = model
    model.eval()
    post_results = evaluate_model(encoder, sym, eval_imgs, "Post-train")

    # Save model
    model.save_pretrained(out_dir / "student_model")
    processor.save_pretrained(out_dir / "student_model")
    print(f"\nModel saved to {out_dir / 'student_model'}")

    # Save results
    results = {
        "config": {
            "model": args.model,
            "subgroup": args.subgroup,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "n_train": len(train_imgs),
            "n_eval": len(eval_imgs),
            "resumed_from": args.resume,
        },
        "pre_train": pre_results,
        "post_train": post_results,
        "training_log": logs,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    print(f"\n{'=' * 60}")
    print("COMPARISON")
    print(f"{'=' * 60}")
    for pre, post in zip(pre_results, post_results):
        name = pre["image"]
        delta_vbias = abs(post["vbias"]) - abs(pre["vbias"])
        delta_cv = post["mean_cv"] - pre["mean_cv"]
        marker = "✓" if delta_vbias < -0.05 else ("✗" if delta_vbias > 0.05 else "~")
        print(
            f"  {marker} {name:<20} |vbias| {abs(pre['vbias']):.3f}→{abs(post['vbias']):.3f} ({delta_vbias:+.3f})  "
            f"CV {pre['mean_cv']:.3f}→{post['mean_cv']:.3f} ({delta_cv:+.3f})"
        )


if __name__ == "__main__":
    main()
