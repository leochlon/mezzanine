#!/usr/bin/env python3
"""Measure the warrant gap of Depth Anything under D4 geometric symmetry.

The "warrant gap" quantifies how much a model's predictions change under
symmetry transforms that *should* be equivariant.  For monocular depth,
a vertically flipped image should produce a vertically flipped depth map.
If the model instead always predicts "bottom = closer", the gap is large.

This script:
  1. Loads test images (from URLs, local paths, or a standard dataset)
  2. For each image, runs Depth Anything on all D4 (or subgroup) transforms
  3. Inverse-transforms each depth prediction back to the original frame
  4. Measures disagreement across views (= the warrant gap)
  5. Computes the orbit-averaged prediction and compares to original
  6. Outputs quantitative metrics + diagnostic visualizations

Usage:
    python scripts/measure_depth_warrant_gap.py \
        --model depth-anything/Depth-Anything-V2-Small-hf \
        --subgroup d4 \
        --images url \
        --out out_depth_warrant \
        --device cuda

    # Just vertical flip (fastest, tests the main bias):
    python scripts/measure_depth_warrant_gap.py --subgroup vflip

    # Use local images:
    python scripts/measure_depth_warrant_gap.py --images path/to/img1.jpg path/to/img2.jpg
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

# ── Test images ──────────────────────────────────────────────────────────

# A mix of standard perspective + unusual viewpoints
DEFAULT_TEST_URLS = [
    # Standard forward-facing scenes
    ("coco_cats", "http://images.cocodataset.org/val2017/000000039769.jpg"),
    ("coco_baseball", "http://images.cocodataset.org/val2017/000000397133.jpg"),
    ("coco_kitchen", "http://images.cocodataset.org/val2017/000000037777.jpg"),
    # Top-down / overhead views (where bias should be worst)
    ("coco_food_topdown", "http://images.cocodataset.org/val2017/000000080340.jpg"),
    ("coco_pizza_topdown", "http://images.cocodataset.org/val2017/000000227765.jpg"),
    # Upward-looking
    ("coco_plane_sky", "http://images.cocodataset.org/val2017/000000146457.jpg"),
]


def load_test_images(sources: list[str] | None) -> list[tuple[str, np.ndarray]]:
    """Load test images from URLs, local paths, or defaults."""
    import requests

    pairs: list[tuple[str, np.ndarray]] = []

    if sources is None or sources == ["url"]:
        print(f"Loading {len(DEFAULT_TEST_URLS)} default test images from COCO...")
        for name, url in DEFAULT_TEST_URLS:
            try:
                resp = requests.get(url, timeout=15)
                resp.raise_for_status()
                img = Image.open(__import__("io").BytesIO(resp.content)).convert("RGB")
                pairs.append((name, np.asarray(img)))
                print(f"  ✓ {name} ({img.size[0]}×{img.size[1]})")
            except Exception as e:
                print(f"  ✗ {name}: {e}")
    else:
        for path in sources:
            try:
                img = Image.open(path).convert("RGB")
                name = Path(path).stem
                pairs.append((name, np.asarray(img)))
                print(f"  ✓ {name} ({img.size[0]}×{img.size[1]})")
            except Exception as e:
                print(f"  ✗ {path}: {e}")

    if not pairs:
        raise RuntimeError("No images loaded!")
    return pairs


# ── Warrant gap metrics ──────────────────────────────────────────────────


def compute_warrant_gap(
    depths_aligned: list[np.ndarray],
) -> dict:
    """Compute warrant gap metrics from a list of depth maps aligned to the same frame.

    Returns dict with:
        - mean_std: mean per-pixel std across views (absolute)
        - mean_cv: mean coefficient of variation (std/mean)
        - max_std: max per-pixel std
        - vertical_bias_original: vertical gradient strength in the original prediction
        - vertical_bias_averaged: vertical gradient strength after orbit-averaging
    """
    stack = np.stack(depths_aligned, axis=0)  # (K, H, W)
    pixel_mean = stack.mean(axis=0)  # (H, W)
    pixel_std = stack.std(axis=0)  # (H, W)

    # Coefficient of variation (avoid div by zero)
    with np.errstate(divide="ignore", invalid="ignore"):
        cv = np.where(pixel_mean > 1e-8, pixel_std / pixel_mean, 0.0)

    # Vertical bias: correlation between row index and mean depth of that row
    h = pixel_mean.shape[0]
    row_indices = np.arange(h, dtype=np.float64)

    def vertical_gradient_strength(depth_map: np.ndarray) -> float:
        """Pearson correlation between row index and row-mean depth."""
        row_means = depth_map.mean(axis=1).astype(np.float64)
        if row_means.std() < 1e-10:
            return 0.0
        corr = np.corrcoef(row_indices, row_means)[0, 1]
        return float(corr)

    orbit_avg = pixel_mean  # = orbit-averaged prediction

    return {
        "mean_std": float(pixel_std.mean()),
        "mean_cv": float(cv.mean()),
        "max_std": float(pixel_std.max()),
        "vertical_bias_original": vertical_gradient_strength(depths_aligned[0]),
        "vertical_bias_averaged": vertical_gradient_strength(orbit_avg),
    }


# ── Visualization ────────────────────────────────────────────────────────


def save_diagnostics(
    name: str,
    original_img: np.ndarray,
    depths_aligned: list[np.ndarray],
    element_names: list[str],
    out_dir: Path,
):
    """Save a diagnostic image showing original, per-view depths, std map, and orbit average."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not available, skipping visualization)")
        return

    k = len(depths_aligned)
    stack = np.stack(depths_aligned, axis=0)
    orbit_avg = stack.mean(axis=0)
    pixel_std = stack.std(axis=0)

    # Normalize all depth maps to same scale for visualization
    all_depths = np.concatenate([d.ravel() for d in depths_aligned])
    vmin, vmax = np.percentile(all_depths, [2, 98])

    ncols = min(k + 3, 6)  # original + views + std + orbit_avg
    nrows = (k + 3 + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    axes = np.atleast_2d(axes).ravel()

    # Original image
    axes[0].imshow(original_img)
    axes[0].set_title("Original image", fontsize=9)
    axes[0].axis("off")

    # Per-view depth maps (aligned to original frame)
    for i, (depth, ename) in enumerate(zip(depths_aligned, element_names)):
        ax = axes[1 + i]
        ax.imshow(depth, cmap="inferno", vmin=vmin, vmax=vmax)
        ax.set_title(f"Depth ({ename})", fontsize=8)
        ax.axis("off")

    # Std map
    ax_std = axes[1 + k]
    im_std = ax_std.imshow(pixel_std, cmap="hot")
    ax_std.set_title("Per-pixel std", fontsize=9)
    ax_std.axis("off")
    plt.colorbar(im_std, ax=ax_std, fraction=0.046)

    # Orbit average
    ax_avg = axes[2 + k]
    ax_avg.imshow(orbit_avg, cmap="inferno", vmin=vmin, vmax=vmax)
    ax_avg.set_title("Orbit average", fontsize=9)
    ax_avg.axis("off")

    # Hide unused axes
    for j in range(3 + k, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"Warrant gap: {name}", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / f"warrant_gap_{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_dir / f'warrant_gap_{name}.png'}")


# ── Main ─────────────────────────────────────────────────────────────────

D4_ELEMENT_NAMES = [
    "identity",
    "rot90",
    "rot180",
    "rot270",
    "vflip",
    "hflip",
    "transpose",
    "anti-transpose",
]


def main():
    parser = argparse.ArgumentParser(
        description="Measure Depth Anything warrant gap under D4 symmetry"
    )
    parser.add_argument(
        "--model",
        default="depth-anything/Depth-Anything-V2-Small-hf",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--subgroup",
        default="d4",
        choices=["d4", "vflip", "flips", "rotations", "identity"],
        help="Which D4 subgroup to test",
    )
    parser.add_argument(
        "--images",
        nargs="*",
        default=None,
        help="Image paths or 'url' for default test images",
    )
    parser.add_argument("--out", default="out_depth_warrant", help="Output directory")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load images
    images = load_test_images(args.images)

    # Initialize model
    print(f"\nLoading {args.model} on {args.device}...")
    from mezzanine.encoders.depth_anything import (
        DepthAnythingEncoder,
        DepthAnythingEncoderConfig,
    )
    from mezzanine.symmetries.depth_geometric import (
        DepthGeometricSymmetry,
        DepthGeometricSymmetryConfig,
    )

    encoder = DepthAnythingEncoder(
        DepthAnythingEncoderConfig(model_name=args.model),
        device=args.device,
    )
    sym = DepthGeometricSymmetry(DepthGeometricSymmetryConfig(subgroup=args.subgroup))

    elements = sym.elements
    element_names = [D4_ELEMENT_NAMES[i] for i in elements]
    print(
        f"Symmetry subgroup: {args.subgroup} ({len(elements)} elements: {element_names})"
    )

    # Measure warrant gap per image
    all_results = []

    for img_name, img_np in images:
        print(f"\n{'=' * 60}")
        print(f"Image: {img_name} ({img_np.shape[1]}×{img_np.shape[0]})")

        # Generate all transformed views
        views = sym.batch(img_np)

        # Run depth prediction on all views (individually — transforms may change dimensions)
        depth_maps = [encoder.predict(v) for v in views]

        # Inverse-transform each depth map back to the original frame
        depths_aligned = [sym.inverse(d, i) for i, d in enumerate(depth_maps)]

        # Compute warrant gap
        metrics = compute_warrant_gap(depths_aligned)
        metrics["image"] = img_name

        print(f"  Mean per-pixel std:       {metrics['mean_std']:.4f}")
        print(f"  Mean coeff of variation:  {metrics['mean_cv']:.4f}")
        print(f"  Max per-pixel std:        {metrics['max_std']:.4f}")
        print(f"  Vertical bias (original): {metrics['vertical_bias_original']:.4f}")
        print(f"  Vertical bias (averaged): {metrics['vertical_bias_averaged']:.4f}")
        bias_reduction = abs(metrics["vertical_bias_original"]) - abs(
            metrics["vertical_bias_averaged"]
        )
        print(f"  Bias reduction:           {bias_reduction:+.4f}")

        all_results.append(metrics)

        # Save visualization
        save_diagnostics(img_name, img_np, depths_aligned, element_names, out_dir)

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    mean_cv = np.mean([r["mean_cv"] for r in all_results])
    mean_vbias_orig = np.mean([abs(r["vertical_bias_original"]) for r in all_results])
    mean_vbias_avg = np.mean([abs(r["vertical_bias_averaged"]) for r in all_results])
    print(f"  Mean warrant gap (CV):        {mean_cv:.4f}")
    print(f"  Mean |vertical bias| original: {mean_vbias_orig:.4f}")
    print(f"  Mean |vertical bias| averaged: {mean_vbias_avg:.4f}")
    print(f"  Mean bias reduction:           {mean_vbias_orig - mean_vbias_avg:.4f}")

    # Save results
    results_path = out_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "config": {
                    "model": args.model,
                    "subgroup": args.subgroup,
                    "elements": element_names,
                    "n_images": len(images),
                },
                "per_image": all_results,
                "summary": {
                    "mean_warrant_gap_cv": float(mean_cv),
                    "mean_vertical_bias_original": float(mean_vbias_orig),
                    "mean_vertical_bias_averaged": float(mean_vbias_avg),
                    "mean_bias_reduction": float(mean_vbias_orig - mean_vbias_avg),
                },
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
