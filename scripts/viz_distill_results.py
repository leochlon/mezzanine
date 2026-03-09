#!/usr/bin/env python3
"""Visualize before/after depth maps from distillation training.

Generates side-by-side comparisons: original image | teacher depth | student depth
for selected eval images.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


def load_model(path: str, device: str):
    processor = AutoImageProcessor.from_pretrained(path)
    model = AutoModelForDepthEstimation.from_pretrained(path).to(device).eval()
    return processor, model


def predict_depth(processor, model, img_np, device):
    inputs = processor(images=img_np, return_tensors="pt", keep_aspect_ratio=False).to(
        device
    )
    with torch.no_grad():
        out = model(**inputs)
    depth = out.predicted_depth.squeeze().cpu().numpy()
    return depth


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--teacher", default="depth-anything/Depth-Anything-V2-Small-hf"
    )
    parser.add_argument("--student", required=True, help="Path to student model dir")
    parser.add_argument("--eval-dir", required=True, help="Directory with eval images")
    parser.add_argument("--out", default="viz_results", help="Output directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--images", nargs="*", help="Specific image stems to visualize (default: all)"
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading teacher model...")
    t_proc, t_model = load_model(args.teacher, args.device)
    print("Loading student model...")
    s_proc, s_model = load_model(args.student, args.device)

    eval_dir = Path(args.eval_dir)
    image_files = sorted(list(eval_dir.glob("*.jpg")) + list(eval_dir.glob("*.png")))

    if args.images:
        image_files = [f for f in image_files if f.stem in args.images]

    print(f"Visualizing {len(image_files)} images...")

    for img_path in image_files:
        img = np.asarray(Image.open(img_path).convert("RGB"))
        name = img_path.stem

        teacher_depth = predict_depth(t_proc, t_model, img, args.device)
        student_depth = predict_depth(s_proc, s_model, img, args.device)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(img)
        axes[0].set_title(f"{name}\n(original)")
        axes[0].axis("off")

        axes[1].imshow(teacher_depth, cmap="inferno")
        axes[1].set_title("Teacher (pretrained)")
        axes[1].axis("off")

        axes[2].imshow(student_depth, cmap="inferno")
        axes[2].set_title("Student (distilled)")
        axes[2].axis("off")

        plt.tight_layout()
        save_path = out_dir / f"{name}.png"
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"  Saved {save_path}")

    print(f"\nDone! {len(image_files)} visualizations in {out_dir}/")


if __name__ == "__main__":
    main()
