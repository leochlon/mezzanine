from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def save_diagnostics(result: Dict[str, Any], out_path: Path) -> None:
    cos = result["metrics"]["cos"]
    rank_mean = result["metrics"]["rank"]["mean"]
    r10 = result["metrics"]["rank"]["r@10"]

    labels = ["persist", "no_action", "action", "action_shuf"]

    fig = plt.figure(figsize=(12, 4))

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.bar(range(len(labels)), [cos[k] for k in labels])
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, rotation=20)
    ax1.set_ylim(0.0, 1.0)
    ax1.set_title("Mean cosine(pred, true)")

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.bar(range(len(labels)), [rank_mean[k] for k in labels])
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=20)
    ax2.set_title("Mean retrieval rank (lower is better)")

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.bar(range(len(labels)), [r10[k] for k in labels])
    ax3.set_xticks(range(len(labels)))
    ax3.set_xticklabels(labels, rotation=20)
    ax3.set_ylim(0.0, 1.0)
    ax3.set_title("Retrieval R@10 (higher is better)")

    fig.suptitle(f"Latent dynamics distillation — {result['make_break']['verdict']}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_montage(samples_test: List[Dict[str, Any]], out_path: Path, n_rows: int = 8, thumb: int = 192) -> None:
    rng = np.random.default_rng(0)
    n_rows = min(n_rows, len(samples_test))
    idxs = rng.choice(len(samples_test), size=n_rows, replace=False)

    cols = ["obs_t", "true_{t+Δ}"]
    W = thumb
    H = thumb
    pad = 8
    title_h = 18

    canvas = Image.new("RGB", (pad + len(cols) * (W + pad), pad + n_rows * (H + pad) + title_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for c, name in enumerate(cols):
        draw.text((pad + c * (W + pad), pad), name, fill=(0, 0, 0), font=font)

    y = pad + title_h
    for r, i in enumerate(idxs):
        s = samples_test[int(i)]
        im0 = Image.fromarray(s["img_t"]).resize((W, H))
        im1 = Image.fromarray(s["img_tp"]).resize((W, H))
        canvas.paste(im0, (pad + 0 * (W + pad), y))
        canvas.paste(im1, (pad + 1 * (W + pad), y))
        # row label
        lab = f"{s.get('game','')}"
        draw.text((pad + 2, y + 2), lab, fill=(255, 255, 255), font=font)
        y += H + pad

    canvas.save(out_path)
