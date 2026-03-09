"""Depth Anything V2 encoder — outputs dense depth maps (H, W) not embeddings.

This wraps the HuggingFace Depth Anything V2 model for use with
Mezzanine's symmetry measurement and distillation recipes.

Uses AutoImageProcessor.post_process_depth_estimation() for proper
interpolation back to original image size, following the official HF docs.

Available models (pass as model_name):
  - depth-anything/Depth-Anything-V2-Small-hf  (25M params)
  - depth-anything/Depth-Anything-V2-Base-hf   (98M params)
  - depth-anything/Depth-Anything-V2-Large-hf  (335M params)

Usage:
    encoder = DepthAnythingEncoder(DepthAnythingEncoderConfig())
    depth_map = encoder.predict(image_hw3)  # -> (H, W) float32
    depth_maps = encoder.predict_batch(images)  # -> list of (H, W)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List

import numpy as np
import torch
from PIL import Image

from ..core.cache import hash_dict
from ..registry import ENCODERS
from .base import Encoder


@dataclass
class DepthAnythingEncoderConfig:
    model_name: str = "depth-anything/Depth-Anything-V2-Small-hf"
    batch_size: int = 8
    fp16: bool = True


class DepthAnythingEncoder(Encoder):
    """Depth Anything V2 — frozen monocular depth estimation.

    Unlike HFVisionEncoder, this produces dense (H, W) depth maps,
    not pooled embedding vectors.  The `encode` method is kept for
    compatibility but `predict` / `predict_batch` are the primary API.

    Depth maps are relative (not metric): higher values = farther.
    The raw output is not normalized — use normalize_depth() if needed.
    """

    NAME = "depth_anything"
    DESCRIPTION = "Depth Anything V2 monocular depth (dense H×W output)."

    def __init__(
        self, cfg: DepthAnythingEncoderConfig | None = None, device: str = "cuda"
    ):
        if cfg is None:
            cfg = DepthAnythingEncoderConfig()
        self.cfg = cfg
        self.device = device

        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.processor = AutoImageProcessor.from_pretrained(cfg.model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(cfg.model_name).to(
            device
        )
        self.model.eval()

    def fingerprint(self) -> str:
        d = asdict(self.cfg)
        d["__class__"] = self.__class__.__name__
        return hash_dict(d)

    def predict(self, img: np.ndarray | Image.Image) -> np.ndarray:
        """Predict depth for a single image.

        Args:
            img: (H, W, 3) uint8 numpy array or PIL Image

        Returns:
            (H, W) float32 relative depth map (higher = farther),
            interpolated back to the original image resolution.
        """
        return self.predict_batch([img])[0]

    def predict_batch(self, imgs: List[np.ndarray | Image.Image]) -> List[np.ndarray]:
        """Predict depth for a batch of images.

        Uses post_process_depth_estimation() for proper interpolation
        back to each image's original size.

        Returns list of (H, W) float32 depth maps.
        """
        results: List[np.ndarray] = []
        bs = self.cfg.batch_size

        for i in range(0, len(imgs), bs):
            chunk = imgs[i : i + bs]
            pil_imgs = []
            for x in chunk:
                if isinstance(x, np.ndarray):
                    pil_imgs.append(Image.fromarray(x))
                else:
                    pil_imgs.append(x)

            target_sizes = [(img.height, img.width) for img in pil_imgs]

            inputs = self.processor(images=pil_imgs, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                with torch.amp.autocast(
                    "cuda", enabled=(self.cfg.fp16 and self.device.startswith("cuda"))
                ):
                    outputs = self.model(**inputs)

            # Use the official post-processing which handles interpolation
            post_processed = self.processor.post_process_depth_estimation(
                outputs, target_sizes=target_sizes
            )

            for item in post_processed:
                depth_tensor = item["predicted_depth"]  # (H, W) tensor
                depth_np = depth_tensor.detach().cpu().float().numpy()
                results.append(depth_np)

        return results

    def encode(self, imgs: List[np.ndarray]) -> np.ndarray:
        """Compatibility with Encoder base class — returns flattened depth maps.

        For most use cases, prefer predict_batch() which returns proper 2D maps.
        """
        depths = self.predict_batch(imgs)
        # Flatten to (N, H*W) for compatibility
        return np.stack([d.ravel() for d in depths], axis=0).astype(np.float32)


def normalize_depth(depth: np.ndarray) -> np.ndarray:
    """Min-max normalize a depth map to [0, 1]."""
    dmin, dmax = depth.min(), depth.max()
    if dmax - dmin < 1e-8:
        return np.zeros_like(depth)
    return (depth - dmin) / (dmax - dmin)


# Register
ENCODERS.register("depth_anything")(DepthAnythingEncoder)
