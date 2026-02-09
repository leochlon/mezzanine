from __future__ import annotations

from dataclasses import dataclass

from .hf_vision import HFVisionEncoder, HFVisionEncoderConfig


@dataclass
class HFDINOv2EncoderConfig(HFVisionEncoderConfig):
    model_name: str = "facebook/dinov2-base"
    # DINOv2 often uses CLS token; but mean_std with earlier layer can help for 2D scenes.


class HFDINOv2Encoder(HFVisionEncoder):
    NAME = "dinov2"
    DESCRIPTION = "DINOv2 vision backbone (HF) with configurable pooling/layer."

    def __init__(self, cfg: HFDINOv2EncoderConfig, device: str = "cuda"):
        super().__init__(cfg, device=device)


# Register
from ..registry import ENCODERS
ENCODERS.register('dinov2')( HFDINOv2Encoder )
