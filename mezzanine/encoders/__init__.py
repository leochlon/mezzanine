from .base import Encoder
from .hf_language import HFLanguageEncoder, HFLanguageEncoderConfig

__all__ = [
    "Encoder",
    "HFLanguageEncoder","HFLanguageEncoderConfig",
]


# LJ / MD encoders rely on SciPy (for KD-trees). Keep optional.
try:  # pragma: no cover
    from .lj import LJFlattenEncoder, LJFlattenEncoderConfig, LJRDFEncoder, LJRDFEncoderConfig
    __all__ += ["LJFlattenEncoder","LJFlattenEncoderConfig","LJRDFEncoder","LJRDFEncoderConfig"]
except Exception:
    pass

# Vision encoders are optional (they may require torchvision / image backends).
try:  # pragma: no cover
    from .hf_vision import HFVisionEncoder, HFVisionEncoderConfig  # type: ignore
    __all__ += ["HFVisionEncoder","HFVisionEncoderConfig"]
except Exception:
    pass

try:  # pragma: no cover
    from .hf_clip import HFCLIPVisionEncoder, HFCLIPVisionEncoderConfig  # type: ignore
    __all__ += ["HFCLIPVisionEncoder","HFCLIPVisionEncoderConfig"]
except Exception:
    pass

try:  # pragma: no cover
    from .hf_dino import HFDINOv2Encoder, HFDINOv2EncoderConfig  # type: ignore
    __all__ += ["HFDINOv2Encoder","HFDINOv2EncoderConfig"]
except Exception:
    pass

try:  # pragma: no cover
    from .hf_causal_lm import HFCausalLMEncoder, HFCausalLMEncoderConfig  # type: ignore
    __all__ += ["HFCausalLMEncoder","HFCausalLMEncoderConfig"]
except Exception:
    pass
