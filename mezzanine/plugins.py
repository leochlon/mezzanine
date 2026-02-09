"""Explicit imports that populate the registries.

Mezzanine avoids "magic" entrypoints for v1.0: if you want plugins, you can
import them explicitly (or later we can add setuptools entrypoints).
"""


def load_builtin_plugins() -> None:
    # Worlds
    from .worlds import iphyre  # noqa: F401
    from .worlds import hf_dataset  # noqa: F401
    from .worlds import hf_qa  # noqa: F401
    from .worlds import lerobot  # noqa: F401
    from .worlds import gymnasium  # noqa: F401
    from .worlds import gw_merger_lal  # noqa: F401

    # Optional LJ / MD adapter (requires h5py)
    try:  # pragma: no cover
        from .worlds import lj_fluid  # noqa: F401
    except Exception:
        pass

    # Symmetries
    from .symmetries import view  # noqa: F401
    from .symmetries import order  # noqa: F401
    from .symmetries import factorization  # noqa: F401
    from .symmetries import action  # noqa: F401
    from .symmetries import gw_observation_lal  # noqa: F401
    from .symmetries import lj  # noqa: F401

    # Encoders
    try:  # pragma: no cover
        from .encoders import hf_vision  # noqa: F401
    except Exception:
        pass
    from .encoders import hf_clip  # noqa: F401
    from .encoders import hf_dino  # noqa: F401
    from .encoders import hf_language  # noqa: F401
    from .encoders import hf_causal_lm  # noqa: F401

    # Optional LJ / MD encoders (requires SciPy)
    try:  # pragma: no cover
        from .encoders import lj  # noqa: F401
    except Exception:
        pass

    # Recipes
    try:  # pragma: no cover
        from .recipes import iphyre_latent_dynamics  # noqa: F401
    except Exception:
        pass
