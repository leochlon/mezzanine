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
    from .worlds import audio_folder  # noqa: F401
    from .worlds import finance_csv  # noqa: F401
    from .worlds import esc50  # noqa: F401
    from .worlds import urbansound8k  # noqa: F401

    # Numerical kernels / physics add-ons (pure NumPy)
    from .worlds import kepler_root_npz  # noqa: F401
    from .worlds import linear_system_npz  # noqa: F401
    from .worlds import ode_npz  # noqa: F401
    from .worlds import integration_npz  # noqa: F401
    from .worlds import eigen_npz  # noqa: F401

    # Optional PDEBench adapter (requires h5py)
    try:  # pragma: no cover
        from .worlds import pdebench_h5  # noqa: F401
    except Exception:
        pass

    # Optional: materials benchmarks (requires matbench + dependencies)
    try:  # pragma: no cover
        from .worlds import matbench_task  # noqa: F401
    except Exception:
        pass

    # Optional LJ / MD adapter (requires h5py)
    try:  # pragma: no cover
        from .worlds import lj_fluid  # noqa: F401
    except Exception:
        pass
    # Optional QG / jet adapter (requires EnergyFlow)
    try:  # pragma: no cover
        from .worlds import qg_jets  # noqa: F401
    except Exception:
        pass

    # Symmetries
    from .symmetries import view  # noqa: F401
    from .symmetries import order  # noqa: F401
    from .symmetries import factorization  # noqa: F401
    from .symmetries import action  # noqa: F401
    from .symmetries import gw_observation_lal  # noqa: F401
    from .symmetries import lj  # noqa: F401
    from .symmetries import qg as qg_symmetries  # noqa: F401
    from .symmetries import audio_playback  # noqa: F401
    from .symmetries import market_bar_offset  # noqa: F401
    from .symmetries import ens_member  # noqa: F401
    from .symmetries import field_codec  # noqa: F401

    # Numerical kernels / physics add-ons
    from .symmetries import angle_wrap  # noqa: F401
    from .symmetries import node_permutation  # noqa: F401
    from .symmetries import time_origin_shift  # noqa: F401
    from .symmetries import circular_shift  # noqa: F401
    from .symmetries import periodic_translation  # noqa: F401

    from .symmetries import depth_geometric  # noqa: F401

    try:  # pragma: no cover
        from .symmetries import space_group  # noqa: F401
    except Exception:
        pass

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
    from .encoders import qg as qg_encoders  # noqa: F401

    try:  # pragma: no cover
        from .encoders import depth_anything  # noqa: F401
    except Exception:
        pass

    # Recipes
    try:  # pragma: no cover
        from .recipes import iphyre_latent_dynamics  # noqa: F401
    except Exception:
        pass
