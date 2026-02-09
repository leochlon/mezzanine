from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from .recipe_base import Recipe

# Recipes may depend on optional backends (vision/physics/robotics libs). We keep
# the registry import-tolerant so `import mezzanine` works in minimal setups.
_REGISTRY: Dict[str, Type[Recipe]] = {}

def _try_register(cls: Type[Recipe]) -> None:
    _REGISTRY[cls.NAME] = cls

try:  # pragma: no cover
    from .iphyre_latent_dynamics import IPhyreLatentDynamicsRecipe
    _try_register(IPhyreLatentDynamicsRecipe)
except Exception:
    pass

try:  # pragma: no cover
    from .hf_text_order_distill import HFTextOrderDistillRecipe
    _try_register(HFTextOrderDistillRecipe)
except Exception:
    pass

try:  # pragma: no cover
    from .hf_llm_hiddenstate_order_distill import HFLLMHiddenStateOrderDistill
    _try_register(HFLLMHiddenStateOrderDistill)
except Exception:
    pass

try:  # pragma: no cover
    from .gw_smbh_viz import GWSMBHVizRecipe
    _try_register(GWSMBHVizRecipe)
except Exception:
    pass


try:  # pragma: no cover
    from .lj_fluid_distill import LJFluidDistillRecipe
    _try_register(LJFluidDistillRecipe)
except Exception:
    pass



def list_recipes() -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for k, cls in _REGISTRY.items():
        out[k] = {"description": getattr(cls, "DESCRIPTION", ""), "class": cls.__name__}
    return out


def get_recipe(name: str) -> Type[Recipe]:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown recipe: {name}. Available: {', '.join(sorted(_REGISTRY.keys()))}")
    return _REGISTRY[name]


def run_recipe(
    name: str,
    *,
    out_dir: Path,
    overrides: Optional[Dict[str, Any]] = None,
    argv: Optional[list[str]] = None,
) -> Dict[str, Any]:
    cls = get_recipe(name)
    recipe = cls(out_dir=out_dir, config=overrides or {})
    return recipe.run(argv=argv or [])
