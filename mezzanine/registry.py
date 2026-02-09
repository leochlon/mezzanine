from __future__ import annotations

from typing import TYPE_CHECKING, Type

from .core.registry import Registry

if TYPE_CHECKING:
    from .worlds.base import WorldAdapter
    from .symmetries.base import Symmetry
    from .encoders.base import Encoder

# Public registries (runtime-typed to avoid import cycles)
ADAPTERS: Registry[Type["WorldAdapter"]] = Registry(kind="adapter")
SYMMETRIES: Registry[Type["Symmetry"]] = Registry(kind="symmetry")
ENCODERS: Registry[Type["Encoder"]] = Registry(kind="encoder")

# Recipes remain in mezzanine.recipes.registry to keep CLI stable.
