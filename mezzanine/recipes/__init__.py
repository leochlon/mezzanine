"""Recipe package.

Recipes are runnable end-to-end experiments.
Prefer `mezzanine run <recipe>` for reproducibility.
"""

from .recipe_base import Recipe
from .registry import list_recipes, get_recipe, run_recipe

__all__ = ["Recipe", "list_recipes", "get_recipe", "run_recipe"]
