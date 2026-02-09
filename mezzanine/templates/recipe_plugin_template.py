"""Recipe plugin template.

Recipes are runnable "papers in code": adapters + symmetries + encoders + pipelines.
They should write:
  - results.json
  - diagnostics plots (png)
  - (optional) montage grid / qualitative examples
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict

from mezzanine.recipes.recipe_base import Recipe


class MyRecipe(Recipe):
    NAME = "my_recipe"
    DESCRIPTION = "One-line description of the experiment."

    def run(self, argv: list[str]) -> Dict[str, Any]:
        p = argparse.ArgumentParser(prog=self.NAME)
        self.add_common_args(p)

        # TODO: add args for world/encoder/symmetry knobs
        p.add_argument("--foo", type=int, default=123)

        args = p.parse_args(argv)

        # Merge config defaults
        file_cfg = {}  # load_config(args.config) if desired
        cfg = {}       # deep_update(file_cfg, self.config) if desired

        ctx = self.build_context(args)

        # TODO: implement experiment, save artifacts into ctx.out_dir
        result: Dict[str, Any] = {"ok": True}

        (ctx.out_dir / "results.json").write_text(json.dumps(result, indent=2))
        return result
