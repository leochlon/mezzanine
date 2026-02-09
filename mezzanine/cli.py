from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from .plugins import load_builtin_plugins
from .recipes.registry import list_recipes, run_recipe
from .registry import ADAPTERS, ENCODERS, SYMMETRIES


def _json_or_empty(s: str | None) -> Dict[str, Any]:
    if not s:
        return {}
    return json.loads(s)


def cmd_list(_: argparse.Namespace) -> None:
    rec = list_recipes()
    print(json.dumps(rec, indent=2, sort_keys=True))


def cmd_list_adapters(_: argparse.Namespace) -> None:
    print(json.dumps(ADAPTERS.list(), indent=2, sort_keys=True))


def cmd_list_encoders(_: argparse.Namespace) -> None:
    print(json.dumps(ENCODERS.list(), indent=2, sort_keys=True))


def cmd_list_symmetries(_: argparse.Namespace) -> None:
    print(json.dumps(SYMMETRIES.list(), indent=2, sort_keys=True))


def cmd_run(args: argparse.Namespace, unknown: list[str]) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    overrides = _json_or_empty(args.overrides)

    result = run_recipe(
        args.recipe,
        out_dir=out_dir,
        overrides=overrides,
        argv=unknown,
    )
    print(json.dumps(result, indent=2))


def main() -> None:
    # Ensure registries are populated
    load_builtin_plugins()

    p = argparse.ArgumentParser(prog="mezzanine")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="List available recipes")
    p_list.set_defaults(func=cmd_list)

    p_la = sub.add_parser("list-adapters", help="List registered world adapters")
    p_la.set_defaults(func=cmd_list_adapters)

    p_le = sub.add_parser("list-encoders", help="List registered encoders/backbones")
    p_le.set_defaults(func=cmd_list_encoders)

    p_ln = sub.add_parser("list-symmetries", help="List registered symmetry families")
    p_ln.set_defaults(func=cmd_list_symmetries)

    p_run = sub.add_parser("run", help="Run a recipe: mezzanine run <name> --out OUTDIR [recipe args...]")
    p_run.add_argument("recipe", type=str, help="Recipe name (see `mezzanine list`).")
    p_run.add_argument("--out", type=str, required=True, help="Output directory for artifacts/results.")
    p_run.add_argument("--overrides", type=str, default=None, help="JSON dict of config overrides applied as defaults.")

    # We intentionally pass-through all unknown args to the recipe.
    args, unknown = p.parse_known_args()
    if args.cmd == "run":
        cmd_run(args, unknown)
    else:
        args.func(args)


if __name__ == "__main__":
    main()
