from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml  # type: ignore
    _HAS_YAML = True
except Exception:
    _HAS_YAML = False


def load_config(path: Optional[str]) -> Dict[str, Any]:
    """Load JSON/YAML config. Returns {} if path is None."""
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    txt = p.read_text()
    suf = p.suffix.lower()
    if suf in [".yaml", ".yml"]:
        if not _HAS_YAML:
            raise RuntimeError("YAML config requested but PyYAML not installed. Install: pip install mezzanine[yaml]")
        return yaml.safe_load(txt) or {}
    return json.loads(txt)


def save_config(path: str | Path, cfg: Dict[str, Any]) -> None:
    p = Path(path)
    suf = p.suffix.lower()
    if suf in [".yaml", ".yml"]:
        if not _HAS_YAML:
            raise RuntimeError("YAML output requested but PyYAML not installed. Install: pip install mezzanine[yaml]")
        p.write_text(yaml.safe_dump(cfg, sort_keys=False))
        return
    p.write_text(json.dumps(cfg, indent=2, sort_keys=True))


def deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    """Recursive dict update (upd overrides base)."""
    out = dict(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)  # type: ignore[arg-type]
        else:
            out[k] = v
    return out
