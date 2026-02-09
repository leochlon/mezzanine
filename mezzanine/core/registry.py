from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, Iterable, Iterator, Optional, Tuple, Type, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class RegistryItem(Generic[T]):
    name: str
    obj: T
    description: str = ""


class Registry(Generic[T]):
    """A small, explicit registry for pluggable components.

    Mezzanine uses registries for:
      - world adapters
      - symmetry families
      - encoders/backbones
      - recipes (thin runnable presets)

    Why a registry?
      - makes CLI discovery easy (`mezzanine list-*`)
      - avoids hidden imports / "magic" plugin systems
      - keeps reproducibility (names map to code objects)
    """

    def __init__(self, *, kind: str):
        self.kind = kind
        self._items: Dict[str, RegistryItem[T]] = {}

    def register(self, name: Optional[str] = None, *, description: str = "") -> Callable[[T], T]:
        def _decorator(obj: T) -> T:
            key = name or getattr(obj, "NAME", None) or getattr(obj, "__name__", None)
            if not key:
                raise ValueError(f"Cannot infer registry name for {obj!r}")
            if key in self._items:
                raise KeyError(f"Duplicate {self.kind} registration: {key}")
            self._items[key] = RegistryItem(name=key, obj=obj, description=description or getattr(obj, "DESCRIPTION", ""))
            return obj
        return _decorator

    def get(self, name: str) -> T:
        if name not in self._items:
            raise KeyError(f"Unknown {self.kind}: {name}. Available: {sorted(self._items.keys())}")
        return self._items[name].obj

    def items(self) -> Dict[str, RegistryItem[T]]:
        return dict(self._items)

    def list(self) -> Dict[str, Dict[str, Any]]:
        return {k: {"description": v.description, "object": getattr(v.obj, "__name__", str(v.obj))} for k, v in self._items.items()}

    def __contains__(self, name: str) -> bool:
        return name in self._items

    def __iter__(self) -> Iterator[Tuple[str, RegistryItem[T]]]:
        for k, v in self._items.items():
            yield k, v
