# Mezzanine plugin API (minimal)

Mezzanine is built around **registries** so that experiments are composable:

- **Adapters**: load a *world* (dataset/environment) into standardized examples.
- **Symmetries**: generate alternative *views* of the same underlying instance (view/order/factorization/action-shuffle).
- **Encoders**: map raw inputs to embeddings (JEPA/CLIP/DINO, HF language models, …).
- **Recipes**: runnable end-to-end experiments ("paper figures in code").

The core design goal is that new worlds/symmetries/encoders can be added **without touching the core**:
just implement a small class and register it.

## 1) Adapters (worlds)

**Where:** `mezzanine/worlds/`

**Base class:** `mezzanine.worlds.base.WorldAdapter`

**Required methods**
- `fingerprint(self) -> str`  
  Return a *stable* hash of everything that changes the realized data distribution.
- `load(self) -> dict`  
  Return `{"train": [...], "test": [...], "meta": {...}}`.

**Example shape**
- supervised text: `{"text": str, "label": int}`
- supervised vision: `{"image": np.ndarray(H,W,3), "label": int}`
- dynamics: `{"obs_t": ..., "obs_tp": ..., "action": ...}`

**Registration**
```python
from mezzanine.registry import ADAPTERS

@ADAPTERS.register("my_adapter")
class MyAdapter(WorldAdapter):
    ...
```

Templates: `mezzanine/templates/adapter_plugin_template.py`

## 2) Symmetries

**Where:** `mezzanine/symmetries/`

**Base class:** `mezzanine.symmetries.base.Symmetry`

**Required method**
- `sample(self, x, *, seed: int)`  
  Return a single symmetry-sampled variant of `x`.

**Registration**
```python
from mezzanine.registry import SYMMETRIES

@SYMMETRIES.register("my_symmetry")
class MySymmetry(Symmetry):
    ...
```

Templates: `mezzanine/templates/symmetry_plugin_template.py`

## 3) Encoders / backbones

**Where:** `mezzanine/encoders/`

**Base class:** `mezzanine.encoders.base.Encoder`

**Required methods**
- `fingerprint(self) -> str`
- `encode(self, inputs: list[...]) -> np.ndarray` (shape `[N, D]`, float32)

**Registration**
```python
from mezzanine.registry import ENCODERS

@ENCODERS.register("my_encoder")
class MyEncoder(Encoder):
    ...
```

Templates: `mezzanine/templates/encoder_plugin_template.py`

## 4) Recipes

**Where:** `mezzanine/recipes/`

**Base class:** `mezzanine.recipes.recipe_base.Recipe`

**Pattern**
- parse CLI args
- build `RunContext` (`cache`, `logger`, `seed`, output dir)
- glue together adapters + symmetries + encoders + pipelines
- write `results.json` + at least one diagnostic figure

To make a recipe discoverable in v1.x, add it to `mezzanine/recipes/registry.py`.

Templates: `mezzanine/templates/recipe_plugin_template.py`

## 5) Best practices (so others can reproduce you)

- Put *every* randomness source under `--seed`.
- Use deterministic subsampling (`mezzanine.core.deterministic`).
- Make your adapter/encoder fingerprints reflect all knobs.
- Cache expensive encodings (`LatentCache`) keyed by those fingerprints.
- Write your experiment as a recipe so anyone can re-run from CLI.
