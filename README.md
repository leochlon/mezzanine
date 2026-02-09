# Mezzanine (v1.0)

Mezzanine is a small research toolkit for **measuring instability** (the *warrant gap*) and for **distilling symmetry‑marginalized invariants** (“distill the expectation”) into a single forward pass.

> World models ≠ pixel prediction.  
> Warranted inference ≠ maximum likelihood on a single realization.

## What you get in v1.0

### Registries (discoverability + plugins)
- **Adapters registry**: HuggingFace Datasets, LeRobot (HF robotics datasets), Gymnasium, I‑PHYRE
- **Symmetries registry**: view / order / factorization / action‑shuffle
- **Backbone registry**: HF vision (I‑JEPA/ViT/DINOv2/…), CLIP vision, DINOv2, HF language encoders
- **Recipes registry**: runnable end‑to‑end presets (start with I‑PHYRE)

### Caching (latents to disk)
Latents are cached to disk keyed by:
- **world fingerprint** (dataset config + deterministic subsampling)
- **encoder fingerprint** (checkpoint + pooling/layer config)

This makes experiments cheap to re-run and easy to reproduce/share.

### Auto‑tuning (“hard but not dead” pilots)
A generic `AutoTuner` is included (used by recipes as needed) to search for regimes that:
- are not trivially easy (no signal)
- are not impossible (dead)
- maximize the effect size you care about (e.g., action helps vs no‑action)

### Reproducible configs
- YAML/JSON config loading
- deterministic subsampling
- global seeding helpers

### Optional logging (never required)
- `--log wandb` (if installed)
- `--log tensorboard` (if installed)
- default is no-op logging

---

## Install

Minimal:
```bash
pip install -e .
```

With YAML configs:
```bash
pip install -e ".[yaml]"
```

With i‑PHYRE:
```bash
pip install -e ".[iphyre]"
```

Everything:
```bash
pip install -e ".[all]"
```

---

## CLI quickstart

List runnable recipes:
```bash
mezzanine list
```

List built-in components:
```bash
mezzanine list-adapters
mezzanine list-encoders
mezzanine list-symmetries
```

Run the I‑PHYRE latent dynamics distillation (physics puzzles):
```bash
mezzanine run iphyre_latent_dynamics --out out_iphyre \
  --games hole,seesaw \
  --n_train 3000 --n_test 768 \
  --delta_seconds 4.0 \
  --embed_mode mean_std --embed_layer -4
```

Add caching:
```bash
mezzanine run iphyre_latent_dynamics --out out_iphyre \
  --cache_dir ~/.cache/mezzanine_latents \
  --games hole,seesaw
```

Use a config file as defaults, override on CLI:
```bash
mezzanine run iphyre_latent_dynamics --out out_iphyre \
  --config configs/iphyre.yml \
  --delta_seconds 2.0
```

Enable W&B logging (optional):
```bash
mezzanine run iphyre_latent_dynamics --out out_iphyre \
  --log wandb --wandb_project mezzanine \
  --games hole,seesaw
```

Outputs (in `--out`):
- `results.json`
- `diagnostics.png`
- `montage.png`

---

## Notes on extending

### Add a new adapter
Create `mezzanine/worlds/my_world.py` and register:
```python
from mezzanine.registry import ADAPTERS
@ADAPTERS.register("my_world")
class MyWorldAdapter(WorldAdapter):
    ...
```

### Add a new symmetry
Create `mezzanine/symmetries/my_symmetry.py` and register:
```python
from mezzanine.registry import SYMMETRIES
@SYMMETRIES.register("my_symmetry")
class MySymmetry(Symmetry):
    ...
```

### Add a new backbone/encoder
Create `mezzanine/encoders/my_encoder.py` and register:
```python
from mezzanine.registry import ENCODERS
@ENCODERS.register("my_encoder")
class MyEncoder(Encoder):
    ...
```


### Text / LLM recipe: sentence-order symmetry distillation

This recipe demonstrates the *same* "distill the expectation" pattern in language:

```bash
mezzanine run hf_text_order_distill --out out_text \
  --dataset ag_news --n_train 5000 --n_test 2000 \
  --model_name distilbert-base-uncased \
  --k_train 8 --k_test 16
```

It measures how much a classifier's predictions change when you shuffle the order of sentences,
and then distills the symmetry-marginalized teacher into a single-pass student head.

See also:
- `docs/plugin_api.md` (minimal plugin API + templates)
- `docs/llm_examples.md` (LLM-oriented usage patterns)


- **NEW (v1.2):** `hf_llm_hiddenstate_order_distill` — logits→hidden-state distillation for order-invariant BoolQ.
