# Mezzanine (v1.0)

Mezzanine is a small research toolkit for **distilling symmetry‑marginalized invariants** (“distill the expectation”), and for **measuring instability** (the *warrant gap*) and **distilling symmetry‑marginalized invariant world models** into a single forward pass.

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

With quark/gluon jets (EnergyFlow):
```bash
pip install -e ".[qg]"
pip install pillow  # only needed for --make_gif
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

## Particle physics: quark/gluon jets (EnergyFlow)

This recipe measures the warrant gap under **particle permutation + internal SO(2) rotations**
and distills a **single‑pass student** that approximates the symmetry‑marginalized predictor.

Run:
```bash
mezzanine run qg_jets_distill --out out_qg \
  --encoder qg_flatten \
  --num_data 50000 --n_train 20000 --n_test 5000 \
  --max_particles 64 \
  --k_train 8 --k_test 16 \
  --theta_max 6.283185307179586 \
  --hard_label_weight 0.1 \
  --make_gif --gif_bins 180 --gif_extent 3.2 --gif_ms 90
```

Notes:
- `--cache_dir` is for the **latent cache**; dataset caching is `--ef_cache_dir` (defaults to `~/.energyflow`).
- If the GIF looks empty, increase `--gif_extent` (typical `phi` range is about `[-π, π]`).

Outputs (in `--out`):
- `results.json`
- `probs_test_views.npz`
- `jet_nuisance.gif` (only if `--make_gif`)

---

## Robotics: LeRobot latent dynamics + planning (multiple morphologies)

This recipe mirrors the **same latent-dynamics distillation + action-shuffle counterfactual**
as `iphyre_latent_dynamics`, and adds a lightweight downstream demo:
**goal-conditioned action retrieval** (select the action that reaches a target latent state).

It also includes an optional **V-JEPA 2-AC-style 1-step CEM planning objective** in latent space
as an offline proxy.

### PushT (image; single-arm manipulation)
```bash
mezzanine run lerobot_latent_dynamics --out out_pusht \
  --repo_id lerobot/pusht_image \
  --camera_key observation.image \
  --action_key action \
  --n_train 4000 --n_test 2000 --delta_steps 1 \
  --do_planning --plan_candidates 32 --plan_eval 512 \
  --do_cem --cem_eval 128
```

### ALOHA Sim (image; bimanual manipulation)
```bash
mezzanine run lerobot_latent_dynamics --out out_aloha_sim \
  --repo_id lerobot/aloha_sim_transfer_cube_scripted_image \
  --camera_key observation.images.top \
  --action_key action \
  --n_train 4000 --n_test 2000 --delta_steps 1 \
  --do_planning
```

### LIBERO-10 (image; multi-task manipulation)
```bash
mezzanine run lerobot_latent_dynamics --out out_libero \
  --repo_id lerobot/libero_10_image_subtask \
  --camera_key observation.images.image \
  --action_key action \
  --n_train 4000 --n_test 2000 --delta_steps 1 \
  --do_planning
```

**Make/Break condition:** all latent-dynamics recipes use the same criterion:
action must improve retrieval (vs no-action) and action-shuffle must hurt, using mean-rank/R@10 thresholds.

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


### Finance recipe: bar-offset symmetry distillation (CSV)

This recipe uses a simple tabular head on return-window features, measures the *bar-offset* warrant gap, then distills
the symmetry-marginalized teacher back into a single-pass student:

The repo includes a small public OHLCV sample CSV for offline runs/tests:
- `examples/data/spy_daily_ohlcv.csv` — SPY.US daily bars (Stooq export), columns `Date,Open,High,Low,Close,Volume` (2000 rows)

```bash
mezzanine run finance_csv_bar_offset_distill --out out_finance \
  --path examples/data/spy_daily_ohlcv.csv \
  --timestamp_col Date --close_col Close \
  --n_train 1400 --n_test 400 \
  --lookback 32 --max_offset 1 --trend_lookback 128 \
  --k_train 8 --k_test 16
```

Outputs (in `--out`):
- `results.json` (includes a `make_break` verdict)
- `diagnostics.png`
