# Physics add-ons

This doc covers the **real-data physics add-ons** in Mezzanine, and how to run them.

## Crystal structures / space groups (materials)

**Adapter:** `matbench_task` (optional dependency: `matbench`)

**Symmetry:** `space_group` (optional dependency: `pymatgen`)

**Recipe:** `crystal_spacegroup_distill`

### Install

From the repo root:

```bash
pip install -e ".[materials]"
```

Optional: run the teacher as a gold-standard GNN (MEGNet / M3GNet / CHGNet via `matgl`):

```bash
pip install -e ".[materials,materials_gnn]"
```

### Run (MLP teacher + MLP student)

This is the “intentionally non-invariant featurization” baseline:

```bash
mezzanine run crystal_spacegroup_distill --out out_matbench_mlp \
  --dataset_name matbench_mp_e_form --fold 0 \
  --max_atoms 64 \
  --k_train 4 --k_test 8
```

### Run (gold-standard teacher families)

Mezzanine currently keeps the **student** as an MLP on the naive padded `(Z, frac_coords)` featurization, and lets you
swap the **teacher** family:

- `--teacher_family matgl_megnet`
- `--teacher_family matgl_m3gnet`
- `--teacher_family matgl_chgnet`

Example:

```bash
mezzanine run crystal_spacegroup_distill --out out_matbench_megnet_teacher \
  --dataset_name matbench_mp_e_form --fold 0 \
  --teacher_family matgl_megnet --teacher_batch_size 32 \
  --max_atoms 64 \
  --k_train 4 --k_test 8
```

Notes:
- `matgl_*` teachers use the DGL backend (`MATGL_BACKEND=DGL`) and will run on `cpu` unless CUDA is available.
- If you hit memory issues, reduce `--teacher_batch_size` and/or `--n_train`.

### Outputs

In `--out`:
- `results.json`

---

## Numerical kernels (toy)

These experiments are **small, fast, and self-contained**. They demonstrate the same Mezzanine pattern (“distill the expectation”) on numerical tasks with exact symmetries:

- **Teacher orbit-averaging**: costs **K forward passes per example** (one per symmetry view).
- **Student distillation**: learns to match the orbit-averaged target in **one forward pass**.

### Generate toy datasets

From the repo root:

```bash
python examples/kepler_generate_dataset.py --out data/kepler_root_toy.npz --n_train 50000 --n_test 10000
python examples/linear_system_generate_dataset.py --out data/linear_system_toy.npz --n_train 50000 --n_test 10000
python examples/ode_generate_dataset.py --out data/ode_lorenz_toy.npz --system lorenz --n_traj 400 --t_max 40 --dt 0.01
python examples/integration_generate_dataset.py --out data/integration_toy.npz --n_train 50000 --n_test 10000 --n_grid 128
python examples/eigen_generate_dataset.py --out data/eigen_toy.npz --n_train 20000 --n_test 5000 --n 64 --density 0.05 --k 5
```

Notes:
- The dataset generators accept a few legacy flags (e.g. `--n_grid`, `--n_traj`, `--t_max`, `--density`, `--spd`) so older notebooks still run.

### Run (recipes)

All recipes take `--dataset` and write `results.json` to `--out`:

```bash
mezzanine run kepler_root_distill --out out_kepler --dataset data/kepler_root_toy.npz --k_train 4 --k_test 16
mezzanine run linear_system_permutation_distill --out out_linear --dataset data/linear_system_toy.npz --k_train 4 --k_test 16
mezzanine run ode_time_origin_distill --out out_ode --dataset data/ode_lorenz_toy.npz --teacher_include_time --k_train 4 --k_test 16
mezzanine run integration_circular_shift_distill --out out_integration --dataset data/integration_toy.npz --k_train 4 --k_test 16
mezzanine run eigen_permutation_distill --out out_eigen --dataset data/eigen_toy.npz --k_train 4 --k_test 16
```

### Symmetry-aware kernel demo (Triton)

Run the fused Clebsch–Gordan contraction demo:

```bash
python examples/cg_fused_kernel_demo.py --B 4096 --I 256 --J 256 --O 256 --P 32 --dtype bf16 --outdir cg_demo_out
```

### PDEBench (optional: requires `h5py`)

PDEBench uses an HDF5 adapter (`pdebench_h5`). Install `h5py` (or `pip install -e \".[md]\"`) and run:

```bash
mezzanine run pdebench_translation_distill --out out_pdebench \
  --dataset data/pdebench_toy.h5 \
  --train_group train --test_group test \
  --x_key u0 --y_key u1 \
  --axes 0 --max_shift 8 \
  --k_train 4 --k_test 16
```

### Visuals (GIF hero tiles)

The helper in `numerical_visualiser/generate_hero_gifs.py` can generate the six experiment GIFs and a tiled animated hero, given:
- toy datasets in `data/`
- recipe outputs (each with `results.json`) in a runs directory

See the script’s `--help` for the expected folder layout.
