# Mezzanine examples

These are small, runnable scripts that demonstrate the API without requiring
you to write a full recipe.

For fully reproducible end-to-end runs, prefer `mezzanine run <recipe>`.

Kernel demos:
- `python examples/cg_fused_kernel_demo.py --outdir cg_demo_out`

Ensemble warrant gap and E/F distillation (MD17 interatomic potentials):
- `python examples/mezzanine_ensemble_ef_runner.py --out runs/quick --molecule ethanol --quick`
- `examples/mezzanine_ensemble_ef_colab.ipynb` drives the same runner on Colab.

Self-contained: it needs only numpy and torch, and reimplements the ensemble
axis of the Mezzanine pattern in one file so it can be uploaded to a Colab
session on its own. Teachers are a deep ensemble of conservative SchNet-lite
potentials, the gap is member spread, and the student is a single-pass model
with forces as the energy gradient. Add `--parallel 8 --batch 1024 --hidden 128`
on an A100; at the defaults the run is kernel-launch-bound.

