#!/usr/bin/env python3
"""
mezzanine_ensemble_ef_runner.py
================================
Ensemble-member warrant gap + distillation for ENERGY AND FORCE interatomic
potentials, on real data, in one file.  Mezzanine pattern, ensemble axis:

    views      = members of a deep ensemble       (mezzanine.symmetries.ens_member)
    teacher    = ensemble mean E and mean F        (orbit-averaged soft target)
    gap        = spread across members             (warrant_gap_regression, verbatim maths)
    student    = single-pass MLIP with F = -dE/dR  (train_regressor_distill pattern,
                                                    hard_label_weight knob)

Pipeline
--------
 1. Load a real E/F dataset: MD17 (auto-download, --molecule) or --npz (rMD17 / yours).
 2. Train M teachers (SchNet-lite invariant GNN, conservative forces) on n_train
    DFT-labelled configs with different seeds/splits.
 3. Sweep the energy/force loss weight on a single model -> the E-vs-F trade-off,
    plotted as a Pareto curve (this is the "they don't learn well together" claim).
 4. Ensemble warrant gap on E and F (test set).
 5. Distil a student to the ensemble-mean E and F.  The teachers also label
    n_unlabeled extra configs for free (that is where distillation wins; set
    --n_unlabeled 0 for the pure-Mezzanine setting).
 6. Numerical SO(3) x permutation equivariance check of student forces via
    frame un-transformation  g^-1 f(g x)   (the operation the crystal recipes lack).
 7. Short NVE MD with single / ensemble / student: energy drift + blow-up detection
    (the test that distinguishes an E/F-consistent model from one with good MAEs).

Outputs (--out)
---------------
 results.json            Mezzanine-schema (teacher / student / distill / make_break ...)
 summary.csv             one row per model
 predictions_test.npz    every prediction on the test set, for your own plots
 fig_pareto.png  fig_parity.png  fig_gap.png  fig_md.png  fig_train.png

Colab
-----
 !pip -q install matplotlib                      # torch is preinstalled
 # upload this file (Files pane, or google.colab.files.upload()), then:
 !python mezzanine_ensemble_ef_runner.py --out runs/ethanol --molecule ethanol
 from IPython.display import Image, display
 for f in ["fig_pareto","fig_parity","fig_gap","fig_md","fig_train"]:
     display(Image(f"runs/ethanol/{f}.png"))

 ~6-10 min on a T4 at defaults (10 model fits).  --quick for a ~2 min smoke test.
 MAEs at the default budget sit above published SchNet numbers; raise --steps.
 Aspirin/paracetamol (21/20 atoms) take ~2x ethanol.

GPU utilisation
---------------
 At --batch 32 on a 9-atom molecule each step is microseconds of arithmetic; the
 run is kernel-launch-bound and ANY GPU shows single-digit utilisation.  Two knobs:
   --batch / --hidden / --n_int   make each kernel do real work (free on an A100)
   --parallel N                   train the N independent fits (members + sweep)
                                  concurrently in spawned processes on one GPU
 A100 recipe (~4 min wall clock):
   --batch 1024 --hidden 128 --n_int 4 --steps 6000 --members 8 --parallel 8 \
   --n_unlabeled 8000 --n_test 2000
 --tf32 trades ~1e-3 force precision for speed; the equivariance criterion relaxes.

Porting into mezzanine
----------------------
 warrant_gap_regression / the soft+hard loss are copied 1:1 from
 mezzanine.pipelines.regression_distill; EnsembleMemberSymmetry mirrors the
 sample()/batch() interface used by recipes/neuralgcm_ens_warrant_distill.py.
 Drop the body of main() into a Recipe.run() and it registers as a recipe.

Units: kcal/mol, kcal/mol/Angstrom, Angstrom, fs, amu (MD17 native).
Energies are handled RELATIVE to the training mean (float32 safety); the mean is
stored in results.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as TF

# ----------------------------------------------------------------------------- units
ACC_CONV = 4.184e-4          # (kcal/mol/A)/amu -> A/fs^2   (== 1 eV/A/amu * 0.0433641 * 9.6485e-3)
KB = 0.0019872041            # kcal/mol/K
MASS = {1: 1.008, 6: 12.011, 7: 14.007, 8: 15.999, 9: 18.998, 16: 32.06, 17: 35.45}

MD17_BASE = "http://www.quantum-machine.org/gdml/data/npz/"
MD17_FILES = {
    "ethanol": "md17_ethanol.npz",
    "malonaldehyde": "md17_malonaldehyde.npz",
    "uracil": "md17_uracil.npz",
    "toluene": "md17_toluene.npz",
    "salicylic": "md17_salicylic.npz",
    "naphthalene": "md17_naphthalene.npz",
    "aspirin": "md17_aspirin.npz",
    "benzene": "md17_benzene2017.npz",
    "azobenzene": "md17_azobenzene.npz",
    "paracetamol": "md17_paracetamol.npz",
}


# ============================================================================ data
def download_md17(molecule: str, data_dir: Path) -> Path:
    fname = MD17_FILES[molecule]
    dst = data_dir / fname
    if dst.exists():
        return dst
    data_dir.mkdir(parents=True, exist_ok=True)
    url = MD17_BASE + fname
    print(f"[data] downloading {url}")
    try:
        urllib.request.urlretrieve(url, dst)
        return dst
    except Exception as e:
        print(f"[data] direct download failed ({e}); trying torch_geometric fallback")
    try:
        from torch_geometric.datasets import MD17  # type: ignore

        ds = MD17(root=str(data_dir / "pyg"), name=molecule)
        raw = Path(ds.raw_paths[0])
        return raw
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            f"Could not obtain {fname}. Download it manually from {url} and pass --npz PATH."
        ) from e


def load_ef(path: Path) -> Dict[str, np.ndarray]:
    """MD17 (R,E,F,z) or rMD17 (coords,energies,forces,nuclear_charges). kcal/mol units."""
    with np.load(path, allow_pickle=True) as z:
        k = set(z.files)
        if {"R", "E", "F", "z"} <= k:
            R, E, F, Z = z["R"], z["E"], z["F"], z["z"]
        elif {"coords", "energies", "forces", "nuclear_charges"} <= k:
            R, E, F, Z = z["coords"], z["energies"], z["forces"], z["nuclear_charges"]
        else:
            raise KeyError(
                f"unrecognised npz keys {sorted(k)}; need MD17 (R,E,F,z) or "
                "rMD17 (coords,energies,forces,nuclear_charges)"
            )
    R = np.asarray(R, dtype=np.float64)
    F = np.asarray(F, dtype=np.float64)
    E = np.asarray(E, dtype=np.float64).reshape(-1)
    Z = np.asarray(Z).reshape(-1).astype(np.int64)
    assert R.ndim == 3 and R.shape[2] == 3 and F.shape == R.shape and R.shape[1] == Z.shape[0]
    return {"R": R, "E": E, "F": F, "Z": Z}


# ============================================================================ mezzanine primitives (verbatim maths)
def warrant_gap_regression(pred_views: np.ndarray) -> Dict[str, float]:
    """pred_views [K, N, D].  Identical to mezzanine.pipelines.regression_distill."""
    if pred_views.ndim != 3:
        raise ValueError(f"pred_views must be [K,N,D], got {pred_views.shape}")
    mu = pred_views.mean(axis=0, keepdims=True)
    diff = pred_views - mu
    return {
        "gap_mse": float(np.mean(diff**2)),
        "gap_l2": float(np.mean(np.linalg.norm(diff, axis=-1))),
        "gap_rms": float(np.sqrt(np.mean(diff**2))),  # same units as the quantity
    }


class EnsembleMemberSymmetry:
    """Member exchangeability as a symmetry family (mirrors mezzanine.symmetries.ens_member)."""

    NAME = "ens_member"

    def __init__(self, num_members: int):
        self.num_members = int(num_members)

    def sample(self, x: Any, *, seed: int) -> int:
        return int(np.random.default_rng(int(seed)).integers(0, self.num_members))

    def batch(self, x: Any, k: int, *, seed: int) -> List[int]:
        return [self.sample(x, seed=int(seed) + i) for i in range(int(k))]


# ============================================================================ model
class SSP(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return TF.softplus(x) - math.log(2.0)


class Interaction(nn.Module):
    def __init__(self, hidden: int, n_rbf: int):
        super().__init__()
        self.filt = nn.Sequential(nn.Linear(n_rbf, hidden), SSP(), nn.Linear(hidden, hidden))
        self.lin_in = nn.Linear(hidden, hidden, bias=False)
        self.out = nn.Sequential(nn.Linear(hidden, hidden), SSP(), nn.Linear(hidden, hidden))

    def forward(self, h: torch.Tensor, rbf: torch.Tensor, fc: torch.Tensor) -> torch.Tensor:
        W = self.filt(rbf) * fc[..., None]            # [B,A,A,H]  continuous filter
        m = torch.einsum("bijh,bjh->bih", W, self.lin_in(h))
        return h + self.out(m)


class SchNetLite(nn.Module):
    """E(3)-invariant energy, conservative forces via autograd.  Fixed molecule (single A)."""

    def __init__(self, hidden: int = 64, n_rbf: int = 32, n_int: int = 3, cutoff: float = 5.0,
                 e_scale: float = 1.0, n_species: int = 100):
        super().__init__()
        self.cutoff = float(cutoff)
        self.emb = nn.Embedding(n_species, hidden)
        centers = torch.linspace(0.0, cutoff, n_rbf)
        self.register_buffer("centers", centers)
        spacing = float(centers[1] - centers[0])
        self.gamma = 0.5 / spacing**2
        self.ints = nn.ModuleList([Interaction(hidden, n_rbf) for _ in range(n_int)])
        self.readout = nn.Sequential(nn.Linear(hidden, hidden), SSP(), nn.Linear(hidden, 1))
        self.register_buffer("e_scale", torch.tensor(float(e_scale)))

    def energy(self, R: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        B, A, _ = R.shape
        h = self.emb(Z)[None].expand(B, A, -1)
        diff = R[:, :, None, :] - R[:, None, :, :]
        d = torch.sqrt((diff * diff).sum(-1).clamp_min(1e-12))          # [B,A,A]
        mask = 1.0 - torch.eye(A, device=R.device, dtype=R.dtype)
        rbf = torch.exp(-self.gamma * (d[..., None] - self.centers) ** 2)
        fc = 0.5 * (torch.cos(math.pi * d / self.cutoff) + 1.0) * (d < self.cutoff).to(R.dtype) * mask
        for blk in self.ints:
            h = blk(h, rbf, fc)
        e_atom = self.readout(h).squeeze(-1)                              # [B,A]
        return e_atom.sum(-1) * self.e_scale                              # [B]  (relative energy)

    def forward(self, R: torch.Tensor, Z: torch.Tensor, create_graph: bool = False):
        R = R.detach().requires_grad_(True)
        E = self.energy(R, Z)
        (dE,) = torch.autograd.grad(E.sum(), R, create_graph=create_graph)
        return E, -dE


def predict_t(model: nn.Module, R: torch.Tensor, Z: torch.Tensor, bs: int = 256):
    model.eval()
    Es, Fs = [], []
    for i in range(0, R.shape[0], bs):
        with torch.enable_grad():
            E, F = model(R[i:i + bs], Z, create_graph=False)
        Es.append(E.detach())
        Fs.append(F.detach())
    return torch.cat(Es), torch.cat(Fs)


def predict_np(model: nn.Module, R: np.ndarray, Z: torch.Tensor, device: str):
    E, F = predict_t(model, torch.as_tensor(R, dtype=torch.float32, device=device), Z)
    return E.cpu().numpy().astype(np.float64), F.cpu().numpy().astype(np.float64)


# ============================================================================ training
def to_tensors(R: np.ndarray, E: np.ndarray, F: np.ndarray, device: str,
               soft: Optional[Tuple[np.ndarray, np.ndarray]] = None,
               labeled: Optional[np.ndarray] = None) -> Dict[str, torch.Tensor]:
    d = {
        "R": torch.as_tensor(R, dtype=torch.float32, device=device),
        "E": torch.as_tensor(E, dtype=torch.float32, device=device),
        "F": torch.as_tensor(F, dtype=torch.float32, device=device),
        "labeled": torch.as_tensor(np.ones(len(R), bool) if labeled is None else labeled, device=device),
    }
    if soft is not None:
        d["E_soft"] = torch.as_tensor(soft[0], dtype=torch.float32, device=device)
        d["F_soft"] = torch.as_tensor(soft[1], dtype=torch.float32, device=device)
    return d


def _train_job(job: Dict[str, Any]) -> Dict[str, Any]:
    """One independent supervised fit.  Runs in the main process or a spawned worker."""
    import os
    device = job["device"]
    if device == "cpu":
        torch.set_num_threads(max(1, (os.cpu_count() or 1) // max(1, int(job["n_parallel"]))))
    if job.get("tf32"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    with np.load(job["subset"]) as z:
        R, E, F, Z = z["R"], z["E_rel"], z["F"], z["Z"]
    Zt = torch.as_tensor(Z, device=device)
    tr = to_tensors(R[job["tr"]], E[job["tr"]], F[job["tr"]], device)
    va = to_tensors(R[job["va"]], E[job["va"]], F[job["va"]], device)
    model = SchNetLite(**job["model_kw"]).to(device)
    curves: List[Tuple[str, int, float, float]] = []
    info = fit(model, Zt, tr, va, FitCfg(**job["fit"]), job["sig_E"], job["sig_F"], device,
               job["seed"], job["tag"], curves)
    torch.save(model.state_dict(), job["ckpt"])
    return {"tag": job["tag"], "curves": curves, "info": info}


@dataclass
class FitCfg:
    steps: int
    batch: int
    lr: float
    force_weight: float          # lambda_F in [0,1]; loss = (1-l) E-term + l F-term (each std-normalised)
    hard_label_weight: float = 1.0  # w: (1-w) soft(ensemble) + w hard(DFT)   -- mezzanine convention
    grad_clip: float = 10.0
    eval_every: int = 100


def fit(model: nn.Module, Z: torch.Tensor, tr: Dict[str, torch.Tensor], va: Dict[str, torch.Tensor],
        cfg: FitCfg, sig_E: float, sig_F: float, device: str, seed: int, tag: str,
        curves: List[Tuple[str, int, float, float]]) -> Dict[str, float]:
    """tr: R,E,F,labeled(bool) [+ E_soft,F_soft]. Loss = (1-w) L_soft(all) + w L_hard(labeled)."""
    torch.manual_seed(int(seed))
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.steps)
    g = torch.Generator().manual_seed(int(seed))
    N = tr["R"].shape[0]
    fw, w = float(cfg.force_weight), float(cfg.hard_label_weight)
    have_soft = "E_soft" in tr

    def nloss(Ep, Fp, Et, Ft):
        le = ((Ep - Et) / sig_E).pow(2).mean()
        lf = ((Fp - Ft) / sig_F).pow(2).mean()
        return (1.0 - fw) * le + fw * lf

    best_loss, best_state = float("inf"), None
    t0 = time.time()
    for step in range(1, cfg.steps + 1):
        model.train()
        idx = torch.randint(0, N, (cfg.batch,), generator=g).to(device)
        Ep, Fp = model(tr["R"][idx], Z, create_graph=True)
        loss = torch.zeros((), device=device)
        if have_soft and w < 1.0:
            loss = loss + (1.0 - w) * nloss(Ep, Fp, tr["E_soft"][idx], tr["F_soft"][idx])
        if w > 0.0:
            m = tr["labeled"][idx]
            if bool(m.any()):
                loss = loss + w * nloss(Ep[m], Fp[m], tr["E"][idx][m], tr["F"][idx][m])
        if not loss.requires_grad:
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        sched.step()
        if step % cfg.eval_every == 0 or step == cfg.steps:
            Ev, Fv = predict_t(model, va["R"], Z)
            e_mae = (Ev - va["E"]).abs().mean().item()
            f_mae = (Fv - va["F"]).abs().mean().item()
            vloss = nloss(Ev, Fv, va["E"], va["F"]).item()
            curves.append((tag, step, e_mae, f_mae))
            if vloss < best_loss:
                best_loss = vloss
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            print(f"  [{tag}] step {step:5d}/{cfg.steps}  val E_mae {e_mae:.4f}  F_mae {f_mae:.4f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"val_loss_norm": float(best_loss)}


# ============================================================================ metrics / checks
def ef_metrics(E_pred, F_pred, E_true, F_true) -> Dict[str, float]:
    dE = E_pred - E_true
    dF = F_pred - F_true
    return {
        "E_mae": float(np.abs(dE).mean()),
        "E_rmse": float(np.sqrt((dE**2).mean())),
        "F_mae": float(np.abs(dF).mean()),
        "F_rmse": float(np.sqrt((dF**2).mean())),
    }


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    q = rng.normal(size=4); q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def equivariance_check(model: nn.Module, Z_np: np.ndarray, R: np.ndarray, device: str,
                       n: int = 64, trials: int = 4, seed: int = 0) -> Dict[str, float]:
    """Frame un-transformation: compare f(x) with g^-1 f(g x), g in SO(3) x S_A(within species) x T(3)."""
    rng = np.random.default_rng(seed)
    R0 = R[:n]
    Zt = torch.as_tensor(Z_np, device=device)
    E0, F0 = predict_np(model, R0, Zt, device)
    worst_E, worst_F = 0.0, 0.0
    for _ in range(trials):
        Rot = random_rotation(rng)
        t = rng.uniform(-3, 3, size=3)
        perm = np.arange(len(Z_np))
        for z in np.unique(Z_np):
            idx = np.where(Z_np == z)[0]
            perm[idx] = idx[rng.permutation(idx.size)]
        inv = np.argsort(perm)
        R1 = (R0 @ Rot.T + t)[:, perm]
        E1, F1 = predict_np(model, R1, torch.as_tensor(Z_np[perm], device=device), device)
        F1_can = (F1 @ Rot)[:, inv]                      # undo rotation, undo permutation
        worst_E = max(worst_E, float(np.abs(E1 - E0).max()))
        worst_F = max(worst_F, float(np.abs(F1_can - F0).max()))
    scale = float(np.abs(F0).mean())
    return {"max_abs_dE": worst_E, "max_abs_dF": worst_F, "rel_dF": worst_F / max(scale, 1e-12)}


def run_md(ef_fn: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]], R0: np.ndarray,
           Z_np: np.ndarray, steps: int, dt: float, T0: float, seed: int, device: str) -> Dict[str, Any]:
    """NVE velocity-Verlet.  ef_fn(R[1,A,3]) -> (E[1], F[1,A,3]) in kcal/mol units."""
    m_np = np.array([MASS[int(z)] for z in Z_np], dtype=np.float64)
    rng = np.random.default_rng(seed)
    v = rng.normal(size=R0.shape) * np.sqrt(KB * T0 / m_np[:, None] * ACC_CONV)   # A/fs
    v -= (v * m_np[:, None]).sum(0) / m_np.sum()
    m = torch.as_tensor(m_np, dtype=torch.float32, device=device)[:, None]
    R = torch.as_tensor(R0, dtype=torch.float32, device=device)
    v = torch.as_tensor(v, dtype=torch.float32, device=device)

    def step_ef(Rt):
        E, F = ef_fn(Rt[None])
        return E[0].detach(), F[0].detach()

    E, F = step_ef(R)
    a = F / m * ACC_CONV
    ke = 0.5 * (m * v * v).sum() / ACC_CONV
    etot0 = (E + ke).item()
    dmax0 = torch.cdist(R, R).max().item()
    ndof = max(1, 3 * len(Z_np) - 3)
    drift, temp = [], []
    blow = None
    for t in range(steps):
        v = v + 0.5 * dt * a
        R = R + dt * v
        E, F = step_ef(R)
        a = F / m * ACC_CONV
        v = v + 0.5 * dt * a
        ke = 0.5 * (m * v * v).sum() / ACC_CONV
        etot = (E + ke).item()
        drift.append(etot - etot0)
        temp.append(2.0 * ke.item() / (ndof * KB))
        dmax = torch.cdist(R, R).max().item()
        if not math.isfinite(etot) or dmax > 2.5 * dmax0 + 2.0:
            blow = t
            break
    d = np.asarray(drift, dtype=np.float64)
    fin = d[np.isfinite(d)]
    return {
        "drift": d, "temperature": np.asarray(temp, dtype=np.float64),
        "blowup_step": blow,
        "drift_rms": float(np.sqrt((fin**2).mean())) if fin.size else float("nan"),
        "drift_max_abs": float(np.abs(fin).max()) if fin.size else float("nan"),
        "steps_completed": int(len(d)),
    }


# ============================================================================ plots
def make_figures(out: Path, res: Dict[str, Any], preds: Dict[str, Any], curves, md_runs, mol: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ---- 1. Pareto: E-vs-F trade-off
    sw = res["pareto_sweep"]
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    xs = [s["E_mae"] for s in sw]; ys = [s["F_mae"] for s in sw]
    ax.plot(xs, ys, "-o", color="tab:gray", label="single model, sweep of force weight $\\lambda_F$")
    for s in sw:
        ax.annotate(f"$\\lambda_F$={s['force_weight']:g}", (s["E_mae"], s["F_mae"]),
                    textcoords="offset points", xytext=(5, 4), fontsize=8)
    for mm in res["teacher"]["members"]:
        ax.plot(mm["metrics"]["E_mae"], mm["metrics"]["F_mae"], ".", color="tab:blue", alpha=0.6)
    ax.plot([], [], ".", color="tab:blue", label="ensemble members")
    em = res["teacher"]["ensemble_metrics"]
    ax.plot(em["E_mae"], em["F_mae"], "*", ms=15, color="tab:blue",
            label=f"ensemble mean (K={res['symmetry']['num_members']} passes)")
    st = res["student"]["metrics"]
    ax.plot(st["E_mae"], st["F_mae"], "D", ms=10, color="tab:red", label="distilled student (1 pass)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("energy MAE (kcal/mol)"); ax.set_ylabel("force MAE (kcal/mol/Å)")
    ax.set_title(f"{mol}: energy–force trade-off on test set")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "fig_pareto.png", dpi=150); plt.close(fig)

    # ---- 2. Parity
    E_true, F_true = preds["E_true"], preds["F_true"]
    cols = [("single model", preds["E_single"], preds["F_single"]),
            ("ensemble mean", preds["E_members"].mean(0), preds["F_members"].mean(0)),
            ("distilled student", preds["E_student"], preds["F_student"])]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))
    rng = np.random.default_rng(0)
    for j, (name, Ep, Fp) in enumerate(cols):
        ax = axes[0, j]
        ax.plot(E_true, Ep, ".", ms=3, alpha=0.5)
        lo, hi = float(E_true.min()), float(E_true.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_title(f"{name}\nE MAE {np.abs(Ep - E_true).mean():.3f} kcal/mol")
        ax.set_xlabel("DFT E (rel., kcal/mol)"); ax.set_ylabel("pred E")
        ax = axes[1, j]
        ft, fp = F_true.reshape(-1), Fp.reshape(-1)
        sel = rng.choice(ft.size, size=min(6000, ft.size), replace=False)
        ax.plot(ft[sel], fp[sel], ".", ms=2, alpha=0.4)
        lo, hi = float(ft.min()), float(ft.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_title(f"F MAE {np.abs(Fp - F_true).mean():.3f} kcal/mol/Å")
        ax.set_xlabel("DFT F component"); ax.set_ylabel("pred F")
    fig.suptitle(f"{mol}: parity on test set"); fig.tight_layout()
    fig.savefig(out / "fig_parity.png", dpi=150); plt.close(fig)

    # ---- 3. Warrant gap: distribution + does member disagreement predict student error?
    Fm = preds["F_members"]                                   # [M,N,A,3]
    gap_cfg = np.sqrt(((Fm - Fm.mean(0, keepdims=True)) ** 2).mean(axis=(0, 2, 3)))   # per-config rms spread
    err_student = np.sqrt(((preds["F_student"] - F_true) ** 2).mean(axis=(1, 2)))
    err_single = np.sqrt(((preds["F_single"] - F_true) ** 2).mean(axis=(1, 2)))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axes[0]
    ax.hist(gap_cfg, bins=40, color="tab:blue", alpha=0.8)
    ax.axvline(np.median(gap_cfg), color="k", ls="--", label=f"median {np.median(gap_cfg):.3f}")
    ax.set_xlabel("per-config ensemble force gap, RMS over members/atoms (kcal/mol/Å)")
    ax.set_ylabel("count"); ax.set_title("ensemble warrant gap on forces (test)"); ax.legend()
    ax = axes[1]
    ax.plot(gap_cfg, err_single, ".", ms=3, alpha=0.5, label=f"single  ρ={spearman(gap_cfg, err_single):.2f}")
    ax.plot(gap_cfg, err_student, ".", ms=3, alpha=0.5, label=f"student ρ={spearman(gap_cfg, err_student):.2f}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("ensemble force gap (kcal/mol/Å)"); ax.set_ylabel("force RMSE vs DFT (kcal/mol/Å)")
    ax.set_title("gap as an error predictor (Spearman ρ)"); ax.legend(); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "fig_gap.png", dpi=150); plt.close(fig)

    # ---- 4. MD
    if md_runs:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        dt = res["md"]["dt_fs"]
        for name, r in md_runs.items():
            t = np.arange(1, len(r["drift"]) + 1) * dt
            lab = name + (f"  (blow-up @ {r['blowup_step'] * dt:.0f} fs)" if r["blowup_step"] is not None else "")
            axes[0].plot(t, r["drift"], lw=1.2, label=lab)
            axes[1].plot(t, r["temperature"], lw=1.2, label=name)
        axes[0].set_xlabel("time (fs)"); axes[0].set_ylabel("E_tot(t) − E_tot(0)  (kcal/mol)")
        axes[0].set_title(f"NVE energy conservation, {res['md']['T0_K']:g} K"); axes[0].legend(fontsize=8)
        axes[1].set_xlabel("time (fs)"); axes[1].set_ylabel("instantaneous T (K)"); axes[1].legend(fontsize=8)
        for ax in axes:
            ax.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(out / "fig_md.png", dpi=150); plt.close(fig)

    # ---- 5. Training curves
    tags = sorted(set(c[0] for c in curves), key=lambda s: (not s.startswith("teacher"), s))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for tag in tags:
        pts = [(c[1], c[2], c[3]) for c in curves if c[0] == tag]
        s = np.array(pts)
        kw = dict(lw=2.2, color="tab:red") if tag == "student" else dict(lw=1, alpha=0.7)
        axes[0].plot(s[:, 0], s[:, 1], label=tag, **kw)
        axes[1].plot(s[:, 0], s[:, 2], label=tag, **kw)
    axes[0].set_ylabel("val E MAE (kcal/mol)"); axes[1].set_ylabel("val F MAE (kcal/mol/Å)")
    for ax in axes:
        ax.set_xlabel("step"); ax.set_yscale("log"); ax.grid(True, which="both", alpha=0.3)
    axes[1].legend(fontsize=7, ncol=2)
    fig.suptitle("validation curves"); fig.tight_layout()
    fig.savefig(out / "fig_train.png", dpi=150); plt.close(fig)


# ============================================================================ main
def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--molecule", default="ethanol", choices=sorted(MD17_FILES))
    p.add_argument("--npz", default="", help="local MD17/rMD17 npz; overrides --molecule")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--n_train", type=int, default=1000, help="DFT-labelled configs (teachers + student hard labels)")
    p.add_argument("--n_unlabeled", type=int, default=3000, help="extra configs the ensemble labels for the student")
    p.add_argument("--n_test", type=int, default=1000)
    p.add_argument("--members", type=int, default=5)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--n_rbf", type=int, default=32)
    p.add_argument("--n_int", type=int, default=3)
    p.add_argument("--cutoff", type=float, default=5.0)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--force_weight", type=float, default=0.9, help="teacher/single lambda_F")
    p.add_argument("--sweep", default="0.0,0.5,0.9,0.99,1.0", help="lambda_F values for the Pareto sweep")
    p.add_argument("--hard_label_weight", type=float, default=0.5, help="student: 0=pure soft, 1=pure supervised")
    p.add_argument("--student_hidden", type=int, default=0, help="0 = same as teacher")
    p.add_argument("--student_int", type=int, default=0, help="0 = same as teacher")
    p.add_argument("--student_steps", type=int, default=0, help="0 = same as --steps")
    p.add_argument("--md_steps", type=int, default=2000)
    p.add_argument("--md_dt", type=float, default=0.5, help="fs")
    p.add_argument("--md_T", type=float, default=300.0, help="K")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--parallel", type=int, default=0,
                   help="concurrent fits for members + sweep (spawned processes, one GPU). 0 = auto, 1 = sequential")
    p.add_argument("--tf32", action="store_true", help="allow TF32 matmuls (A100/H100); relaxes equivariance criterion")
    p.add_argument("--quick", action="store_true", help="~2 min smoke test")
    args = p.parse_args(argv)

    if args.quick:
        args.n_train, args.n_unlabeled, args.n_test = 200, 400, 200
        args.steps, args.members, args.hidden, args.n_int = 300, 3, 32, 2
        args.sweep, args.md_steps = "0.0,0.9,1.0", 300
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print(f"[cfg] device={device}  {vars(args)}")

    # ---------------------------------------------------------------- data
    npz_path = Path(args.npz) if args.npz else download_md17(args.molecule, Path(args.data_dir))
    mol = Path(args.npz).stem if args.npz else args.molecule
    D = load_ef(npz_path)
    N_all, A = D["R"].shape[0], D["R"].shape[1]
    need = args.n_train + args.n_unlabeled + args.n_test
    if need > N_all:
        f = N_all / need
        args.n_train, args.n_unlabeled = int(args.n_train * f), int(args.n_unlabeled * f)
        args.n_test = N_all - args.n_train - args.n_unlabeled
        print(f"[data] only {N_all} configs; rescaled to train/unlab/test = {args.n_train}/{args.n_unlabeled}/{args.n_test}")
    perm = np.random.default_rng(args.seed).permutation(N_all)
    i_lab = perm[: args.n_train]
    i_unl = perm[args.n_train: args.n_train + args.n_unlabeled]
    i_test = perm[args.n_train + args.n_unlabeled: args.n_train + args.n_unlabeled + args.n_test]
    E_mean = float(D["E"][i_lab].mean())
    E_rel = D["E"] - E_mean
    sig_E = max(float(E_rel[i_lab].std()), 1e-3)
    sig_F = max(float(D["F"][i_lab].std()), 1e-3)
    Z_np = D["Z"]; Z = torch.as_tensor(Z_np, device=device)
    print(f"[data] {mol}: {N_all} configs, {A} atoms (Z={Z_np.tolist()}), "
          f"E std {sig_E:.3f} kcal/mol, F std {sig_F:.3f} kcal/mol/Å, E_mean {E_mean:.2f}")

    # Compact subset for workers + local index spaces
    i_all = np.concatenate([i_lab, i_unl, i_test])
    L = {"lab": np.arange(len(i_lab)), "unl": len(i_lab) + np.arange(len(i_unl)),
         "test": len(i_lab) + len(i_unl) + np.arange(len(i_test))}
    Rs, Es, Fs = D["R"][i_all], E_rel[i_all], D["F"][i_all]
    subset_path = out / "subset.npz"
    np.savez(subset_path, R=Rs.astype(np.float32), E_rel=Es.astype(np.float32), F=Fs.astype(np.float32), Z=Z_np)

    def split(seed):
        r = np.random.default_rng(seed).permutation(len(L["lab"]))
        n_val = max(1, int(0.1 * len(L["lab"])))
        return L["lab"][r[n_val:]], L["lab"][r[:n_val]]

    model_kw = dict(hidden=args.hidden, n_rbf=args.n_rbf, n_int=args.n_int, cutoff=args.cutoff, e_scale=sig_E)

    def new_model(hidden=None, n_int=None):
        kw = dict(model_kw)
        if hidden: kw["hidden"] = hidden
        if n_int: kw["n_int"] = n_int
        return SchNetLite(**kw).to(device)

    curves: List[Tuple[str, int, float, float]] = []
    R_test, E_test, F_test = Rs[L["test"]], Es[L["test"]], Fs[L["test"]]
    t_start = time.time()

    # ---------------------------------------------------------------- 1+2. independent fits: members (views) + Pareto sweep
    sym = EnsembleMemberSymmetry(args.members)
    sweep_vals = [float(s) for s in args.sweep.split(",") if s.strip()]
    tr0, va0 = split(args.seed * 1000)
    ckdir = out / "models"; ckdir.mkdir(exist_ok=True)
    jobs: List[Dict[str, Any]] = []
    for k in range(args.members):
        seed_k = args.seed * 1000 + k
        tr_i, va_i = split(seed_k)
        jobs.append(dict(tag=f"teacher{k}", kind="member", seed=seed_k, tr=tr_i, va=va_i,
                         fit=dict(steps=args.steps, batch=args.batch, lr=args.lr, force_weight=args.force_weight)))
    for fw in sweep_vals:
        if abs(fw - args.force_weight) < 1e-12:
            continue                                          # member 0 IS this sweep point
        jobs.append(dict(tag=f"sweep_lF{fw:g}", kind="sweep", seed=args.seed * 1000, tr=tr0, va=va0, force_weight=fw,
                         fit=dict(steps=args.steps, batch=args.batch, lr=args.lr, force_weight=fw)))
    n_par = args.parallel if args.parallel > 0 else min(len(jobs), max(1, os.cpu_count() or 1), 8)
    for j in jobs:
        j.update(device=device, subset=str(subset_path), model_kw=model_kw, sig_E=sig_E, sig_F=sig_F,
                 ckpt=str(ckdir / f"{j['tag']}.pt"), n_parallel=n_par, tf32=bool(args.tf32))
    print(f"[fits] {len(jobs)} independent fits ({args.members} members + {len(jobs) - args.members} sweep points), "
          f"parallel={n_par}")
    if n_par > 1 and len(jobs) > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(n_par) as pool:
            outs = pool.map(_train_job, jobs)
    else:
        outs = [_train_job(j) for j in jobs]
    for o in outs:
        curves.extend(o["curves"])

    members, member_res = [], []
    pareto = []
    for j in jobs:
        model = new_model()
        model.load_state_dict(torch.load(j["ckpt"], map_location=device))
        model.eval()
        Ek, Fk = predict_np(model, R_test, Z, device)
        m = ef_metrics(Ek, Fk, E_test, F_test)
        if j["kind"] == "member":
            members.append(model)
            member_res.append({"seed": j["seed"], "metrics": m})
            print(f"[teacher] {j['tag']} test {m}")
        else:
            pareto.append({"force_weight": j["force_weight"], **m})
            print(f"[sweep] lambda_F={j['force_weight']}: {m}")
    if any(abs(fw - args.force_weight) < 1e-12 for fw in sweep_vals):
        pareto.append({"force_weight": args.force_weight, **member_res[0]["metrics"]})
    pareto.sort(key=lambda s: s["force_weight"])

    # ---------------------------------------------------------------- 3. ensemble predictions + warrant gap
    def ensemble_np(R):
        Es, Fs = zip(*[predict_np(mm, R, Z, device) for mm in members])
        return np.stack(Es), np.stack(Fs)          # [M,N], [M,N,A,3]

    E_mem, F_mem = ensemble_np(R_test)
    gap_E = warrant_gap_regression(E_mem[:, :, None])
    gap_F = warrant_gap_regression(F_mem.reshape(args.members, len(i_test), -1))
    ens_metrics = ef_metrics(E_mem.mean(0), F_mem.mean(0), E_test, F_test)
    print(f"[gap] E {gap_E}\n[gap] F {gap_F}\n[ensemble] test {ens_metrics}")

    # ---------------------------------------------------------------- 4. distil student
    pool = np.concatenate([tr0, L["unl"]])
    labeled = np.concatenate([np.ones(len(tr0), bool), np.zeros(len(L["unl"]), bool)])
    if args.hard_label_weight >= 1.0:          # pure supervised: unlabeled rows carry no signal
        pool, labeled = tr0, np.ones(len(tr0), bool)
    print(f"[student] labelling pool of {len(pool)} configs with the ensemble ({len(tr0)} DFT-labelled)")
    E_soft, F_soft = ensemble_np(Rs[pool])
    pseudo_quality = ef_metrics(E_soft.mean(0)[~labeled], F_soft.mean(0)[~labeled],
                                Es[pool][~labeled], Fs[pool][~labeled]) if (~labeled).any() else None
    student = new_model(args.student_hidden or None, args.student_int or None)
    st_cfg = FitCfg(args.student_steps or args.steps, args.batch, args.lr, args.force_weight,
                    hard_label_weight=args.hard_label_weight)
    fit(student, Z, to_tensors(Rs[pool], Es[pool], Fs[pool], device, soft=(E_soft.mean(0), F_soft.mean(0)), labeled=labeled),
        to_tensors(Rs[va0], Es[va0], Fs[va0], device), st_cfg, sig_E, sig_F, device, args.seed * 1000 + 777, "student", curves)
    torch.save(student.state_dict(), ckdir / "student.pt")
    E_st, F_st = predict_np(student, R_test, Z, device)
    st_metrics = ef_metrics(E_st, F_st, E_test, F_test)
    st_to_ens = ef_metrics(E_st, F_st, E_mem.mean(0), F_mem.mean(0))
    single_metrics = member_res[0]["metrics"]
    E_single, F_single = predict_np(members[0], R_test, Z, device)
    print(f"[student] test {st_metrics}\n[student] to ensemble mean {st_to_ens}")

    # ---------------------------------------------------------------- 5. equivariance
    eq_student = equivariance_check(student, Z_np, R_test, device, seed=args.seed)
    eq_single = equivariance_check(members[0], Z_np, R_test, device, seed=args.seed)
    print(f"[equivariance] student {eq_student}")

    # ---------------------------------------------------------------- 6. MD
    md_runs: Dict[str, Dict[str, Any]] = {}
    if args.md_steps > 0:
        R0 = R_test[0]

        def single_fn(Rt):
            with torch.enable_grad():
                return members[0](Rt, Z)

        def student_fn(Rt):
            with torch.enable_grad():
                return student(Rt, Z)

        def ensemble_fn(Rt):
            with torch.enable_grad():
                outs = [mm(Rt, Z) for mm in members]
            return torch.stack([o[0] for o in outs]).mean(0), torch.stack([o[1] for o in outs]).mean(0)

        for name, fn in [("single", single_fn), ("ensemble", ensemble_fn), ("student", student_fn)]:
            for mm in members + [student]:
                mm.eval()
            r = run_md(fn, R0, Z_np, args.md_steps, args.md_dt, args.md_T, args.seed, device)
            md_runs[name] = r
            print(f"[md] {name}: drift_rms {r['drift_rms']:.4f} kcal/mol, max {r['drift_max_abs']:.4f}, "
                  f"blow-up {r['blowup_step']}, steps {r['steps_completed']}")

    # ---------------------------------------------------------------- 7. verdict + outputs
    crit = {
        "student F_mae <= 0.9 x single F_mae": bool(st_metrics["F_mae"] <= 0.9 * single_metrics["F_mae"]),
        "student E_mae <= 1.25 x single E_mae": bool(st_metrics["E_mae"] <= 1.25 * single_metrics["E_mae"]),
        f"student equivariant (rel dF < {'1e-2' if args.tf32 else '1e-3'})":
            bool(eq_student["rel_dF"] < (1e-2 if args.tf32 else 1e-3)),
    }
    if md_runs:
        crit["student MD no blow-up"] = md_runs["student"]["blowup_step"] is None
    verdict = "MAKE OK" if all(crit.values()) else "BREAK / INCONCLUSIVE X"

    res: Dict[str, Any] = {
        "exp": "mlip_ensemble_ef_distill",
        "device": device,
        "runtime_s": float(time.time() - t_start),
        "dataset": {"name": mol, "path": str(npz_path), "n_configs": int(N_all), "n_atoms": int(A),
                    "Z": Z_np.tolist(), "n_train_labeled": int(len(i_lab)), "n_unlabeled": int(len(i_unl)),
                    "n_test": int(len(i_test)), "E_mean_kcal_mol": E_mean, "sig_E": sig_E, "sig_F": sig_F,
                    "units": "kcal/mol, kcal/mol/A, A"},
        "symmetry": {"name": sym.NAME, "num_members": args.members},
        "model": {"family": "schnet_lite", "hidden": args.hidden, "n_rbf": args.n_rbf, "n_int": args.n_int,
                  "cutoff": args.cutoff, "forces": "autograd, F=-dE/dR (conservative)"},
        "train": {"steps": args.steps, "batch": args.batch, "lr": args.lr, "force_weight": args.force_weight,
                  "parallel": n_par, "tf32": bool(args.tf32)},
        "teacher": {"family": "schnet_lite_ensemble", "members": member_res, "ensemble_metrics": ens_metrics,
                    "warrant_gap": {"E": gap_E, "F": gap_F}, "forward_passes": args.members},
        "single_model": {"force_weight": args.force_weight, "metrics": single_metrics,
                         "equivariance": eq_single, "forward_passes": 1},
        "pareto_sweep": pareto,
        "student": {"family": "schnet_lite", "hidden": args.student_hidden or args.hidden,
                    "n_int": args.student_int or args.n_int, "metrics": st_metrics,
                    "to_ensemble_mean": st_to_ens, "equivariance": eq_student, "forward_passes": 1},
        "distill": {"hard_label_weight": args.hard_label_weight, "n_pool": int(len(pool)),
                    "n_unlabeled_used": int((~labeled).sum()), "pseudo_label_quality_on_unlabeled": pseudo_quality,
                    "student_steps": st_cfg.steps},
        "md": {"steps": args.md_steps, "dt_fs": args.md_dt, "T0_K": args.md_T,
               **{k: {kk: vv for kk, vv in v.items() if kk not in ("drift", "temperature")} for k, v in md_runs.items()}},
        "make_break": {"criterion": crit, "verdict": verdict},
    }
    (out / "results.json").write_text(json.dumps(res, indent=2))

    preds = {"Z": Z_np, "R_test": R_test, "E_true": E_test, "F_true": F_test, "E_mean": E_mean,
             "E_members": E_mem, "F_members": F_mem, "E_single": E_single, "F_single": F_single,
             "E_student": E_st, "F_student": F_st}
    np.savez_compressed(out / "predictions_test.npz", **preds)
    if md_runs:
        np.savez_compressed(out / "md_traces.npz", **{f"{k}_{kk}": v[kk] for k, v in md_runs.items() for kk in ("drift", "temperature")})

    rows = [("single (member 0)", single_metrics, 1, md_runs.get("single")),
            ("ensemble mean", ens_metrics, args.members, md_runs.get("ensemble")),
            ("distilled student", st_metrics, 1, md_runs.get("student"))]
    with (out / "summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "E_mae", "E_rmse", "F_mae", "F_rmse", "forward_passes", "md_drift_rms", "md_blowup_step"])
        for name, m, cost, r in rows:
            w.writerow([name, f"{m['E_mae']:.5f}", f"{m['E_rmse']:.5f}", f"{m['F_mae']:.5f}", f"{m['F_rmse']:.5f}",
                        cost, "" if r is None else f"{r['drift_rms']:.5f}", "" if r is None else r["blowup_step"]])

    make_figures(out, res, preds, curves, md_runs, mol)

    print("\n================ summary ================")
    print(f"{'model':20s} {'E_mae':>8s} {'F_mae':>8s} {'passes':>6s} {'MD drift rms':>13s} {'blow-up':>8s}")
    for name, m, cost, r in rows:
        print(f"{name:20s} {m['E_mae']:8.4f} {m['F_mae']:8.4f} {cost:6d} "
              f"{(r['drift_rms'] if r else float('nan')):13.4f} {str(r['blowup_step'] if r else '-'):>8s}")
    print(f"ensemble warrant gap: E rms {gap_E['gap_rms']:.4f} kcal/mol, F rms {gap_F['gap_rms']:.4f} kcal/mol/Å")
    print(f"verdict: {verdict}   ({res['runtime_s']:.0f}s)   outputs -> {out}")
    return res


if __name__ == "__main__":
    main()
