"""Generate a realistic Lennard–Jones (LJ) fluid / droplet dataset.

This script produces an HDF5 file that can be consumed by the Mezzanine
`LJFluidH5Adapter` world adapter.

Design goals
------------
- **Realistic MD workflow** for an MD student:
  - Reduced LJ units (epsilon=sigma=mass=kB=1)
  - Periodic boundary conditions (cubic box)
  - Truncated *force-shifted* Lennard–Jones potential (smooth force at r_c)
  - Neighbor list via SciPy cKDTree (periodic), with skin + rebuild criterion
  - Langevin thermostat (BAOAB splitting)
  - Equilibration + production, with trajectory striding
- **Reproducible**: all randomness is controlled by a seed.

The output is meant to be *structural* snapshots (positions + box) paired with
labels. We store two label types:
  1) `state_id` = index of the simulated state point (rho, T)
  2) `phase`    = 0/1 label from a per-snapshot cluster analysis (droplet vs vapor)

You can train either a multiclass “identify the state point” model or a binary
“droplet vs vapor” model.

Example
-------
python examples/lj_generate_dataset.py \
  --out data/lj_fluid.h5 \
  --N 864 \
  --rhos 0.05 0.10 0.20 \
  --temps 0.55 0.70 0.90 1.20 \
  --train_reps 2 --test_reps 1 \
  --equil 40000 --prod 80000 --stride 200
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
from scipy.spatial import cKDTree


# -----------------------------
# MD core (LJ + neighbor list)
# -----------------------------


def wrap_pbc(x: np.ndarray, L: float) -> np.ndarray:
    """Wrap positions into [0, L)."""
    return np.mod(x, L)


def minimum_image(dr: np.ndarray, L: float) -> np.ndarray:
    """Apply minimum image convention for cubic PBC."""
    return dr - L * np.round(dr / L)


def random_fcc_positions(*, N: int, L: float) -> np.ndarray:
    """Generate FCC lattice positions for N = 4*n^3 in a cubic box of size L."""
    # FCC: 4 atoms per cubic unit cell.
    n_cell_float = (N / 4.0) ** (1.0 / 3.0)
    n_cell = int(round(n_cell_float))
    if 4 * n_cell**3 != N:
        raise ValueError(
            f"FCC initialization requires N=4*n^3. Got N={N}; closest n={n_cell} gives {4*n_cell**3}."
        )
    a = L / float(n_cell)
    basis = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
        ],
        dtype=np.float64,
    )
    pos = np.zeros((N, 3), dtype=np.float64)
    idx = 0
    for ix in range(n_cell):
        for iy in range(n_cell):
            for iz in range(n_cell):
                cell_origin = a * np.array([ix, iy, iz], dtype=np.float64)
                for b in basis:
                    pos[idx] = cell_origin + a * b
                    idx += 1
    # wrap for safety
    return wrap_pbc(pos, L)


def maxwell_boltzmann_velocities(*, rng: np.random.Generator, N: int, T: float, mass: float = 1.0) -> np.ndarray:
    """Sample velocities from Maxwell–Boltzmann at temperature T (reduced units)."""
    std = np.sqrt(float(T) / float(mass))
    v = rng.normal(0.0, std, size=(N, 3)).astype(np.float64)
    # Remove net momentum (use N-1 dof for temperature diagnostics later).
    v -= v.mean(axis=0, keepdims=True)
    return v


@dataclass
class LJMDConfig:
    # System
    N: int = 864
    rho: float = 0.10
    T: float = 0.70

    # Potential (reduced units: epsilon=sigma=1)
    rc: float = 2.5
    skin: float = 0.3
    mass: float = 1.0

    # Integrator / thermostat
    dt: float = 0.004
    gamma: float = 1.0  # friction in 1/tau

    # Run lengths
    equil_steps: int = 40000
    prod_steps: int = 80000
    stride: int = 200

    # Cluster analysis (droplet vs vapor)
    r_cluster: float = 1.5
    cluster_frac_threshold: float = 0.30

    # Seed
    seed: int = 0

    def validate(self) -> None:
        if self.N <= 0:
            raise ValueError("N must be > 0")
        if not (self.rho > 0):
            raise ValueError("rho must be > 0")
        if not (self.T > 0):
            raise ValueError("T must be > 0")
        if not (self.rc > 0):
            raise ValueError("rc must be > 0")
        if not (self.skin >= 0):
            raise ValueError("skin must be >= 0")
        if not (self.dt > 0):
            raise ValueError("dt must be > 0")
        if not (self.gamma >= 0):
            raise ValueError("gamma must be >= 0")
        if self.equil_steps < 0 or self.prod_steps <= 0:
            raise ValueError("equil_steps must be >= 0 and prod_steps must be > 0")
        if self.stride <= 0:
            raise ValueError("stride must be > 0")
        if not (self.r_cluster > 0):
            raise ValueError("r_cluster must be > 0")
        if not (0.0 < self.cluster_frac_threshold <= 1.0):
            raise ValueError("cluster_frac_threshold must be in (0,1]")


class NeighborList:
    """Simple Verlet-style neighbor list for a cubic periodic box.

    Uses SciPy cKDTree with `boxsize=L` to build unique i<j pairs within
    (rc + skin). Rebuilt when the maximum displacement since last rebuild
    exceeds skin/2.
    """

    def __init__(self, *, rc: float, skin: float):
        self.rc = float(rc)
        self.skin = float(skin)
        self.r_build = float(rc + skin)
        self._pairs: np.ndarray | None = None
        self._x_ref: np.ndarray | None = None

    def maybe_rebuild(self, x: np.ndarray, L: float) -> None:
        if self._pairs is None or self._x_ref is None:
            self.rebuild(x, L)
            return

        if self.skin <= 0:
            # no skin: rebuild every step
            self.rebuild(x, L)
            return

        dr = minimum_image(x - self._x_ref, L)
        max_disp = float(np.sqrt((dr * dr).sum(axis=1)).max())
        if max_disp > 0.5 * self.skin:
            self.rebuild(x, L)

    def rebuild(self, x: np.ndarray, L: float) -> None:
        # cKDTree periodic support expects points in [0, L)
        xw = wrap_pbc(x, L)
        tree = cKDTree(xw, boxsize=float(L))
        pairs = tree.query_pairs(r=self.r_build, output_type="ndarray")
        # pairs shape: [M,2], i<j
        self._pairs = pairs.astype(np.int64, copy=False)
        self._x_ref = xw.copy()

    @property
    def pairs(self) -> np.ndarray:
        if self._pairs is None:
            raise RuntimeError("Neighbor list not built yet")
        return self._pairs


def lj_force_shift_constants(rc: float) -> Tuple[float, float]:
    """Return (U(rc), F_rad(rc)) for reduced LJ at r=rc."""
    inv_r = 1.0 / float(rc)
    inv_r2 = inv_r * inv_r
    inv_r6 = inv_r2**3
    inv_r12 = inv_r6**2
    Uc = 4.0 * (inv_r12 - inv_r6)
    F_rad_c = 24.0 * inv_r * (2.0 * inv_r12 - inv_r6)
    return float(Uc), float(F_rad_c)


def compute_forces(
    x: np.ndarray,
    *,
    L: float,
    pairs: np.ndarray,
    rc: float,
) -> Tuple[np.ndarray, float, float]:
    """Compute LJ forces, potential energy, and virial using a force-shifted LJ.

    Returns:
      f: [N,3]
      U: total potential energy
      W: total virial sum over pairs of r_ij · f_ij
    """
    N = x.shape[0]
    f = np.zeros((N, 3), dtype=np.float64)
    U = 0.0
    W = 0.0

    if pairs.size == 0:
        return f, U, W

    i = pairs[:, 0]
    j = pairs[:, 1]
    dr = x[i] - x[j]
    dr = minimum_image(dr, L)
    r2 = (dr * dr).sum(axis=1)
    rc2 = float(rc) ** 2
    mask = r2 < rc2
    if not np.any(mask):
        return f, U, W

    dr = dr[mask]
    r2 = r2[mask]
    i = i[mask]
    j = j[mask]

    r = np.sqrt(r2)
    inv_r = 1.0 / r
    inv_r2 = inv_r * inv_r
    inv_r6 = inv_r2**3
    inv_r12 = inv_r6**2

    # Unshifted LJ
    U0 = 4.0 * (inv_r12 - inv_r6)
    F_rad = 24.0 * inv_r * (2.0 * inv_r12 - inv_r6)

    # Force-shift so that F(rc)=0 and U(rc)=0.
    Uc, F_rc = lj_force_shift_constants(rc)
    U_fs = U0 - Uc + (r - float(rc)) * F_rc
    F_fs = F_rad - F_rc

    # Vector forces
    fij = (F_fs * inv_r)[:, None] * dr  # [M,3]

    # Accumulate
    np.add.at(f, i, fij)
    np.add.at(f, j, -fij)

    # Energies / virial
    U = float(U_fs.sum())
    # W = sum r_ij · f_ij
    W = float((dr * fij).sum())
    return f, U, W


def kinetic_energy(v: np.ndarray, mass: float = 1.0) -> float:
    return float(0.5 * float(mass) * (v * v).sum())


def temperature_from_kinetic(K: float, N: int) -> float:
    # 3(N-1) dof after removing COM momentum.
    dof = max(1, 3 * (int(N) - 1))
    return float(2.0 * K / float(dof))


def pressure_from_virial(*, rho: float, T: float, W: float, V: float) -> float:
    # Reduced units: P = rho*T + W/(3V)
    return float(rho * T + W / (3.0 * V))


def union_find_largest_component(n: int, edges: np.ndarray) -> int:
    """Return size of the largest connected component for an undirected graph."""
    parent = np.arange(n, dtype=np.int64)
    size = np.ones(n, dtype=np.int64)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in edges:
        ra = find(int(a))
        rb = find(int(b))
        if ra == rb:
            continue
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]
    return int(size.max(initial=0))


def droplet_label_from_snapshot(x: np.ndarray, *, L: float, r_cluster: float, frac_thr: float) -> int:
    """Binary label: 1 if a macroscopic droplet/cluster is present, else 0.

    We build a neighbor graph using a physical cutoff r_cluster (typically ~1.4–1.6
    in reduced LJ units around the first coordination shell) and compute the
    largest connected component size.

    This is a common operational definition in droplet/vapor simulations at
    low global density.
    """
    xw = wrap_pbc(x, L)
    tree = cKDTree(xw, boxsize=float(L))
    pairs = tree.query_pairs(r=float(r_cluster), output_type="ndarray")
    if pairs.size == 0:
        return 0
    largest = union_find_largest_component(xw.shape[0], pairs)
    frac = float(largest) / float(xw.shape[0])
    return 1 if frac >= float(frac_thr) else 0


def run_lj_nvt(cfg: LJMDConfig) -> Dict[str, np.ndarray]:
    """Run an NVT (Langevin) simulation and return sampled frames + thermodynamics."""
    cfg.validate()
    rng = np.random.default_rng(cfg.seed)

    N = int(cfg.N)
    rho = float(cfg.rho)
    T = float(cfg.T)
    L = float((N / rho) ** (1.0 / 3.0))
    V = L**3

    x = random_fcc_positions(N=N, L=L)
    # Break perfect symmetry slightly
    x = wrap_pbc(x + rng.normal(0.0, 1e-3, size=x.shape), L)
    v = maxwell_boltzmann_velocities(rng=rng, N=N, T=T, mass=float(cfg.mass))

    nbl = NeighborList(rc=float(cfg.rc), skin=float(cfg.skin))
    nbl.rebuild(x, L)
    f, U, W = compute_forces(x, L=L, pairs=nbl.pairs, rc=float(cfg.rc))

    dt = float(cfg.dt)
    gamma = float(cfg.gamma)
    mass = float(cfg.mass)

    # BAOAB Langevin constants
    a = float(np.exp(-gamma * dt)) if gamma > 0 else 1.0
    b = float(np.sqrt((1.0 - a * a) * T / mass)) if gamma > 0 else 0.0

    def step(x: np.ndarray, v: np.ndarray, f: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        # B: half kick
        v = v + 0.5 * dt * f / mass

        # A: half drift
        x = x + 0.5 * dt * v
        x = wrap_pbc(x, L)

        # O: thermostat
        if gamma > 0:
            v = a * v + b * rng.normal(0.0, 1.0, size=v.shape)
            # remove COM momentum drift
            v = v - v.mean(axis=0, keepdims=True)

        # A: half drift
        x = x + 0.5 * dt * v
        x = wrap_pbc(x, L)

        # rebuild nlist if needed
        nbl.maybe_rebuild(x, L)
        f, U, W = compute_forces(x, L=L, pairs=nbl.pairs, rc=float(cfg.rc))

        # B: half kick
        v = v + 0.5 * dt * f / mass
        v = v - v.mean(axis=0, keepdims=True)

        return x, v, f, U, W

    # Equilibration
    for _ in range(int(cfg.equil_steps)):
        x, v, f, U, W = step(x, v, f)

    # Production (sample)
    frames: List[np.ndarray] = []
    thermo_U: List[float] = []
    thermo_T: List[float] = []
    thermo_P: List[float] = []
    thermo_phase: List[int] = []

    for t in range(int(cfg.prod_steps)):
        x, v, f, U, W = step(x, v, f)
        if (t % int(cfg.stride)) == 0:
            K = kinetic_energy(v, mass=mass)
            T_inst = temperature_from_kinetic(K, N)
            P_inst = pressure_from_virial(rho=rho, T=T_inst, W=W, V=V)
            phase = droplet_label_from_snapshot(
                x,
                L=L,
                r_cluster=float(cfg.r_cluster),
                frac_thr=float(cfg.cluster_frac_threshold),
            )
            frames.append(x.astype(np.float32, copy=True))
            thermo_U.append(float(U) / float(N))
            thermo_T.append(float(T_inst))
            thermo_P.append(float(P_inst))
            thermo_phase.append(int(phase))

    return {
        "pos": np.stack(frames, axis=0),
        "box": np.full((len(frames),), float(L), dtype=np.float32),
        "U_per_particle": np.array(thermo_U, dtype=np.float32),
        "T_inst": np.array(thermo_T, dtype=np.float32),
        "P_inst": np.array(thermo_P, dtype=np.float32),
        "phase": np.array(thermo_phase, dtype=np.int64),
        "rho": np.full((len(frames),), float(rho), dtype=np.float32),
        "T": np.full((len(frames),), float(T), dtype=np.float32),
    }


# -----------------------------
# Dataset assembly / IO
# -----------------------------


@dataclass
class DatasetGenConfig:
    out: str
    seed: int = 0

    # State points grid
    rhos: Tuple[float, ...] = (0.05, 0.10, 0.20)
    temps: Tuple[float, ...] = (0.55, 0.70, 0.90, 1.20)

    # Replicates per state point
    train_reps: int = 2
    test_reps: int = 1

    # MD config defaults
    N: int = 864
    rc: float = 2.5
    skin: float = 0.3
    mass: float = 1.0
    dt: float = 0.004
    gamma: float = 1.0
    equil: int = 40000
    prod: int = 80000
    stride: int = 200
    r_cluster: float = 1.5
    cluster_frac_threshold: float = 0.30


def _concat(records: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    keys = sorted(records[0].keys())
    out: Dict[str, np.ndarray] = {}
    for k in keys:
        out[k] = np.concatenate([r[k] for r in records], axis=0)
    return out


def build_dataset(cfg: DatasetGenConfig) -> None:
    out_path = Path(cfg.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    state_points: List[Tuple[float, float]] = [(float(r), float(T)) for r in cfg.rhos for T in cfg.temps]
    n_states = len(state_points)

    train_records: List[Dict[str, np.ndarray]] = []
    test_records: List[Dict[str, np.ndarray]] = []

    # Deterministic seeding scheme
    def run_seed(split: str, state_id: int, rep: int) -> int:
        base = int(cfg.seed)
        s = 1000003 * base + 9176 * state_id + 7919 * rep
        if split == "test":
            s += 424242
        return int(s % (2**32 - 1))

    for state_id, (rho, T) in enumerate(state_points):
        print(f"[dataset] state_id={state_id}/{n_states-1} rho={rho:.4f} T={T:.4f}")

        for rep in range(int(cfg.train_reps)):
            md_cfg = LJMDConfig(
                N=int(cfg.N),
                rho=float(rho),
                T=float(T),
                rc=float(cfg.rc),
                skin=float(cfg.skin),
                mass=float(cfg.mass),
                dt=float(cfg.dt),
                gamma=float(cfg.gamma),
                equil_steps=int(cfg.equil),
                prod_steps=int(cfg.prod),
                stride=int(cfg.stride),
                r_cluster=float(cfg.r_cluster),
                cluster_frac_threshold=float(cfg.cluster_frac_threshold),
                seed=run_seed("train", state_id, rep),
            )
            rec = run_lj_nvt(md_cfg)
            rec["state_id"] = np.full((rec["pos"].shape[0],), int(state_id), dtype=np.int64)
            rec["rep"] = np.full((rec["pos"].shape[0],), int(rep), dtype=np.int64)
            train_records.append(rec)

        for rep in range(int(cfg.test_reps)):
            md_cfg = LJMDConfig(
                N=int(cfg.N),
                rho=float(rho),
                T=float(T),
                rc=float(cfg.rc),
                skin=float(cfg.skin),
                mass=float(cfg.mass),
                dt=float(cfg.dt),
                gamma=float(cfg.gamma),
                equil_steps=int(cfg.equil),
                prod_steps=int(cfg.prod),
                stride=int(cfg.stride),
                r_cluster=float(cfg.r_cluster),
                cluster_frac_threshold=float(cfg.cluster_frac_threshold),
                seed=run_seed("test", state_id, rep),
            )
            rec = run_lj_nvt(md_cfg)
            rec["state_id"] = np.full((rec["pos"].shape[0],), int(state_id), dtype=np.int64)
            rec["rep"] = np.full((rec["pos"].shape[0],), int(rep), dtype=np.int64)
            test_records.append(rec)

    train = _concat(train_records)
    test = _concat(test_records)

    meta = {
        "generator": "examples/lj_generate_dataset.py",
        "cfg": asdict(cfg),
        "state_points": [{"state_id": i, "rho": r, "T": t} for i, (r, t) in enumerate(state_points)],
        "label_fields": {
            "state_id": "multiclass (rho,T) index",
            "phase": "binary droplet=1 vs vapor=0 from cluster analysis",
        },
        "units": {
            "length": "sigma",
            "energy": "epsilon",
            "temperature": "epsilon/kB",
            "time": "tau = sigma*sqrt(m/epsilon)",
            "pressure": "epsilon/sigma^3",
        },
        "potential": {
            "type": "LJ",
            "cutoff": float(cfg.rc),
            "shift": "force-shifted",
        },
        "thermostat": {
            "type": "Langevin",
            "integrator": "BAOAB",
            "gamma": float(cfg.gamma),
            "dt": float(cfg.dt),
        },
    }

    with h5py.File(out_path, "w") as h5:
        h5.attrs["meta_json"] = json.dumps(meta)
        for split_name, arrs in [("train", train), ("test", test)]:
            grp = h5.create_group(split_name)
            for k, v in arrs.items():
                # chunk and compress for realistic sizes
                if k == "pos":
                    chunks = (1, v.shape[1], 3)
                else:
                    chunks = True
                grp.create_dataset(
                    k,
                    data=v,
                    compression="gzip",
                    compression_opts=4,
                    shuffle=True,
                    chunks=chunks,
                )
        h5.flush()

    print(f"[done] wrote {out_path}  (train_frames={train['pos'].shape[0]}, test_frames={test['pos'].shape[0]})")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--N", type=int, default=864)
    p.add_argument("--rhos", type=float, nargs="+", default=[0.05, 0.10, 0.20])
    p.add_argument("--temps", type=float, nargs="+", default=[0.55, 0.70, 0.90, 1.20])
    p.add_argument("--train_reps", type=int, default=2)
    p.add_argument("--test_reps", type=int, default=1)

    p.add_argument("--rc", type=float, default=2.5)
    p.add_argument("--skin", type=float, default=0.3)
    p.add_argument("--dt", type=float, default=0.004)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--equil", type=int, default=40000)
    p.add_argument("--prod", type=int, default=80000)
    p.add_argument("--stride", type=int, default=200)

    p.add_argument("--r_cluster", type=float, default=1.5)
    p.add_argument("--cluster_frac_threshold", type=float, default=0.30)

    args = p.parse_args()

    cfg = DatasetGenConfig(
        out=str(args.out),
        seed=int(args.seed),
        N=int(args.N),
        rhos=tuple(float(x) for x in args.rhos),
        temps=tuple(float(x) for x in args.temps),
        train_reps=int(args.train_reps),
        test_reps=int(args.test_reps),
        rc=float(args.rc),
        skin=float(args.skin),
        dt=float(args.dt),
        gamma=float(args.gamma),
        equil=int(args.equil),
        prod=int(args.prod),
        stride=int(args.stride),
        r_cluster=float(args.r_cluster),
        cluster_frac_threshold=float(args.cluster_frac_threshold),
    )
    build_dataset(cfg)


if __name__ == "__main__":
    main()
