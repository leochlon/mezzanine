from __future__ import annotations

"""LALSuite-based gravitational-wave BBH merger *world* adapter.

This adapter samples *intrinsic* binary black hole (BBH) parameters for a dataset
of merger "world instances".

It does **not** generate detector strain directly. The intent is that a symmetry
(e.g., `gw_observation_lal`) will:
  - sample *extrinsic* parameters (sky location, inclination, polarization, distance, phase/time)
  - generate a waveform using LALSimulation
  - project onto a detector response
  - add a detector noise realization

Why this split?
  - you can treat extrinsics + noise as symmetries while keeping the intrinsics
    invariant across views of the same underlying merger.

Notes
-----
- This adapter is "proper" in the sense that it is designed to be paired with
  LALSuite/LALSimulation waveform generation (IMRPhenom / SEOBNR families, etc.).
- LALSuite is not a transitive dependency of Mezzanine. If you want to generate
  waveforms inside Mezzanine, install LALSuite in your environment.

Example output (one item)
-------------------------
{
  "id": 123,
  "intrinsics": {
     "m1_solar": ...,
     "m2_solar": ...,
     "mchirp_solar": ...,
     "q": ...,
     "eta": ...,
     "chi1": [s1x, s1y, s1z],
     "chi2": [s2x, s2y, s2z],
     "chi_eff": ...,
  }
}

"""


from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Literal, Tuple

import numpy as np

from ..core.cache import hash_dict
from .base import WorldAdapter
from ..registry import ADAPTERS


def _log_uniform(rng: np.random.Generator, lo: float, hi: float, *, size: int) -> np.ndarray:
    lo = float(lo)
    hi = float(hi)
    if not (lo > 0.0 and hi > 0.0 and hi > lo):
        raise ValueError(f"log-uniform requires 0 < lo < hi, got lo={lo}, hi={hi}")
    u = rng.uniform(np.log(lo), np.log(hi), size=size)
    return np.exp(u)


def _m1_m2_from_mchirp_q(mchirp: np.ndarray, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert chirp mass and mass ratio q>=1 to component masses.

    Derivation (q = m1/m2 >= 1):
      Mchirp = (m1 m2)^(3/5) / (m1+m2)^(1/5)
      => m1 = Mchirp * q^(2/5) * (q+1)^(1/5)
         m2 = m1 / q
    """
    m1 = mchirp * (q ** (2.0 / 5.0)) * ((q + 1.0) ** (1.0 / 5.0))
    m2 = m1 / q
    return m1, m2


def _eta_from_q(q: np.ndarray) -> np.ndarray:
    # eta = m1 m2 / (m1+m2)^2 = q / (1+q)^2 for q>=1 (with q=m1/m2)
    return q / ((1.0 + q) ** 2)


def _sample_isotropic_spin_components(
    rng: np.random.Generator, *, chi_max: float, n: int
) -> np.ndarray:
    """Sample dimensionless spin vectors uniformly in direction with magnitude in [0, chi_max]."""
    chi_max = float(chi_max)
    if not (0.0 <= chi_max <= 0.999999):
        raise ValueError(f"chi_max must be in [0, 1), got {chi_max}")
    # magnitude
    chi = rng.uniform(0.0, chi_max, size=n)
    # isotropic direction on sphere
    u = rng.uniform(-1.0, 1.0, size=n)  # cos(theta)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    sin_theta = np.sqrt(np.clip(1.0 - u * u, 0.0, 1.0))
    sx = chi * sin_theta * np.cos(phi)
    sy = chi * sin_theta * np.sin(phi)
    sz = chi * u
    return np.stack([sx, sy, sz], axis=1)


@dataclass
class GWMergerLALAdapterConfig:
    """Config for sampling intrinsic BBH parameters.

    Units:
      - masses are in **solar masses**
    """

    seed: int = 0
    n_train: int = 2000
    n_test: int = 400

    # Intrinsic priors
    mchirp_min_solar: float = 1e5
    mchirp_max_solar: float = 1e7
    q_min: float = 1.0
    q_max: float = 10.0

    mass_prior: Literal["loguniform", "uniform"] = "loguniform"
    q_prior: Literal["uniform", "loguniform"] = "uniform"

    # Spin prior: aligned (z-only) or isotropic vectors
    spin_prior: Literal["aligned_z", "isotropic"] = "aligned_z"
    chi_max: float = 0.99
    chi_z_min: float = -0.99
    chi_z_max: float = 0.99

    # Optional: annotate which waveform family you'll use downstream
    approximant: str = "IMRPhenomXPHM"

    def validate(self) -> None:
        if self.n_train <= 0 or self.n_test <= 0:
            raise ValueError("n_train and n_test must be > 0")
        if not (self.mchirp_min_solar > 0 and self.mchirp_max_solar > self.mchirp_min_solar):
            raise ValueError("mchirp_min_solar and mchirp_max_solar must satisfy 0 < min < max")
        if not (self.q_min >= 1.0 and self.q_max > self.q_min):
            raise ValueError("q_min must be >= 1 and q_max > q_min")
        if not (0.0 <= self.chi_max < 1.0):
            raise ValueError("chi_max must be in [0, 1)")
        if not (-1.0 < self.chi_z_min < 1.0 and -1.0 < self.chi_z_max < 1.0 and self.chi_z_max > self.chi_z_min):
            raise ValueError("chi_z_min/chi_z_max must be within (-1,1) and max>min")
        if self.spin_prior not in ("aligned_z", "isotropic"):
            raise ValueError(f"Unsupported spin_prior: {self.spin_prior}")
        if self.mass_prior not in ("loguniform", "uniform"):
            raise ValueError(f"Unsupported mass_prior: {self.mass_prior}")
        if self.q_prior not in ("uniform", "loguniform"):
            raise ValueError(f"Unsupported q_prior: {self.q_prior}")


@ADAPTERS.register("gw_merger_lal", description="Sample intrinsic BBH parameters for LALSimulation waveform generation.")
class GWMergerLALAdapter(WorldAdapter):
    NAME = "gw_merger_lal"
    DESCRIPTION = "Intrinsic BBH merger parameter sampler (for LALSimulation-based waveform generation). "

    def __init__(self, cfg: GWMergerLALAdapterConfig):
        cfg.validate()
        self.cfg = cfg

    def fingerprint(self) -> str:
        d = asdict(self.cfg)
        d["__class__"] = self.__class__.__name__
        # If LALSuite is installed, include versions to make caching safer.
        try:  # pragma: no cover
            import lal  # type: ignore
            import lalsimulation  # type: ignore
            d["lal_version"] = getattr(lal, "__version__", None)
            d["lalsimulation_version"] = getattr(lalsimulation, "__version__", None)
        except Exception:
            d["lal_version"] = None
            d["lalsimulation_version"] = None
        return hash_dict(d)

    def _sample_intrinsics(self, *, rng: np.random.Generator, n: int) -> List[Dict[str, Any]]:
        cfg = self.cfg

        if cfg.mass_prior == "loguniform":
            mchirp = _log_uniform(rng, cfg.mchirp_min_solar, cfg.mchirp_max_solar, size=n)
        else:
            mchirp = rng.uniform(cfg.mchirp_min_solar, cfg.mchirp_max_solar, size=n)

        if cfg.q_prior == "loguniform":
            q = _log_uniform(rng, cfg.q_min, cfg.q_max, size=n)
        else:
            q = rng.uniform(cfg.q_min, cfg.q_max, size=n)

        # Enforce q>=1 by construction; numerical safety:
        q = np.clip(q, 1.0, None)

        m1, m2 = _m1_m2_from_mchirp_q(mchirp, q)
        eta = _eta_from_q(q)

        if cfg.spin_prior == "aligned_z":
            chi1z = rng.uniform(cfg.chi_z_min, cfg.chi_z_max, size=n)
            chi2z = rng.uniform(cfg.chi_z_min, cfg.chi_z_max, size=n)
            # Optionally cap by chi_max
            chi1z = np.clip(chi1z, -cfg.chi_max, cfg.chi_max)
            chi2z = np.clip(chi2z, -cfg.chi_max, cfg.chi_max)
            chi1 = np.stack([np.zeros(n), np.zeros(n), chi1z], axis=1)
            chi2 = np.stack([np.zeros(n), np.zeros(n), chi2z], axis=1)
        else:
            chi1 = _sample_isotropic_spin_components(rng, chi_max=cfg.chi_max, n=n)
            chi2 = _sample_isotropic_spin_components(rng, chi_max=cfg.chi_max, n=n)

        # Effective aligned spin parameter (common invariant)
        chi_eff = (m1 * chi1[:, 2] + m2 * chi2[:, 2]) / (m1 + m2)

        out: List[Dict[str, Any]] = []
        for i in range(n):
            out.append(
                {
                    "id": int(i),
                    "intrinsics": {
                        "m1_solar": float(m1[i]),
                        "m2_solar": float(m2[i]),
                        "mchirp_solar": float(mchirp[i]),
                        "q": float(q[i]),
                        "eta": float(eta[i]),
                        "chi1": [float(chi1[i, 0]), float(chi1[i, 1]), float(chi1[i, 2])],
                        "chi2": [float(chi2[i, 0]), float(chi2[i, 1]), float(chi2[i, 2])],
                        "chi_eff": float(chi_eff[i]),
                    },
                    "meta": {
                        "approximant": cfg.approximant,
                    },
                }
            )
        return out

    def load(self) -> Dict[str, Any]:
        cfg = self.cfg
        rng_train = np.random.default_rng(cfg.seed)
        rng_test = np.random.default_rng(cfg.seed + 1337)

        train = self._sample_intrinsics(rng=rng_train, n=cfg.n_train)
        test = self._sample_intrinsics(rng=rng_test, n=cfg.n_test)

        return {
            "train": train,
            "test": test,
            "meta": {
                "cfg": asdict(cfg),
                "adapter": self.NAME,
            },
        }
