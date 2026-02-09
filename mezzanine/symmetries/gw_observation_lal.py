from __future__ import annotations

"""LALSimulation-based symmetry: detector observation (extrinsics + noise) for GW mergers.

Given a BBH "world instance" with *intrinsic* parameters, this symmetry samples
a *view* corresponding to:
  - sky orientation (theta, phi) and polarization (psi) -> antenna pattern (F+, Fx)
  - inclination (iota), reference phase (phi_ref)
  - distance
  - Gaussian noise realization (optionally PSD-shaped)

It then generates a frequency-domain waveform via LALSimulation and returns
a detector-projected strain representation.

Design notes
------------
- Inclination is treated as a symmetry here by regenerating the waveform per view.
  This is computationally heavier than storing modes, but avoids requiring mode APIs.
- Detector response uses the standard right-angle interferometer antenna patterns
  in a detector frame (not tied to a specific Earth location). This is "proper"
  in the sense of using the physical antenna response formulas while remaining
  detector-agnostic.
- For real-detector studies, you can extend this symmetry to use LAL detector
  geometry and sidereal time; the rest of the interface remains the same.

Returned view
-------------
{
  "freqs_hz": np.ndarray[F] float32,
  "strain_fd": np.ndarray[F] complex64,
  "psd": np.ndarray[F] float32 or None,
  "view_meta": {...}
}
"""

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np

from .base import Symmetry


# -----------------------------
# Helpers
# -----------------------------


def _msun_si() -> float:
    """Solar mass in SI units (kg)."""
    try:  # pragma: no cover
        import lal  # type: ignore
        return float(getattr(lal, "MSUN_SI"))
    except Exception:
        return 1.9884099021470417e30  # kg


def _pc_si() -> float:
    """Parsec in SI units (m)."""
    try:  # pragma: no cover
        import lal  # type: ignore
        return float(getattr(lal, "PC_SI"))
    except Exception:
        return 3.0856775814913673e16  # m


def _approximant_from_string(s: str):
    """Resolve an approximant name to a LALSimulation enum."""
    try:  # pragma: no cover
        import lalsimulation as lalsim  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "GWObservationLALSymmetry requires `lalsimulation` (LALSuite). "
            "Install LALSuite / lalsimulation in your environment."
        ) from e

    # Try a few common symbol names across LAL versions.
    for fn_name in (
        "GetApproximantFromString",
        "SimInspiralGetApproximantFromString",
        "GetApproximantFromStringSWIG",
    ):
        fn = getattr(lalsim, fn_name, None)
        if fn is not None:
            try:
                return fn(str(s))
            except Exception:
                pass

    # As a last resort, allow passing an integer string.
    try:
        return int(s)
    except Exception as e:
        raise ValueError(
            f"Could not resolve approximant '{s}'. "
            "Try a LAL name like 'IMRPhenomD', 'IMRPhenomXPHM', 'SEOBNRv4PHM', etc."
        ) from e


def _antenna_patterns(theta: float, phi: float, psi: float) -> Tuple[float, float]:
    """Right-angle interferometer antenna patterns (detector frame).

    theta: colatitude in detector frame [0, pi]
    phi: azimuth in detector frame [0, 2pi)
    psi: polarization angle [0, pi)

    Formulas match standard GR detector response for a 90° L-shaped detector.
    """
    ct = float(np.cos(theta))
    c2p = float(np.cos(2.0 * phi))
    s2p = float(np.sin(2.0 * phi))
    c2s = float(np.cos(2.0 * psi))
    s2s = float(np.sin(2.0 * psi))

    f_plus = 0.5 * (1.0 + ct * ct) * c2p * c2s - ct * s2p * s2s
    f_cross = 0.5 * (1.0 + ct * ct) * c2p * s2s + ct * s2p * c2s
    return f_plus, f_cross


def _interp_psd_from_file(freqs: np.ndarray, path: str) -> np.ndarray:
    """Load a 2-column (f, psd) text file and interpolate onto `freqs`."""
    data = np.loadtxt(path, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"PSD file must have at least 2 columns (f, psd); got shape {data.shape}")
    f0 = data[:, 0]
    s0 = data[:, 1]
    # Basic sanitization
    good = (f0 > 0) & np.isfinite(f0) & np.isfinite(s0) & (s0 > 0)
    f0 = f0[good]
    s0 = s0[good]
    if f0.size < 4:
        raise ValueError("PSD file does not contain enough valid samples.")
    # Interpolate in log-log for smoothness
    lf0 = np.log(f0)
    ls0 = np.log(s0)
    lf = np.log(np.clip(freqs, f0.min(), f0.max()))
    ls = np.interp(lf, lf0, ls0)
    return np.exp(ls).astype(np.float32)


def _complex_gaussian_noise_from_psd(
    rng: np.random.Generator, psd: np.ndarray, df: float
) -> np.ndarray:
    """Generate a complex frequency-series noise realization.

    Uses a common convention in GW analysis:
      E[|n(f_k)|^2] = 0.5 * S_n(f_k) / df

    If n = sigma*(a + i b), with a,b ~ N(0,1), then:
      E[|n|^2] = 2*sigma^2  => sigma = 0.5 * sqrt(S_n/df)
    """
    sigma = 0.5 * np.sqrt(np.clip(psd, 1e-30, None) / float(df))
    a = rng.normal(0.0, 1.0, size=psd.shape)
    b = rng.normal(0.0, 1.0, size=psd.shape)
    return (sigma * a + 1j * sigma * b).astype(np.complex64)


# -----------------------------
# Symmetry
# -----------------------------


@dataclass
class GWObservationLALConfig:
    approximant: Optional[str] = None  # default: use x.get("meta", {}).get("approximant")

    # Frequency grid for FD waveforms
    delta_f_hz: float = 1e-4
    f_lower_hz: float = 1e-4
    f_upper_hz: float = 0.1
    f_ref_hz: float = 0.0

    # Extrinsic priors
    distance_mpc_min: float = 100.0
    distance_mpc_max: float = 50000.0
    distance_prior: Literal["loguniform", "uniform"] = "loguniform"

    # Inclination prior
    # If True, isotropic: cos(iota) ~ Uniform(-1,1)
    isotropic_inclination: bool = True
    # Otherwise, fixed inclination (radians)
    inclination_rad: float = 0.0

    # Noise
    add_noise: bool = True
    psd_path: Optional[str] = None  # path to (f, psd) file; overrides psd_model if set
    psd_model: Optional[str] = None  # optional LAL PSD name/model (best-effort)
    whiten: bool = False  # if True, return strain/sqrt(psd)

    def validate(self) -> None:
        if not (self.delta_f_hz > 0):
            raise ValueError("delta_f_hz must be > 0")
        if not (self.f_lower_hz > 0):
            raise ValueError("f_lower_hz must be > 0")
        if not (self.f_upper_hz > self.f_lower_hz):
            raise ValueError("f_upper_hz must be > f_lower_hz")
        if not (self.distance_mpc_min > 0 and self.distance_mpc_max > self.distance_mpc_min):
            raise ValueError("distance_mpc_min/max must satisfy 0 < min < max")


class GWObservationLALSymmetry(Symmetry):
    NAME = "gw_observation_lal"
    DESCRIPTION = "Generate FD detector strain views via LALSimulation (extrinsics + noise)."

    def __init__(self, cfg: GWObservationLALConfig):
        cfg.validate()
        self.cfg = cfg

    def sample(self, x: Any, *, seed: int) -> Dict[str, Any]:
        cfg = self.cfg
        rng = np.random.default_rng(seed)

        intr = x.get("intrinsics", x.get("params", x))
        if intr is None:
            raise ValueError("Expected x to contain an 'intrinsics' dict.")

        m1_solar = float(intr["m1_solar"])
        m2_solar = float(intr["m2_solar"])
        chi1 = intr.get("chi1", [0.0, 0.0, 0.0])
        chi2 = intr.get("chi2", [0.0, 0.0, 0.0])

        # Extrinsics
        if cfg.distance_prior == "loguniform":
            d_mpc = float(np.exp(rng.uniform(np.log(cfg.distance_mpc_min), np.log(cfg.distance_mpc_max))))
        else:
            d_mpc = float(rng.uniform(cfg.distance_mpc_min, cfg.distance_mpc_max))
        distance_si = d_mpc * 1e6 * _pc_si()

        if cfg.isotropic_inclination:
            cosi = float(rng.uniform(-1.0, 1.0))
            iota = float(np.arccos(np.clip(cosi, -1.0, 1.0)))
        else:
            iota = float(cfg.inclination_rad)

        phi_ref = float(rng.uniform(0.0, 2.0 * np.pi))

        # Sky/polarization in detector frame
        cos_theta = float(rng.uniform(-1.0, 1.0))
        theta = float(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
        phi = float(rng.uniform(0.0, 2.0 * np.pi))
        psi = float(rng.uniform(0.0, np.pi))
        f_plus, f_cross = _antenna_patterns(theta, phi, psi)

        # Waveform generation (FD)
        try:  # pragma: no cover
            import lal  # type: ignore
            import lalsimulation as lalsim  # type: ignore
        except Exception as e:
            raise RuntimeError("GWObservationLALSymmetry requires `lal` + `lalsimulation` (LALSuite).") from e

        m1_si = m1_solar * _msun_si()
        m2_si = m2_solar * _msun_si()

        approx_name = (
            cfg.approximant
            or x.get("meta", {}).get("approximant")
            or x.get("meta", {}).get("waveform")
            or "IMRPhenomD"
        )
        approximant = _approximant_from_string(str(approx_name))

        # Params dict (empty by default)
        try:
            params = lal.CreateDict()
        except Exception:
            params = None

        # Some LAL versions require longAscNodes, eccentricity, meanPerAno even for quasi-circular.
        long_asc_nodes = 0.0
        eccentricity = 0.0
        mean_per_ano = 0.0

        df = float(cfg.delta_f_hz)
        fmin = float(cfg.f_lower_hz)
        fmax = float(cfg.f_upper_hz)
        fref = float(cfg.f_ref_hz)

        # Generate polarizations.
        # Note: different LALSuite versions expose different python-callable signatures.
        try:
            hp, hc = lalsim.SimInspiralChooseFDWaveform(
                m1_si,
                m2_si,
                float(chi1[0]),
                float(chi1[1]),
                float(chi1[2]),
                float(chi2[0]),
                float(chi2[1]),
                float(chi2[2]),
                distance_si,
                iota,
                phi_ref,
                long_asc_nodes,
                eccentricity,
                mean_per_ano,
                df,
                fmin,
                fmax,
                fref,
                params,
                approximant,
            )
        except TypeError:
            # Older signature variants: try without fmax.
            try:
                hp, hc = lalsim.SimInspiralChooseFDWaveform(
                    m1_si,
                    m2_si,
                    float(chi1[0]),
                    float(chi1[1]),
                    float(chi1[2]),
                    float(chi2[0]),
                    float(chi2[1]),
                    float(chi2[2]),
                    distance_si,
                    iota,
                    phi_ref,
                    long_asc_nodes,
                    eccentricity,
                    mean_per_ano,
                    df,
                    fmin,
                    fref,
                    params,
                    approximant,
                )
            except Exception as e:
                raise RuntimeError(
                    "Failed to call lalsimulation.SimInspiralChooseFDWaveform. "
                    "Your LALSuite version may expose a different signature."
                ) from e

        # Convert to numpy
        hp_fd = np.asarray(hp.data.data, dtype=np.complex64)
        hc_fd = np.asarray(hc.data.data, dtype=np.complex64)
        freqs = (np.arange(hp_fd.shape[0], dtype=np.float32) * df).astype(np.float32)

        # Detector projection
        strain = (f_plus * hp_fd + f_cross * hc_fd).astype(np.complex64)

        # PSD/noise
        psd: Optional[np.ndarray] = None
        if cfg.add_noise or cfg.whiten:
            if cfg.psd_path:
                psd = _interp_psd_from_file(freqs, cfg.psd_path)
            elif cfg.psd_model:
                # Best-effort: try to call a LAL PSD function if present.
                fn = None
                cand = f"SimNoisePSD{cfg.psd_model}"
                fn = getattr(lalsim, cand, None)
                if fn is None:
                    fn = getattr(lalsim, cfg.psd_model, None)
                if fn is not None:
                    try:
                        psd_series = lal.CreateREAL8FrequencySeries(
                            "psd",
                            lal.LIGOTimeGPS(0),
                            0.0,
                            df,
                            lal.DimensionlessUnit,
                            hp_fd.shape[0],
                        )
                        # Fill psd_series.data.data in-place.
                        lalsim.SimNoisePSD(psd_series, 0, fn)  # type: ignore
                        psd = np.asarray(psd_series.data.data, dtype=np.float32)
                    except Exception:
                        psd = None
                # If still None, fall back to white PSD.

            if psd is None:
                psd = np.ones_like(freqs, dtype=np.float32)

        if cfg.add_noise:
            noise = _complex_gaussian_noise_from_psd(rng, psd, df)
            strain = (strain + noise).astype(np.complex64)

        if cfg.whiten:
            denom = np.sqrt(np.clip(psd, 1e-30, None)).astype(np.float32)
            strain = (strain / denom).astype(np.complex64)

        return {
            "freqs_hz": freqs,
            "strain_fd": strain,
            "psd": psd,
            "view_meta": {
                "distance_mpc": d_mpc,
                "inclination_rad": iota,
                "phi_ref": phi_ref,
                "sky_theta": theta,
                "sky_phi": phi,
                "psi": psi,
                "f_plus": float(f_plus),
                "f_cross": float(f_cross),
                "approximant": str(approx_name),
            },
        }


# Register
from ..registry import SYMMETRIES

SYMMETRIES.register("gw_observation_lal")(GWObservationLALSymmetry)
