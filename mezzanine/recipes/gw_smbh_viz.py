
from __future__ import annotations

"""GW SMBH merger visualization recipe (GR waveform + honest visuals).

Run:
  mezzanine run gw_smbh_viz --out OUTDIR [args]

This recipe:
  1) samples intrinsic BBH parameters with `GWMergerLALAdapter`
  2) samples multiple symmetry "views" (extrinsics + optional noise) with `GWObservationLALSymmetry`
  3) saves **waveform-based** visuals:
        - |h(f)| overlays across symmetry views
        - band-limited reconstructed time-domain strain
        - spectrogram
        - a **chirp.gif** animation derived from the waveform itself (not a cartoon orbit)

Notes
-----
- The waveform is produced by LALSuite (lalsimulation) through the symmetry.
- The time-series reconstruction uses an inverse real FFT of the one-sided FD strain.
  For smoother plots/animation, we optionally **zero-pad** the FD series up to a higher
  Nyquist frequency. This is *band-limited interpolation* (no new physics is added).
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _ensure_matplotlib() -> None:
    # Import lazily so `mezzanine list` still works in minimal environments.
    import matplotlib  # noqa: F401


def _fd_to_td_irfft_padded(
    freqs_hz: np.ndarray,
    h_fd: np.ndarray,
    *,
    pad_to_f_nyquist_hz: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Convert a one-sided complex frequency series to a real time series via irfft.

    Assumptions:
      - freqs are uniform starting at 0 with step df
      - h_fd corresponds to positive frequencies (including DC)
      - we reconstruct a real time-series via np.fft.irfft

    If pad_to_f_nyquist_hz is set > freqs[-1], we zero-pad the FD series up to that
    Nyquist. This increases the time sampling rate (smaller dt) while keeping the
    same total duration (T = 1/df). It's a band-limited interpolation.
    """
    freqs = np.asarray(freqs_hz, dtype=np.float64)
    h_fd = np.asarray(h_fd)

    if freqs.ndim != 1 or h_fd.ndim != 1 or freqs.shape[0] != h_fd.shape[0]:
        raise ValueError("freqs_hz and strain_fd must be 1D arrays of equal length")
    if freqs.shape[0] < 4:
        raise ValueError("Need at least a few frequency bins")

    df = float(freqs[1] - freqs[0])
    if not np.allclose(np.diff(freqs), df, rtol=1e-4, atol=1e-12):
        raise ValueError("freqs_hz must be uniformly spaced")

    # Optionally pad
    n_pos = int(freqs.shape[0])
    h_pos = np.array(h_fd, dtype=np.complex128, copy=True)

    if pad_to_f_nyquist_hz is not None:
        f_nyq_target = float(pad_to_f_nyquist_hz)
        if f_nyq_target > float(freqs[-1]) + 0.5 * df:
            n_pos_pad = int(np.floor(f_nyq_target / df)) + 1
            if n_pos_pad > n_pos:
                h_pad = np.zeros((n_pos_pad,), dtype=np.complex128)
                h_pad[:n_pos] = h_pos
                h_pos = h_pad
                n_pos = n_pos_pad

    # irfft expects length N//2+1 = n_pos; output length N = 2*(n_pos-1)
    n_time = 2 * (n_pos - 1)
    dt = 1.0 / (n_time * df)
    t = np.arange(n_time, dtype=np.float64) * dt

    # Ensure DC and Nyquist are real for a real-valued signal.
    h_pos[0] = complex(float(np.real(h_pos[0])), 0.0)
    h_pos[-1] = complex(float(np.real(h_pos[-1])), 0.0)

    x = np.fft.irfft(h_pos, n=n_time)
    return t.astype(np.float64), x.astype(np.float64), float(dt)


def _estimate_envelope_and_inst_freq(t: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return amplitude envelope and instantaneous frequency estimate.

    Uses Hilbert transform to compute an analytic signal, then differentiates its phase.
    """
    from scipy.signal import hilbert, medfilt

    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if t.ndim != 1 or x.ndim != 1 or t.shape[0] != x.shape[0]:
        raise ValueError("t and x must be 1D arrays of equal length")
    if t.shape[0] < 16:
        env = np.abs(x)
        finst = np.full_like(env, np.nan, dtype=np.float64)
        return env, finst

    # Remove DC offset for stability
    x0 = x - float(np.mean(x))

    a = hilbert(x0)
    env = np.abs(a)
    phase = np.unwrap(np.angle(a))
    dphi_dt = np.gradient(phase, t)
    finst = dphi_dt / (2.0 * np.pi)

    # Smooth a bit (median filter) to reduce phase slips
    k = min(31, max(5, (t.shape[0] // 100) | 1))  # odd kernel
    if k % 2 == 0:
        k += 1
    if k >= 5 and k < t.shape[0]:
        finst = medfilt(finst, kernel_size=k)

    return env.astype(np.float64), finst.astype(np.float64)


def _simple_view_feature(view: Dict[str, Any], *, fmin: float, fmax: float) -> np.ndarray:
    """A fast, dependency-free feature: standardized log-magnitude spectrum."""
    f = np.asarray(view["freqs_hz"], dtype=np.float64)
    h = np.asarray(view["strain_fd"])
    mag = np.abs(h).astype(np.float64)

    sel = (f >= float(fmin)) & (f <= float(fmax)) & np.isfinite(f) & np.isfinite(mag)
    if sel.sum() < 8:
        sel = (f > 0) & np.isfinite(f) & np.isfinite(mag)
    v = np.log(mag[sel] + 1e-30)
    v = (v - v.mean()) / (v.std() + 1e-6)
    return v.astype(np.float32)


def _pca_2d(X: np.ndarray) -> np.ndarray:
    """2D PCA via SVD; returns [N,2]."""
    X = np.asarray(X, dtype=np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    U, S, _Vt = np.linalg.svd(X, full_matrices=False)
    Z = U[:, :2] * S[:2]
    return Z.astype(np.float32)


def _f_isco_gw_hz(m_total_solar: float) -> float:
    """GW frequency at Schwarzschild ISCO for the dominant quadrupole (approx)."""
    G = 6.67430e-11
    c = 299792458.0
    MSUN = 1.98847e30
    M = float(m_total_solar) * MSUN
    return float(c**3 / (np.pi * (6.0 ** 1.5) * G * M))


def _chirp_gif_from_td(
    *,
    t: np.ndarray,
    x: np.ndarray,
    out_path: Path,
    title: str,
    seconds_before_peak: float,
    window_seconds: float,
    fps: int,
    n_frames: int,
    f_ylim_hz: Optional[float] = None,
    annotate: Optional[str] = None,
) -> Dict[str, float]:
    """Create an animation from the reconstructed time-domain strain.

    The "merger time" is approximated as the peak of the amplitude envelope.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)

    env, finst = _estimate_envelope_and_inst_freq(t, x)

    # Define peak as "merger"
    i_peak = int(np.nanargmax(env))
    t_peak = float(t[i_peak])

    # Segment we animate: [t_peak - seconds_before_peak, t_peak]
    t0 = float(max(float(t[0]), t_peak - float(seconds_before_peak)))
    t1 = float(t_peak)
    sel = (t >= t0) & (t <= t1)
    t_seg = t[sel]
    x_seg = x[sel]
    env_seg = env[sel]
    finst_seg = finst[sel]

    # Robust defaults if segment is too short
    if t_seg.size < 32:
        # Fall back to last part of series
        sel = t >= (t_peak - min(float(seconds_before_peak), float(t[-1] - t[0])))
        t_seg = t[sel]
        x_seg = x[sel]
        env_seg = env[sel]
        finst_seg = finst[sel]

    # Relative time axis for readability
    tr = t_seg - t_peak

    # Plot limits
    # Display window (seconds) in the animation axes; we slide it as time advances.
    win = float(max(1.0, min(float(window_seconds), float(seconds_before_peak))))
    x_lim = (-win, 0.0)
    # Use a robust y-limit based on envelope
    y_max = float(np.quantile(np.abs(x_seg), 0.999)) if x_seg.size else 1.0
    if not np.isfinite(y_max) or y_max <= 0:
        y_max = float(np.max(np.abs(x_seg))) if x_seg.size else 1.0
    y_max = max(y_max, 1e-24)

    # Instantaneous frequency range
    f_good = np.isfinite(finst_seg) & (finst_seg > 0)
    f95 = float(np.quantile(finst_seg[f_good], 0.95)) if np.any(f_good) else (f_ylim_hz or 1.0)
    f_top = float(f_ylim_hz) if f_ylim_hz is not None else max(1e-6, 1.25 * f95)

    # Choose frame times
    n_frames = int(max(16, n_frames))
    frame_times = np.linspace(tr.min(), 0.0, n_frames)

    # Figure: 2 rows
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    fig.suptitle(title)

    # Top: strain
    ax1.plot(tr, x_seg, linewidth=1.0, alpha=0.9, label="strain (band-limited)")
    ax1.plot(tr, env_seg, linewidth=1.0, alpha=0.7, label="envelope")
    ax1.plot(tr, -env_seg, linewidth=1.0, alpha=0.7)
    vline1 = ax1.axvline(tr.min(), linewidth=1.0)
    ax1.set_xlim(*x_lim)
    ax1.set_ylim(-1.1 * y_max, 1.1 * y_max)
    ax1.set_ylabel("strain")
    ax1.legend(loc="upper left", fontsize=8)

    if annotate:
        ax1.text(
            0.01, 0.98, annotate,
            transform=ax1.transAxes,
            va="top", ha="left",
            fontsize=9,
        )

    # Bottom: instantaneous frequency
    ax2.plot(tr, finst_seg, linewidth=1.0)
    vline2 = ax2.axvline(tr.min(), linewidth=1.0)
    ax2.set_xlim(*x_lim)
    ax2.set_ylim(0.0, f_top)
    ax2.set_xlabel("time (s) relative to peak amplitude")
    ax2.set_ylabel("f_inst (Hz)")

    def update(i: int):
        ti = float(frame_times[i])
        # Slide the visible window so you can see oscillations clearly.
        x_left = max(float(tr.min()), ti - win)
        x_right = min(0.0, x_left + win)
        ax1.set_xlim(x_left, x_right)
        ax2.set_xlim(x_left, x_right)
        vline1.set_xdata([ti, ti])
        vline2.set_xdata([ti, ti])
        return (vline1, vline2)

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / max(1, int(fps)), blit=False)
    anim.save(str(out_path), writer=PillowWriter(fps=int(fps)))
    plt.close(fig)

    return {"t_peak_s": t_peak, "t0_s": t0, "t1_s": t1, "f_inst_p95_hz": float(f95)}


from ..core.config import deep_update, load_config
from .recipe_base import Recipe


class GWSMBHVizRecipe(Recipe):
    NAME = "gw_smbh_viz"
    DESCRIPTION = "Visualize LALSimulation SMBH BBH mergers with symmetry views (GR waveform visuals + chirp GIF)."

    def run(self, argv: list[str]) -> Dict[str, Any]:
        p = argparse.ArgumentParser(prog=self.NAME)
        self.add_common_args(p)

        # World (intrinsics)
        p.add_argument("--n_events", type=int, default=8, help="Number of underlying mergers to sample (for PCA plot).")
        p.add_argument("--event_index", type=int, default=0, help="Which sampled merger to visualize in detail.")
        p.add_argument("--mchirp_min_solar", type=float, default=1e5)
        p.add_argument("--mchirp_max_solar", type=float, default=1e7)
        p.add_argument("--q_min", type=float, default=1.0)
        p.add_argument("--q_max", type=float, default=10.0)
        p.add_argument("--spin_prior", type=str, default="isotropic", choices=["aligned_z", "isotropic"])
        p.add_argument("--approximant", type=str, default="IMRPhenomXPHM")

        # Symmetry sampling
        p.add_argument("--k_views", type=int, default=8, help="How many symmetry views to draw for the chosen event.")
        p.add_argument("--delta_f_hz", type=float, default=1e-4)
        p.add_argument("--f_lower_hz", type=float, default=1e-4)
        p.add_argument("--f_upper_hz", type=float, default=0.1)
        p.add_argument("--distance_mpc_min", type=float, default=100.0)
        p.add_argument("--distance_mpc_max", type=float, default=50000.0)
        p.add_argument("--add_noise", action="store_true", help="Add a noise realization in each view.")
        p.add_argument("--no_noise", action="store_true", help="Force noise off (overrides --add_noise).")
        p.add_argument("--whiten", action="store_true")
        p.add_argument("--psd_path", type=str, default=None)

        # Visualization knobs
        p.add_argument("--plot_fmin_hz", type=float, default=1e-4, help="Min frequency to plot/featurize.")
        p.add_argument("--plot_fmax_hz", type=float, default=0.02, help="Max frequency to plot/featurize.")
        p.add_argument("--plot_last_seconds", type=float, default=2000.0, help="Plot last N seconds before peak amplitude.")

        # For smoother TD plots, pad the FD series up to this Nyquist (Hz).
        p.add_argument("--td_pad_to_f_nyquist_hz", type=float, default=0.5)

        # Waveform-based GIF (recommended)
        p.add_argument("--make_chirp_gif", action="store_true", help="Write chirp.gif based on the reconstructed strain.")
        p.add_argument("--chirp_seconds", type=float, default=3000.0, help="How many seconds before peak to animate.")
        p.add_argument("--chirp_window_seconds", type=float, default=600.0, help="Window length shown in the animation axes.")
        p.add_argument("--chirp_fps", type=int, default=24)
        p.add_argument("--chirp_frames", type=int, default=240)

        p.add_argument("--out_dir", type=str, default=str(self.out_dir))

        args = p.parse_args(argv)

        # Apply config file defaults BEFORE build_context
        file_cfg = load_config(getattr(args, "config", None))
        merged_cfg = deep_update(file_cfg, self.config)
        self.apply_config_defaults(p, args, merged_cfg)

        ctx = self.build_context(args)
        out_dir = ctx.out_dir

        _ensure_matplotlib()
        import matplotlib.pyplot as plt

        # --- Build world + symmetry ---
        from ..worlds.gw_merger_lal import GWMergerLALAdapter, GWMergerLALAdapterConfig
        from ..symmetries.gw_observation_lal import GWObservationLALConfig, GWObservationLALSymmetry

        n_events = int(max(1, args.n_events))
        wcfg = GWMergerLALAdapterConfig(
            seed=int(args.seed),
            n_train=n_events,
            n_test=1,
            mchirp_min_solar=float(args.mchirp_min_solar),
            mchirp_max_solar=float(args.mchirp_max_solar),
            q_min=float(args.q_min),
            q_max=float(args.q_max),
            spin_prior=str(args.spin_prior),
            approximant=str(args.approximant),
        )
        world = GWMergerLALAdapter(wcfg)
        data = world.load()
        events = data["train"]

        idx = int(np.clip(int(args.event_index), 0, len(events) - 1))
        x = events[idx]
        intr = x["intrinsics"]

        # Auto-cap f_upper at (approx) ISCO for very massive systems
        m_total = float(intr["m1_solar"] + intr["m2_solar"])
        f_isco = _f_isco_gw_hz(m_total)
        f_upper = float(args.f_upper_hz)
        f_upper_eff = float(min(f_upper, 1.5 * f_isco))

        add_noise = bool(args.add_noise) and (not bool(args.no_noise))
        ncfg = GWObservationLALConfig(
            approximant=str(args.approximant),
            delta_f_hz=float(args.delta_f_hz),
            f_lower_hz=float(args.f_lower_hz),
            f_upper_hz=f_upper_eff,
            distance_mpc_min=float(args.distance_mpc_min),
            distance_mpc_max=float(args.distance_mpc_max),
            add_noise=add_noise,
            psd_path=str(args.psd_path) if args.psd_path else None,
            whiten=bool(args.whiten),
        )
        nuis = GWObservationLALSymmetry(ncfg)

        # --- Sample K symmetry views for the chosen event ---
        K = int(max(2, args.k_views))
        views: List[Dict[str, Any]] = []
        for k in range(K):
            v = nuis.sample(x, seed=int(args.seed + 10_000 + k))
            views.append(v)

        # --- Save intrinsics + view metadata ---
        (out_dir / "event_intrinsics.json").write_text(json.dumps(intr, indent=2))
        with (out_dir / "views_meta.jsonl").open("w") as f:
            for i, v in enumerate(views):
                meta = dict(v.get("view_meta", {}))
                meta["view_index"] = i
                f.write(json.dumps(meta) + "\n")

        # --- Frequency-domain overlay ---
        fmin_plot = float(args.plot_fmin_hz)
        fmax_plot = float(args.plot_fmax_hz)
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(1, 1, 1)
        for v in views:
            f = np.asarray(v["freqs_hz"], dtype=np.float64)
            h = np.asarray(v["strain_fd"])
            sel = (f >= fmin_plot) & (f <= fmax_plot) & np.isfinite(f)
            ax.plot(f[sel], np.abs(h[sel]))
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("|strain(f)|")
        ax.set_title(f"SMBH BBH: {K} symmetry views overlay (event {idx})")
        ax.set_yscale("log")
        fig.tight_layout()
        fig.savefig(out_dir / "fd_magnitude_overlay.png", dpi=200)
        plt.close(fig)

        # --- Time-domain reconstruction (band-limited), align by peak envelope ---
        pad_nyq = float(args.td_pad_to_f_nyquist_hz) if args.td_pad_to_f_nyquist_hz else None
        td_traces: List[Tuple[np.ndarray, np.ndarray]] = []
        for v in views:
            t, s, _dt = _fd_to_td_irfft_padded(v["freqs_hz"], v["strain_fd"], pad_to_f_nyquist_hz=pad_nyq)
            td_traces.append((t, s))

        # Choose a reference "merger time" as the peak envelope of the first view
        t_ref, s_ref = td_traces[0]
        env_ref, _f_inst_ref = _estimate_envelope_and_inst_freq(t_ref, s_ref)
        i_peak = int(np.nanargmax(env_ref))
        t_peak = float(t_ref[i_peak])

        # Plot window: last plot_last_seconds before peak
        last_s = float(max(0.0, args.plot_last_seconds))
        t_start = max(float(t_ref[0]), t_peak - last_s)
        t_end = t_peak

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(1, 1, 1)
        for (t, s) in td_traces:
            sel = (t >= t_start) & (t <= t_end)
            ax.plot(t[sel] - t_peak, s[sel], linewidth=1.0, alpha=0.9)
        ax.set_xlabel("Time (s) relative to peak amplitude")
        ax.set_ylabel("strain(t)")
        ax.set_title(f"Band-limited reconstructed strain (last {last_s:g} s)")
        fig.tight_layout()
        fig.savefig(out_dir / "td_strain_overlay.png", dpi=200)
        plt.close(fig)

        # --- Spectrogram for the first view (same aligned window) ---
        sel = (t_ref >= t_start) & (t_ref <= t_end)
        s_seg = s_ref[sel]
        t_seg = t_ref[sel] - t_peak

        # Choose NFFT based on segment length
        n = int(s_seg.shape[0])
        # Use a power of 2 <= n/4 (but at least 64)
        nfft = 64
        if n >= 256:
            nfft = 2 ** int(np.floor(np.log2(max(64, min(4096, n // 4)))))
        noverlap = int(0.75 * nfft)

        dt = float(t_ref[1] - t_ref[0])
        fs = 1.0 / dt

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(1, 1, 1)
        ax.specgram(s_seg, NFFT=int(nfft), Fs=float(fs), noverlap=int(noverlap))
        ax.set_ylim(0.0, min(fmax_plot, 0.5 * fs))
        ax.set_xlabel("Time (s) relative to peak")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title("Spectrogram (one symmetry view, band-limited reconstruction)")
        fig.tight_layout()
        fig.savefig(out_dir / "spectrogram.png", dpi=200)
        plt.close(fig)

        # --- Simple "latent" PCA plot across events ---
        rng = np.random.default_rng(int(args.seed) + 2025)
        views_per_event = 4
        feats: List[np.ndarray] = []
        labels: List[int] = []
        for ei, ex in enumerate(events):
            for _ in range(views_per_event):
                v = nuis.sample(ex, seed=int(rng.integers(0, 2**31 - 1)))
                feats.append(_simple_view_feature(v, fmin=fmin_plot, fmax=fmax_plot))
                labels.append(ei)
        D = min(f.shape[0] for f in feats)
        X = np.stack([f[:D] for f in feats], axis=0)
        Z = _pca_2d(X)

        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(1, 1, 1)
        for ei in range(len(events)):
            sel = np.array(labels) == ei
            ax.scatter(Z[sel, 0], Z[sel, 1], label=f"event {ei}", alpha=0.8)
        ax.set_title("View features PCA (points should cluster by intrinsic event)")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(fontsize=8, ncols=2)
        fig.tight_layout()
        fig.savefig(out_dir / "latent_pca.png", dpi=200)
        plt.close(fig)

        chirp_meta: Optional[Dict[str, float]] = None
        chirp_path: Optional[str] = None
        if bool(args.make_chirp_gif):
            annotate = (
                f"m1={intr['m1_solar']:.3g} Msun  m2={intr['m2_solar']:.3g} Msun\n"
                f"q={intr['q']:.3g}  Mc={intr['mchirp_solar']:.3g} Msun\n"
                f"approx={args.approximant}  f_upper_eff≈{f_upper_eff:.3g} Hz\n"
                f"(band-limited reconstruction; zero-padding for smooth dt)"
            )
            chirp_meta = _chirp_gif_from_td(
                t=t_ref,
                x=s_ref,
                out_path=out_dir / "chirp.gif",
                title="GR waveform (detector strain) — time compressed only by playback",
                seconds_before_peak=float(args.chirp_seconds),
                window_seconds=float(args.chirp_window_seconds),
                fps=int(args.chirp_fps),
                n_frames=int(args.chirp_frames),
                f_ylim_hz=min(fmax_plot, max(1e-6, 2.0 * f_upper_eff)),
                annotate=annotate,
            )
            chirp_path = str(out_dir / "chirp.gif")

        result = {
            "exp": self.NAME,
            "world": {"adapter": "gw_merger_lal", "config": asdict(wcfg), "meta": data.get("meta", {})},
            "symmetry": {"name": "gw_observation_lal", "config": asdict(ncfg)},
            "chosen_event_index": idx,
            "intrinsics": intr,
            "derived": {"f_isco_gw_hz": f_isco, "f_upper_eff_hz": f_upper_eff, "chirp_meta": chirp_meta},
            "artifacts": {
                "event_intrinsics_json": str(out_dir / "event_intrinsics.json"),
                "views_meta_jsonl": str(out_dir / "views_meta.jsonl"),
                "fd_magnitude_overlay_png": str(out_dir / "fd_magnitude_overlay.png"),
                "td_strain_overlay_png": str(out_dir / "td_strain_overlay.png"),
                "spectrogram_png": str(out_dir / "spectrogram.png"),
                "latent_pca_png": str(out_dir / "latent_pca.png"),
                "chirp_gif": chirp_path,
            },
        }

        (out_dir / "results.json").write_text(json.dumps(result, indent=2))

        # Log artifacts if a logger is configured
        try:
            ctx.logger.log_artifact(out_dir / "results.json", name="results.json")
            for k, v in result["artifacts"].items():
                if v and Path(v).exists():
                    ctx.logger.log_artifact(Path(v), name=Path(v).name)
        finally:
            try:
                ctx.logger.close()
            except Exception:
                pass

        return result
