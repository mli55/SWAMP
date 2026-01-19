"""
Process simulated SFCW “received” signals to detect a buried target:
- synthesize a measurement with target + clutter + LOS (optional)
- synthesize a reference without target
- subtract reference, window in delay domain, estimate target delay/energy/size
- visualize frequency and time responses (show, not save)

Usage:
  python3 scripts/process_sfcw_received.py \
    --target-scale 1.5 \
    --los-coupling 1.0 \
    --noise-std 0.0
"""

import os
import argparse
import sys
import pathlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# Ensure repo root is on sys.path for in-place imports
ROOT = pathlib.Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
(ROOT / ".mplconfig").mkdir(exist_ok=True)

import matplotlib


def _init_backend():
    """
    Default to Agg for headless safety; users can override via MPLBACKEND env (e.g., TkAgg).
    """
    if "MPLBACKEND" in os.environ:
        return
    matplotlib.use("Agg")


_init_backend()
import matplotlib.pyplot as plt

sys.path.insert(0, str(ROOT))

from rfspsim.media.medium import Medium
from rfspsim.geometry.sampling import sample_ellipse_area, sample_surface_line
from rfspsim.propagation.channel_builders import (
    build_los_taps,
    build_surface_scatter_taps,
    build_target_reflection_taps,
)
from rfspsim.simulators.sfcw_sim import simulate_sfcw


@dataclass
class SceneConfig:
    soil_epsilon_r: float = 4.0
    target_scale: float = 1.0
    target_on: bool = True
    los_coupling: float = 1.0
    surface_reflectivity: float = 0.15


def build_scene(cfg: SceneConfig) -> Dict[str, np.ndarray]:
    air = Medium("air", epsilon_r=1.0, mu_r=1.0)
    soil = Medium("soil", epsilon_r=cfg.soil_epsilon_r, mu_r=1.0)
    interface_z = 0.0

    tx = np.array([-0.5, -0.05])
    rx = np.array([+0.5, -0.05])

    surface_pts, cell_len = sample_surface_line(
        x_min=-1.5, x_max=1.5, n_points=600, z=interface_z, method="uniform"
    )
    surf = build_surface_scatter_taps(
        tx,
        rx,
        surface_pts,
        air=air,
        soil=soil,
        interface_z=interface_z,
        per_point_length=cell_len,
        surface_reflectivity=cfg.surface_reflectivity,
        pol="avg",
        include_fresnel=True,
    )

    targ = None
    target_pts = None
    if cfg.target_on and cfg.target_scale > 0.0:
        center = (0.0, 0.20)
        a = 0.18 * cfg.target_scale  # half major axis (m)
        b = 0.06 * cfg.target_scale  # half minor axis (m)
        target_pts = sample_ellipse_area(center=center, a=a, b=b, n_points=800, seed=0)
        cell_area = np.pi * a * b / len(target_pts)
        targ = build_target_reflection_taps(
            tx,
            rx,
            target_pts,
            air=air,
            soil=soil,
            interface_z=interface_z,
            per_point_area=cell_area,
            target_reflectivity=1.0 + 0j,
            pol="avg",
            include_fresnel=True,
        )

    los = build_los_taps(
        tx, rx, air=air, los_coupling=cfg.los_coupling + 0j, model="inv_d2"
    )

    delays_parts = [los["delays"], surf["delays"]]
    gains_parts = [los["gains"], surf["gains"]]
    if targ is not None:
        delays_parts.append(targ["delays"])
        gains_parts.append(targ["gains"])

    return {
        "tx": tx,
        "rx": rx,
        "surface_pts": surface_pts,
        "target_pts": target_pts,
        "los": los,
        "surf": surf,
        "targ": targ,
        "delays": np.concatenate(delays_parts),
        "gains": np.concatenate(gains_parts),
    }


def simulate(delays: np.ndarray, gains: np.ndarray, noise_std: float = 0.0):
    res = simulate_sfcw(
        f_start=2e9,
        f_step=40e6,
        n_steps=51,
        delays=delays,
        gains=gains,
        baseband=True,
        probe_tone_hz=None,
    )
    S = res["S"]
    if noise_std > 0.0:
        noise = (np.random.normal(scale=noise_std, size=S.shape) +
                 1j * np.random.normal(scale=noise_std, size=S.shape))
        S = S + noise
    res["S_noisy"] = S
    res["h_noisy"] = np.fft.ifft(S)
    return res


def preprocess(S_meas: np.ndarray, S_ref: Optional[np.ndarray], t: np.ndarray,
               target_window_ns: Tuple[float, float], early_window_ns: Tuple[float, float]):
    # Reference subtraction
    S_diff = S_meas - S_ref if S_ref is not None else S_meas
    h_diff = np.fft.ifft(S_diff)

    # Simple early-window gating (zero out LOS/clutter window in time domain)
    t_ns = t * 1e9
    mask_early = (t_ns >= early_window_ns[0]) & (t_ns <= early_window_ns[1])
    h_gated = h_diff.copy()
    h_gated[mask_early] = 0.0
    S_gated = np.fft.fft(h_gated)

    # Target window metrics
    mask_target = (t_ns >= target_window_ns[0]) & (t_ns <= target_window_ns[1])
    h_tgt = h_gated[mask_target]
    t_tgt = t_ns[mask_target]
    peak_idx = int(np.argmax(np.abs(h_tgt))) if h_tgt.size else 0
    peak_delay_ns = float(t_tgt[peak_idx]) if h_tgt.size else float("nan")
    peak_amp = float(np.abs(h_tgt[peak_idx])) if h_tgt.size else float("nan")
    energy = float(np.sum(np.abs(h_tgt) ** 2))

    return {
        "S_diff": S_diff,
        "h_diff": h_diff,
        "S_gated": S_gated,
        "h_gated": h_gated,
        "peak_delay_ns": peak_delay_ns,
        "peak_amp": peak_amp,
        "target_energy": energy,
    }


def calibrate_energy(scales: List[float], cfg: SceneConfig, target_window_ns: Tuple[float, float],
                     early_window_ns: Tuple[float, float], noise_std: float = 0.0) -> Dict[float, float]:
    energies = {}
    for scale in scales:
        cfg_scale = SceneConfig(
            soil_epsilon_r=cfg.soil_epsilon_r,
            target_scale=scale,
            target_on=scale > 0,
            los_coupling=cfg.los_coupling,
            surface_reflectivity=cfg.surface_reflectivity,
        )
        scene = build_scene(cfg_scale)
        res = simulate(scene["delays"], scene["gains"], noise_std=noise_std)
        t = res["t"]
        proc = preprocess(res["S_noisy"], None, t, target_window_ns, early_window_ns)
        energies[scale] = proc["target_energy"]
    return energies


def estimate_size(energy: float, calib: Dict[float, float]) -> float:
    # Fit energy ~ k * scale (through origin) using least squares on calibration data
    xs = np.array(list(calib.keys()), dtype=float)
    ys = np.array(list(calib.values()), dtype=float)
    mask = xs > 0
    if not np.any(mask):
        return float("nan")
    k = float(np.sum(xs[mask] * ys[mask]) / np.sum(xs[mask] ** 2))
    return energy / k if k > 0 else float("nan")


def plot_frequency(freqs: np.ndarray, S_meas: np.ndarray, S_ref: Optional[np.ndarray], S_diff: np.ndarray):
    plt.figure("Freq magnitude")
    freq_g = freqs / 1e9
    plt.plot(freq_g, 20 * np.log10(np.abs(S_meas) + 1e-12), label="meas")
    if S_ref is not None:
        plt.plot(freq_g, 20 * np.log10(np.abs(S_ref) + 1e-12), label="ref (no target)")
    plt.plot(freq_g, 20 * np.log10(np.abs(S_diff) + 1e-12), label="meas - ref")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("|S(f)| (dB)")
    plt.title("Frequency response before/after reference subtraction")
    plt.grid(True, alpha=0.3)
    plt.legend()


def plot_time(t: np.ndarray, h_meas: np.ndarray, h_ref: Optional[np.ndarray], h_diff: np.ndarray,
              h_gated: np.ndarray, target_window_ns: Tuple[float, float], early_window_ns: Tuple[float, float]):
    plt.figure("Time magnitude")
    t_ns = t * 1e9
    plt.plot(t_ns, np.abs(h_meas), label="meas")
    if h_ref is not None:
        plt.plot(t_ns, np.abs(h_ref), label="ref (no target)")
    plt.plot(t_ns, np.abs(h_diff), label="meas - ref")
    plt.plot(t_ns, np.abs(h_gated), label="gated (early nulled)")
    plt.axvspan(early_window_ns[0], early_window_ns[1], color="gray", alpha=0.15, label="early gate")
    plt.axvspan(target_window_ns[0], target_window_ns[1], color="orange", alpha=0.15, label="target window")
    plt.xlabel("Delay (ns)")
    plt.ylabel("|h(t)| (linear)")
    plt.title("Range profile and gating")
    plt.grid(True, alpha=0.3)
    plt.legend()


def plot_calibration(calib: Dict[float, float], energy_meas: float, size_est: float):
    plt.figure("Energy vs. size calibration")
    xs = np.array(sorted(calib.keys()))
    ys = np.array([calib[x] for x in xs])
    plt.plot(xs, ys, "o-", label="calibration")
    plt.axhline(energy_meas, color="r", linestyle="--", label="meas energy")
    plt.axvline(size_est, color="g", linestyle="--", label=f"estimated scale ~ {size_est:.2f}x")
    plt.xlabel("Target scale (x)")
    plt.ylabel("Target window energy (|h|^2 sum)")
    plt.title("Energy vs. target size (through-origin fit)")
    plt.grid(True, alpha=0.3)
    plt.legend()


def main():
    parser = argparse.ArgumentParser(description="Process SFCW received signal to estimate target delay/size.")
    parser.add_argument("--target-scale", type=float, default=1.0, help="Scale factor for target size.")
    parser.add_argument("--los-coupling", type=float, default=1.0, help="LOS coupling amplitude.")
    parser.add_argument("--noise-std", type=float, default=0.0, help="AWGN std dev on complex S(f).")
    parser.add_argument("--target-window", type=float, nargs=2, default=[7.0, 12.0], help="Target window ns.")
    parser.add_argument("--early-window", type=float, nargs=2, default=[0.0, 5.0], help="Early gate ns.")
    parser.add_argument("--calib-scales", nargs="+", type=float, default=[0.5, 1.0, 2.0],
                        help="Scales to build energy calibration curve.")
    args = parser.parse_args()

    # Reference (no target)
    ref_scene = build_scene(SceneConfig(target_on=False, los_coupling=args.los_coupling))
    ref_res = simulate(ref_scene["delays"], ref_scene["gains"], noise_std=args.noise_std)
    S_ref = ref_res["S_noisy"]
    h_ref = ref_res["h_noisy"]

    # Measurement (with target)
    meas_scene = build_scene(SceneConfig(target_scale=args.target_scale, target_on=True,
                                         los_coupling=args.los_coupling))
    meas_res = simulate(meas_scene["delays"], meas_scene["gains"], noise_std=args.noise_std)
    S_meas = meas_res["S_noisy"]
    h_meas = meas_res["h_noisy"]
    t = meas_res["t"]

    proc = preprocess(S_meas, S_ref, t, tuple(args.target_window), tuple(args.early_window))

    # Calibration to estimate size from energy
    calib = calibrate_energy(
        args.calib_scales,
        cfg=SceneConfig(target_scale=1.0, target_on=True, los_coupling=args.los_coupling),
        target_window_ns=tuple(args.target_window),
        early_window_ns=tuple(args.early_window),
        noise_std=args.noise_std,
    )
    size_est = estimate_size(proc["target_energy"], calib)

    print("==== Processing result ====")
    print(f"Target window {args.target_window[0]:.1f}-{args.target_window[1]:.1f} ns:")
    print(f"  Peak delay: {proc['peak_delay_ns']:.3f} ns")
    print(f"  Peak amp  : {proc['peak_amp']:.3e}")
    print(f"  Energy    : {proc['target_energy']:.3e}")
    print(f"Estimated size scale vs. calibration: {size_est:.2f}x")

    # Plots
    plot_frequency(meas_res["freqs"], S_meas, S_ref, proc["S_diff"])
    plot_time(t, h_meas, h_ref, proc["h_diff"], proc["h_gated"],
              tuple(args.target_window), tuple(args.early_window))
    plot_calibration(calib, proc["target_energy"], size_est)

    backend = matplotlib.get_backend().lower()
    if "agg" in backend or backend in {"pdf", "svg", "ps", "cairo"}:
        print("Non-interactive backend in use; set MPLBACKEND=TkAgg (or another GUI backend) to view plots.")
    else:
        plt.show()


if __name__ == "__main__":
    main()
