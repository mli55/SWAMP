"""
Quick analysis script to break down SFCW responses into components and size variants.

Usage (default shows plots, does not save):
  python3 scripts/analyze_sfcw_components.py
Optionally set MPLBACKEND=TkAgg (or other GUI) if default backend is non-interactive.
"""

import argparse
import sys
import pathlib
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

# Ensure repo root is on sys.path for in-place imports
ROOT = pathlib.Path(__file__).resolve().parents[1]
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
    los_coupling: float = 0.0  # set to 0 by default to reveal target differences
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
        # Approximate sweet potato: ~36 cm x 12 cm, buried ~20 cm
        center = (0.0, 0.20)
        a = 0.18 * cfg.target_scale
        b = 0.06 * cfg.target_scale
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
        "air": air,
        "soil": soil,
        "surface_pts": surface_pts,
        "target_pts": target_pts,
        "los": los,
        "surf": surf,
        "targ": targ,
        "delays": np.concatenate(delays_parts),
        "gains": np.concatenate(gains_parts),
    }


def simulate(delays: np.ndarray, gains: np.ndarray):
    return simulate_sfcw(
        f_start=2e9,
        f_step=40e6,
        n_steps=51,
        delays=delays,
        gains=gains,
        baseband=True,
        probe_tone_hz=None,
    )


def window_energy(t: np.ndarray, h: np.ndarray, t_ns: Tuple[float, float]) -> float:
    mask = (t * 1e9 >= t_ns[0]) & (t * 1e9 <= t_ns[1])
    return float(np.sum(np.abs(h[mask]) ** 2))


def build_cases(scales: List[float], los_coupling: float) -> List[Dict[str, object]]:
    cases = []
    for scale in scales:
        cfg = SceneConfig(target_scale=scale, target_on=scale > 0, los_coupling=los_coupling)
        scene = build_scene(cfg)
        res = simulate(scene["delays"], scene["gains"])
        cases.append(
            {
                "cfg": cfg,
                "scene": scene,
                "freqs": res["freqs"],
                "t": res["t"],
                "S": res["S"],
                "h": res["h"],
                "label": f"target {scale:.1f}x" if scale > 0 else "no target",
            }
        )
    return cases


def plot_frequency(cases: List[Dict[str, object]]):
    plt.figure("SFCW |S(f)|")
    for case in cases:
        freqs = case["freqs"] / 1e9
        mag_db = 20 * np.log10(np.abs(case["S"]) + 1e-12)
        plt.plot(freqs, mag_db, label=case["label"])
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("|S(f)| (dB, arb.)")
    plt.title("Stepped-frequency magnitude")
    plt.grid(True, alpha=0.3)
    plt.legend()


def plot_time(cases: List[Dict[str, object]]):
    plt.figure("IFFT |h(t)|")
    for case in cases:
        t_ns = case["t"] * 1e9
        mag = np.abs(case["h"])
        plt.plot(t_ns, mag, label=case["label"])
    plt.xlabel("Delay (ns)")
    plt.ylabel("|h(t)| (linear, arb.)")
    plt.title("Equivalent impulse response")
    plt.grid(True, alpha=0.3)
    plt.legend()


def plot_energy_bars(cases: List[Dict[str, object]], windows_ns: Dict[str, Tuple[float, float]]):
    """
    Plot windowed energies. If exactly two windows are provided, use dual y-axes so each bar group has its own scale.
    Otherwise fall back to a single-axis grouped bar plot.
    """
    win_items = list(windows_ns.items())
    x = np.arange(len(cases))

    if len(win_items) == 2:
        (name_l, range_l), (name_r, range_r) = win_items
        vals_l = [window_energy(c["t"], c["h"], range_l) for c in cases]
        vals_r = [window_energy(c["t"], c["h"], range_r) for c in cases]

        fig, ax_l = plt.subplots(num="Window energies (dual-axis)")
        ax_r = ax_l.twinx()

        width = 0.35
        ax_l.bar(x - width / 2, vals_l, width=width, label=f"{name_l} {range_l[0]}-{range_l[1]} ns", color="#1f77b4")
        ax_r.bar(x + width / 2, vals_r, width=width, label=f"{name_r} {range_r[0]}-{range_r[1]} ns", color="#ff7f0e", alpha=0.8)

        ax_l.set_xticks(x)
        ax_l.set_xticklabels([c["label"] for c in cases], rotation=20)
        ax_l.set_ylabel(f"{name_l} energy (|h|^2 sum)")
        ax_r.set_ylabel(f"{name_r} energy (|h|^2 sum)")
        ax_l.set_title("Windowed energy vs. target size")
        ax_l.grid(True, axis="y", alpha=0.3)

        # Combined legend
        handles = ax_l.get_legend_handles_labels()[0] + ax_r.get_legend_handles_labels()[0]
        labels = ax_l.get_legend_handles_labels()[1] + ax_r.get_legend_handles_labels()[1]
        ax_l.legend(handles, labels, loc="best")
    else:
        plt.figure("Window energies")
        bar_width = 0.8 / len(win_items)
        for idx, (w_name, w_range) in enumerate(win_items):
            vals = [window_energy(c["t"], c["h"], w_range) for c in cases]
            plt.bar(x + idx * bar_width, vals, width=bar_width, label=f"{w_name} {w_range[0]}-{w_range[1]} ns")
        plt.xticks(x + bar_width * (len(win_items) - 1) / 2, [c["label"] for c in cases], rotation=20)
        plt.ylabel("Energy (|h|^2 sum)")
        plt.title("Windowed energy vs. target size")
        plt.grid(True, axis="y", alpha=0.3)
        plt.legend()


def print_summaries(cases: List[Dict[str, object]], windows_ns: Dict[str, Tuple[float, float]]):
    print("==== Summary ====")
    ref = cases[0]
    rss_ref = np.sqrt(np.sum(np.abs(ref["scene"]["gains"]) ** 2))
    for case in cases:
        rss = np.sqrt(np.sum(np.abs(case["scene"]["gains"]) ** 2))
        rel_db = 20 * np.log10(rss / rss_ref + 1e-15)
        peak_s = np.max(np.abs(case["S"]))
        print(f"{case['label']:12s} rss={rss:.3e} (Δ{rel_db:+.2f} dB vs {cases[0]['label']}), peak|S|={peak_s:.3e}")
        for w_name, w_rng in windows_ns.items():
            e = window_energy(case["t"], case["h"], w_rng)
            print(f"  {w_name:10s} {w_rng[0]:.1f}-{w_rng[1]:.1f} ns energy: {e:.3e}")


def main():
    parser = argparse.ArgumentParser(description="Analyze SFCW components and target size impact.")
    parser.add_argument(
        "--target-scales",
        nargs="+",
        type=float,
        default=[0.0, 0.5, 1.0, 1.5, 2.0],
        help="Target scale factors (0 disables target).",
    )
    parser.add_argument(
        "--los-coupling",
        type=float,
        default=0.0,
        help="LOS coupling amplitude; set >0 to include LOS.",
    )
    parser.add_argument(
        "--target-window",
        type=float,
        nargs=2,
        default=[7.0, 12.0],
        help="Target window in ns for energy integration.",
    )
    parser.add_argument(
        "--early-window",
        type=float,
        nargs=2,
        default=[0.0, 5.0],
        help="Early (LOS/surface) window in ns for energy integration.",
    )
    args = parser.parse_args()

    cases = build_cases(args.target_scales, los_coupling=args.los_coupling)

    windows = {"early": tuple(args.early_window), "target": tuple(args.target_window)}
    print_summaries(cases, windows)

    plot_frequency(cases)
    plot_time(cases)
    plot_energy_bars(cases, windows)
    plt.show()


if __name__ == "__main__":
    main()
