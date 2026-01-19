"""
Line-scan SFCW simulation with per-position reference subtraction to form a B-scan, plus optional 2D backprojection
imaging over a soil box.

Geometry: Tx/Rx slide along +x with fixed baseline above the soil interface (z < 0). At each x, simulate a measured
sweep (with target and clutter) and a reference sweep (clutter only), subtract in frequency, window + zero-pad, then
IFFT to get h(t). Optionally form a simple homogeneous backprojection image and report its peak location.

Usage examples:
  python3 scripts/bscan_backprojection.py
  python3 scripts/bscan_backprojection.py --do-backprojection

Note: Derived from scripts/scan_sfcw_bscan.py with changes:
- always run measured + reference per position and subtract in frequency
- removed ground-truth comparison plot; added backprojection peak reporting
- when running headless (Agg/pdf/svg), skip plt.show() and save figures to outputs/
"""

import os
import argparse
import sys
import pathlib
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
(ROOT / ".mplconfig").mkdir(exist_ok=True)

import matplotlib
import matplotlib.pyplot as plt

sys.path.insert(0, str(ROOT))

from rfspsim.media.medium import Medium
from rfspsim.geometry.sampling import sample_surface_line, sample_ellipse_area
from rfspsim.propagation.channel_builders import (
    build_los_taps,
    build_surface_scatter_taps,
    build_target_reflection_taps,
)
from rfspsim.propagation.planar import refracted_two_segment
from rfspsim.simulators.sfcw_sim import simulate_sfcw


@dataclass
class ScanConfig:
    x_start: float = -0.4
    x_stop: float = 0.4
    x_step: float = 0.02
    baseline: float = 0.5
    tx_z: float = -0.05
    rx_z: float = -0.05
    soil_epsilon_r: float = 4.0
    surface_reflectivity: float = 0.15
    los_coupling: float = 1.0
    target_center: Tuple[float, float] = (0.0, 0.20)
    target_length: float = 0.20
    target_thickness: float = 0.06
    target_scale: float = 1.0
    target_on: bool = True
    n_surface: int = 600
    n_target: int = 800
    f_start: float = 2e9
    f_step: float = 40e6
    n_steps: int = 51
    noise_std: float = 0.0
    seed: int = 0
    window: str = "hann"
    zp_factor: int = 1
    subtract_ref: bool = True
    surface_margin: float = 0.6
    min_delay_ns: float = 0.0
    max_delay_ns: float = 15.0
    dyn_range_db: float = 40.0

    # Backprojection
    do_backprojection: bool = False
    bp_dx: float = 0.01
    bp_dz: float = 0.01
    bp_z_max: float = 0.50
    bp_coherent: bool = False
    bp_epsilon_r: Optional[float] = None


@dataclass
class StaticScene:
    air: Medium
    soil: Medium
    interface_z: float
    surface_pts: np.ndarray
    surface_cell_len: float
    target_pts: Optional[np.ndarray]
    target_cell_area: Optional[float]


def build_static_scene(cfg: ScanConfig) -> StaticScene:
    air = Medium("air", epsilon_r=1.0, mu_r=1.0)
    soil = Medium("soil", epsilon_r=cfg.soil_epsilon_r, mu_r=1.0)
    interface_z = 0.0

    surface_pts, cell_len = sample_surface_line(
        x_min=cfg.x_start - cfg.surface_margin,
        x_max=cfg.x_stop + cfg.surface_margin,
        n_points=cfg.n_surface,
        z=interface_z,
        method="uniform",
    )

    a = 0.5 * cfg.target_length * cfg.target_scale
    b = 0.5 * cfg.target_thickness * cfg.target_scale
    target_pts = np.array([[cfg.target_center[0], cfg.target_center[1]]], dtype=float)
    cell_area = np.pi * a * b
    target_pts = sample_ellipse_area(
        center=cfg.target_center,
        a=a,
        b=b,
        n_points=cfg.n_target,
        seed=0,
    )
    cell_area = np.pi * a * b / len(target_pts)

    return StaticScene(
        air=air,
        soil=soil,
        interface_z=interface_z,
        surface_pts=surface_pts,
        surface_cell_len=cell_len,
        target_pts=target_pts,
        target_cell_area=cell_area,
    )


def ellipse_boundary_points(cfg: ScanConfig, n: int = 240) -> np.ndarray:
    """Deterministic ellipse boundary points for delay envelope overlays."""
    cx, cz = cfg.target_center
    a = 0.5 * cfg.target_length * cfg.target_scale
    b = 0.5 * cfg.target_thickness * cfg.target_scale
    th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.stack([cx + a * np.cos(th), cz + b * np.sin(th)], axis=1)


def two_way_tau_refracted(tx: np.ndarray, rx: np.ndarray, pts: np.ndarray, static: StaticScene) -> np.ndarray:
    """Two-way travel time with Snell refraction for arbitrary points."""
    va, vs = static.air.v, static.soil.v
    pts = np.asarray(pts, dtype=float)
    out = np.empty(len(pts), dtype=float)
    for i, p in enumerate(pts):
        _, L_air_in, L_soil_in, _, _ = refracted_two_segment(tx, p, v1=va, v2=vs, interface_z=static.interface_z)
        _, L_soil_out, L_air_out, _, _ = refracted_two_segment(p, rx, v1=vs, v2=va, interface_z=static.interface_z)
        out[i] = L_air_in / va + L_soil_in / vs + L_soil_out / vs + L_air_out / va
    return out


def build_taps_for_position(
    x_center: float,
    static: StaticScene,
    cfg: ScanConfig,
    with_target: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tx = np.array([x_center - cfg.baseline / 2.0, cfg.tx_z])
    rx = np.array([x_center + cfg.baseline / 2.0, cfg.rx_z])

    los = build_los_taps(
        tx, rx, air=static.air, los_coupling=cfg.los_coupling + 0j, model="inv_d2"
    )
    surf = build_surface_scatter_taps(
        tx,
        rx,
        static.surface_pts,
        air=static.air,
        soil=static.soil,
        interface_z=static.interface_z,
        per_point_length=static.surface_cell_len,
        surface_reflectivity=cfg.surface_reflectivity,
        pol="avg",
        include_fresnel=True,
    )

    delays_parts = [los["delays"], surf["delays"]]
    gains_parts = [los["gains"], surf["gains"]]

    if with_target and cfg.target_on and static.target_pts is not None:
        targ = build_target_reflection_taps(
            tx,
            rx,
            static.target_pts,
            air=static.air,
            soil=static.soil,
            interface_z=static.interface_z,
            per_point_area=float(static.target_cell_area),
            target_reflectivity=1.0 + 0j,
            pol="avg",
            include_fresnel=True,
        )
        delays_parts.append(targ["delays"])
        gains_parts.append(targ["gains"])

    return tx, rx, np.concatenate(delays_parts), np.concatenate(gains_parts)


def simulate_with_noise(
    delays: np.ndarray,
    gains: np.ndarray,
    cfg: ScanConfig,
    rng: np.random.Generator,
) -> dict:
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        res = simulate_sfcw(
            f_start=cfg.f_start,
            f_step=cfg.f_step,
            n_steps=cfg.n_steps,
            delays=delays,
            gains=gains,
            baseband=True,
            probe_tone_hz=None,
        )
    S = res["S"]
    if cfg.noise_std > 0.0:
        noise = (rng.normal(scale=cfg.noise_std, size=S.shape) + 1j * rng.normal(scale=cfg.noise_std, size=S.shape))
        S = S + noise

    res["S_noisy"] = S
    return res


def _make_window(n: int, kind: str) -> np.ndarray:
    kind = (kind or "none").lower()
    if kind in {"none", "rect", "rectangular"}:
        return np.ones(n, dtype=float)
    if kind in {"hann", "hanning"}:
        return np.hanning(n).astype(float)
    if kind == "hamming":
        return np.hamming(n).astype(float)
    raise ValueError(f"Unknown window type: {kind}. Use none|hann|hamming")


def sfcw_to_delay(
    S: np.ndarray,
    f_step: float,
    window: str = "hann",
    zp_factor: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert complex S(f) (uniformly spaced) to delay response h(t) via IFFT."""
    S = np.asarray(S)
    n = int(S.shape[-1])
    w = _make_window(n, window)
    S_w = S * w

    zp_factor = max(int(zp_factor), 1)
    n_fft = n * zp_factor
    h = np.fft.ifft(S_w, n=n_fft)

    dt = 1.0 / (n_fft * f_step)
    t = np.arange(n_fft) * dt
    return h, t


def run_line_scan(static: StaticScene, cfg: ScanConfig) -> dict:
    xs = np.arange(cfg.x_start, cfg.x_stop + 1e-12, cfg.x_step)
    rng = np.random.default_rng(cfg.seed)
    H = []
    S_diff = []
    tx_track = []
    rx_track = []
    t_axis = None
    freq_axis = None
    tau_center_gt = []
    tau_band_min = []
    tau_band_max = []
    tau_tap_min = []
    tau_tap_med = []
    tau_tap_max = []
    boundary_pts = ellipse_boundary_points(cfg) if cfg.target_on else None

    for x in xs:
        tx, rx, delays_meas, gains_meas = build_taps_for_position(
            x_center=x, static=static, cfg=cfg, with_target=True
        )
        _, _, delays_ref, gains_ref = build_taps_for_position(
            x_center=x, static=static, cfg=cfg, with_target=False
        )

        meas = simulate_with_noise(delays_meas, gains_meas, cfg, rng=rng)
        ref = simulate_with_noise(delays_ref, gains_ref, cfg, rng=rng)

        S_sel = meas["S_noisy"] - ref["S_noisy"] if cfg.subtract_ref else meas["S_noisy"]

        h_sel, t_new = sfcw_to_delay(
            S_sel,
            f_step=cfg.f_step,
            window=cfg.window,
            zp_factor=cfg.zp_factor,
        )

        tx_track.append(tx)
        rx_track.append(rx)
        H.append(h_sel)
        S_diff.append(S_sel)

        if t_axis is None:
            t_axis = t_new
            freq_axis = meas["freqs"]

        if cfg.target_on and static.target_pts is not None:
            tau_c = two_way_tau_refracted(tx, rx, np.array([cfg.target_center]), static)[0]
            tau_center_gt.append(tau_c)

            if boundary_pts is not None:
                tau_b = two_way_tau_refracted(tx, rx, boundary_pts, static)
                tau_band_min.append(float(np.min(tau_b)))
                tau_band_max.append(float(np.max(tau_b)))

            tau_taps = two_way_tau_refracted(tx, rx, static.target_pts, static)
            tau_tap_min.append(float(np.min(tau_taps)))
            tau_tap_med.append(float(np.median(tau_taps)))
            tau_tap_max.append(float(np.max(tau_taps)))

    return {
        "xs": xs,
        "t": t_axis,
        "freqs": freq_axis,
        "H": np.stack(H),
        "S": np.stack(S_diff),
        "tx_track": np.vstack(tx_track),
        "rx_track": np.vstack(rx_track),
        "tau_center_gt": np.array(tau_center_gt) if tau_center_gt else None,
        "tau_band_gt": (
            np.array(tau_band_min) if tau_band_min else None,
            np.array(tau_band_max) if tau_band_max else None,
        ),
        "tau_taps_stats": {
            "min": np.array(tau_tap_min) if tau_tap_min else None,
            "med": np.array(tau_tap_med) if tau_tap_med else None,
            "max": np.array(tau_tap_max) if tau_tap_max else None,
        },
    }


def backproject_image(
    xs: np.ndarray,
    t: np.ndarray,
    H: np.ndarray,
    tx_track: np.ndarray,
    rx_track: np.ndarray,
    static: StaticScene,
    cfg: ScanConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simple 2D delay-and-sum backprojection over (x,z)."""
    xg = np.arange(cfg.x_start, cfg.x_stop + 1e-12, cfg.bp_dx)
    zg = np.arange(0.0, cfg.bp_z_max + 1e-12, cfg.bp_dz)
    X, Z = np.meshgrid(xg, zg, indexing="xy")

    dt = float(t[1] - t[0]) if len(t) > 1 else 0.0
    n_t = H.shape[1]

    img = np.zeros_like(X, dtype=np.complex128 if cfg.bp_coherent else np.float64)
    P = np.stack([X.ravel(), Z.ravel()], axis=1)

    for m in range(len(xs)):
        tx = tx_track[m]
        rx = rx_track[m]

        tau_flat = two_way_tau_refracted(tx, rx, P, static)
        tau = tau_flat.reshape(X.shape)

        u = tau / dt
        i0 = np.floor(u).astype(np.int32)
        w = u - i0
        valid = (i0 >= 0) & (i0 + 1 < n_t)

        h = H[m]
        h0 = np.zeros_like(X, dtype=np.complex128)
        h1 = np.zeros_like(X, dtype=np.complex128)
        h0[valid] = h[i0[valid]]
        h1[valid] = h[i0[valid] + 1]
        samp = (1.0 - w) * h0 + w * h1

        if cfg.bp_coherent:
            img = img + samp
        else:
            img = img + (np.abs(samp) ** 2)

    img_out = np.abs(img) if cfg.bp_coherent else img
    return xg, zg, img_out


def plot_backprojection(
    xg: np.ndarray,
    zg: np.ndarray,
    img: np.ndarray,
    dyn_range_db: float = 40.0,
    cfg: Optional[ScanConfig] = None,
):
    img_db = 20 * np.log10(np.abs(img) + 1e-12)
    vmax = np.percentile(img_db, 99.5)
    vmin = vmax - float(dyn_range_db)

    plt.figure("Backprojection image (dB)")
    extent = [xg.min(), xg.max(), zg.max(), zg.min()]
    plt.imshow(img_db, extent=extent, aspect="equal", origin="upper", cmap="magma", vmin=vmin, vmax=vmax)
    plt.xlabel("x (m)")
    plt.ylabel("z (m, depth)")
    plt.title("Backprojection (coherent)" if img.dtype != np.float64 else "Backprojection (energy)")
    plt.gca().set_aspect("equal", adjustable="box")
    if cfg is not None and cfg.target_on:
        # Overlay target ground-truth outline (ellipse)
        from matplotlib.patches import Ellipse

        cx, cz = cfg.target_center
        a = 0.5 * cfg.target_length * cfg.target_scale
        b = 0.5 * cfg.target_thickness * cfg.target_scale
        ell = Ellipse(
            (cx, cz),
            width=2.0 * a,
            height=2.0 * b,
            angle=0.0,
            fill=False,
            linewidth=2.0,
            label="Target (GT)",
        )
        plt.gca().add_patch(ell)
        plt.legend(loc="upper right")
    plt.colorbar(label="20log10|I| (a.u.)")
    return plt.gcf()


def plot_bscan(
    xs: np.ndarray,
    t: np.ndarray,
    H: np.ndarray,
    min_delay_ns: float,
    max_delay_ns: float,
    dyn_range_db: float = 40.0,
    tau_center: Optional[np.ndarray] = None,
    tau_band: Optional[Tuple[Optional[np.ndarray], Optional[np.ndarray]]] = None,
    tau_taps_stats: Optional[dict] = None,
):
    t_ns = t * 1e9
    mask = np.ones_like(t_ns, dtype=bool)
    if min_delay_ns > 0:
        mask &= t_ns >= min_delay_ns
    if max_delay_ns > 0:
        mask &= t_ns <= max_delay_ns

    H_mag_db = 20 * np.log10(np.abs(H[:, mask]) + 1e-12)
    t_ns = t_ns[mask]

    vmax = np.percentile(H_mag_db, 99.0)
    vmin = vmax - float(dyn_range_db)

    plt.figure("B-scan |h| (dB)")
    extent = [xs.min(), xs.max(), t_ns[-1], t_ns[0]]
    plt.imshow(
        H_mag_db.T,
        extent=extent,
        aspect="auto",
        origin="upper",
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
    )
    plt.xlabel("Scan x (m)")
    plt.ylabel("Path delay (ns)")
    plt.title("Simulated SFCW line scan (meas - ref)")
    plt.colorbar(label="20log10|h| (a.u.)")
    ax = plt.gca()
    legend_added = False
    if tau_center is not None and len(tau_center) == len(xs):
        ax.plot(xs, tau_center * 1e9, "w--", lw=1.3, label="GT center τ (Snell)")
        legend_added = True
    if tau_band is not None and tau_band[0] is not None and tau_band[1] is not None and len(tau_band[0]) == len(xs):
        ax.fill_between(xs, tau_band[0] * 1e9, tau_band[1] * 1e9, color="cyan", alpha=0.15, label="GT boundary τ band")
        legend_added = True
    if tau_taps_stats:
        med = tau_taps_stats.get("med")
        if med is not None and len(med) == len(xs):
            ax.plot(xs, med * 1e9, color="lime", lw=1.0, ls=":", label="Tap τ median")
            legend_added = True
        tmin = tau_taps_stats.get("min")
        tmax = tau_taps_stats.get("max")
        if tmin is not None and tmax is not None and len(tmin) == len(xs):
            ax.fill_between(xs, tmin * 1e9, tmax * 1e9, color="lime", alpha=0.08, label="Tap τ band")
            legend_added = True
    if legend_added:
        ax.legend(loc="upper right")
    return plt.gcf()


def plot_geometry(static: StaticScene, xs: np.ndarray, cfg: ScanConfig):
    plt.figure("Geometry (top view)")
    plt.scatter(static.surface_pts[:, 0], static.surface_pts[:, 1], s=3, alpha=0.4, label="Surface scatters")
    if cfg.target_on and static.target_pts is not None:
        plt.scatter(static.target_pts[:, 0], static.target_pts[:, 1], s=5, alpha=0.8, label="Target points")
    plt.scatter(xs - cfg.baseline / 2.0, np.full_like(xs, cfg.tx_z), s=8, label="Tx track")
    plt.scatter(xs + cfg.baseline / 2.0, np.full_like(xs, cfg.rx_z), s=8, label="Rx track")
    plt.axhline(static.interface_z, color="k", linestyle="--", alpha=0.4, label="Interface z=0")
    plt.gca().invert_yaxis()
    plt.xlabel("x (m)")
    plt.ylabel("z (m)")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.legend()
    plt.title("Tx/Rx sliding along x over soil box")
    return plt.gcf()


def main():
    defaults = ScanConfig()

    parser = argparse.ArgumentParser(
        description=(
            "Simulate a line-scan SFCW measurement with reference subtraction and render a B-scan, "
            "optionally backproject."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    SUP = argparse.SUPPRESS

    # Scan / geometry
    parser.add_argument("--x-start", type=float, default=SUP, help="Scan start x (m).")
    parser.add_argument("--x-stop", type=float, default=SUP, help="Scan stop x (m).")
    parser.add_argument("--x-step", type=float, default=SUP, help="Scan step (m).")
    parser.add_argument("--baseline", type=float, default=SUP, help="Tx-Rx separation (m), Rx is +x side.")
    parser.add_argument("--tx-z", type=float, default=SUP, help="Tx height (m, z<0 is above soil).")
    parser.add_argument("--rx-z", type=float, default=SUP, help="Rx height (m, z<0 is above soil).")
    parser.add_argument("--soil-epsilon", dest="soil_epsilon_r", type=float, default=SUP, help="Relative permittivity of soil.")
    parser.add_argument("--surface-reflectivity", type=float, default=SUP, help="Surface clutter gain factor.")
    parser.add_argument("--los-coupling", type=float, default=SUP, help="LOS coupling amplitude.")
    parser.add_argument("--surface-margin", type=float, default=SUP, help="Extra x-span for surface scatterers (m).")

    # Target
    parser.add_argument("--target-center-x", dest="target_center_x", type=float, default=SUP, help="Target center x (m).")
    parser.add_argument(
        "--target-depth", dest="target_depth", type=float, default=SUP, help="Target depth (m, positive is below surface)."
    )
    parser.add_argument("--target-length", type=float, default=SUP, help="Target full length (m).")
    parser.add_argument("--target-thickness", type=float, default=SUP, help="Target full thickness (m).")
    parser.add_argument("--target-scale", type=float, default=SUP, help="Scale factor on both axes.")
    parser.add_argument("--n-target", type=int, default=SUP, help="Number of target scatter points.")

    tgt_onoff = parser.add_mutually_exclusive_group()
    tgt_onoff.add_argument("--target", dest="target_on", action="store_true", default=SUP, help="Enable target.")
    tgt_onoff.add_argument(
        "--no-target", dest="target_on", action="store_false", default=SUP, help="Disable target to view clutter-only B-scan."
    )

    # Scatterers
    parser.add_argument("--n-surface", type=int, default=SUP, help="Number of surface scatter points.")

    # SFCW
    parser.add_argument("--f-start", type=float, default=SUP, help="SFCW start freq (Hz).")
    parser.add_argument("--f-step", type=float, default=SUP, help="SFCW step size (Hz).")
    parser.add_argument("--n-steps", type=int, default=SUP, help="Number of frequency points.")

    # Processing
    parser.add_argument("--noise-std", type=float, default=SUP, help="AWGN std dev on complex S(f).")
    parser.add_argument("--min-delay-ns", type=float, default=SUP, help="Minimum delay gate (ns) shown in plots.")
    parser.add_argument("--max-delay-ns", type=float, default=SUP, help="Maximum delay window (ns) shown in plots.")
    parser.add_argument("--seed", type=int, default=SUP, help="Random seed for noise.")
    parser.add_argument("--window", type=str, choices=["none", "hann", "hamming"], default=SUP, help="Frequency window.")
    parser.add_argument("--zp-factor", type=int, default=SUP, help="Zero-padding factor for IFFT length.")
    parser.add_argument("--dyn-range-db", type=float, default=SUP, help="Displayed dynamic range in dB for plots.")

    ref_onoff = parser.add_mutually_exclusive_group()
    ref_onoff.add_argument(
        "--subtract-ref", dest="subtract_ref", action="store_true", default=SUP, help="Enable per-position reference subtraction."
    )
    ref_onoff.add_argument(
        "--no-subtract-ref", dest="subtract_ref", action="store_false", default=SUP, help="Skip per-position reference subtraction."
    )

    # Backprojection
    parser.add_argument("--do-backprojection", action="store_true", default=SUP, help="Run simple 2D backprojection imaging.")
    parser.add_argument("--bp-dx", type=float, default=SUP, help="Backprojection grid dx (m).")
    parser.add_argument("--bp-dz", type=float, default=SUP, help="Backprojection grid dz (m).")
    parser.add_argument("--bp-z-max", type=float, default=SUP, help="Backprojection max depth z (m).")
    parser.add_argument("--bp-coherent", action="store_true", default=SUP, help="Use coherent (complex) backprojection sum.")
    parser.add_argument("--bp-epsilon", dest="bp_epsilon_r", type=float, default=SUP, help="Override epsilon_r for imaging (optional).")

    args = parser.parse_args()

    overrides = vars(args)

    # Compose tuple field target_center from optional CLI components.
    if "target_center_x" in overrides or "target_depth" in overrides:
        cx0, cz0 = defaults.target_center
        cx = overrides.pop("target_center_x", cx0)
        cz = overrides.pop("target_depth", cz0)
        overrides["target_center"] = (cx, cz)

    cfg_kwargs = defaults.__dict__.copy()
    cfg_kwargs.update(overrides)
    cfg = ScanConfig(**cfg_kwargs)

    static = build_static_scene(cfg)
    res = run_line_scan(static, cfg)

    print("==== Line-scan summary ====")
    print(f"Positions: {len(res['xs'])} from {res['xs'][0]:.3f} to {res['xs'][-1]:.3f} m (step {cfg.x_step:.3f} m)")
    bw = cfg.f_step * (cfg.n_steps - 1)
    print(f"Delay axis: 0-{res['t'][-1]*1e9:.2f} ns (dt={(res['t'][1]-res['t'][0])*1e9:.3f} ns), Δf={cfg.f_step/1e6:.1f} MHz, BW≈{bw/1e6:.1f} MHz")
    print(f"Window: {cfg.window}, zp_factor: {cfg.zp_factor}")
    print(f"Target enabled: {cfg.target_on}, ref subtraction: {cfg.subtract_ref}")

    figs = []
    figs.append(
        plot_bscan(
            res["xs"],
            res["t"],
            res["H"],
            min_delay_ns=cfg.min_delay_ns,
            max_delay_ns=cfg.max_delay_ns,
            dyn_range_db=cfg.dyn_range_db,
            tau_center=res.get("tau_center_gt"),
            tau_band=res.get("tau_band_gt"),
            tau_taps_stats=res.get("tau_taps_stats"),
        )
    )
    figs.append(plot_geometry(static, res["xs"], cfg))

    if cfg.do_backprojection:
        xg, zg, img = backproject_image(res["xs"], res["t"], res["H"], res["tx_track"], res["rx_track"], static=static, cfg=cfg)
        fig_bp = plot_backprojection(xg, zg, img, dyn_range_db=cfg.dyn_range_db, cfg=cfg)
        figs.append(fig_bp)
        peak_idx = np.unravel_index(np.argmax(img), img.shape)
        x_peak = xg[peak_idx[1]]
        z_peak = zg[peak_idx[0]]
        peak_val = img[peak_idx]
        print(f"Backprojection peak: x*={x_peak:.3f} m, z*={z_peak:.3f} m, peak={peak_val:.3e}")

    backend = matplotlib.get_backend().lower()
    headless = ("agg" in backend) or (backend in {"pdf", "svg", "ps", "cairo"})
    if headless:
        out_dir = ROOT / "outputs"
        out_dir.mkdir(exist_ok=True)
        names = ["bscan.png", "geometry.png", "backprojection.png"]
        for fig, name in zip(figs, names):
            if fig is None:
                continue
            out_path = out_dir / name
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
            print(f"Saved figure: {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
