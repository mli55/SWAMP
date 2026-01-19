# scripts/run_ofdm_step2.py
import sys
import pathlib

# Ensure imports work whether the package lives in the repo root (rfspsim/)
# or under a src/ layout. Prefer the repo root copy because src/ is empty.
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "src"))

import numpy as np
import matplotlib.pyplot as plt

from rfspsim.media.medium import Medium
from rfspsim.geometry.sampling import sample_ellipse_area, sample_surface_line
from rfspsim.propagation.channel_builders import build_surface_scatter_taps, build_target_reflection_taps
from rfspsim.simulators.wifi_ofdm_sim import simulate_wifi_ofdm
from rfspsim.propagation.channel_builders import (
    build_los_taps,
    build_surface_scatter_taps,
    build_target_reflection_taps,
)


def main():
    air = Medium("air", epsilon_r=1.0, mu_r=1.0)
    soil = Medium("soil", epsilon_r=4.0, mu_r=1.0)
    interface_z = 0.0

    tx = np.array([-0.5, -0.05])
    rx = np.array([+0.5, -0.05])

    surface_pts, cell_len = sample_surface_line(-1.5, 1.5, n_points=600, z=interface_z, method="uniform")

    center = (0.0, 0.10)
    a, b = 0.15, 0.07
    target_pts = sample_ellipse_area(center=center, a=a, b=b, n_points=800, seed=0)
    cell_area = np.pi * a * b / len(target_pts)

    surf = build_surface_scatter_taps(
        tx, rx, surface_pts,
        air=air, soil=soil, interface_z=interface_z,
        per_point_length=cell_len,
        surface_reflectivity=0.15,
        pol="avg",
        include_fresnel=True,
    )
    targ = build_target_reflection_taps(
        tx, rx, target_pts,
        air=air, soil=soil, interface_z=interface_z,
        per_point_area=cell_area,
        target_reflectivity=1.0 + 0j,
        pol="avg",
        include_fresnel=True,
    )
    los = build_los_taps(
    tx, rx, air=air,
    los_coupling=5.0 + 0j,
    model="inv_d2"
)

    delays = np.concatenate([los["delays"], surf["delays"], targ["delays"]])
    gains  = np.concatenate([los["gains"],  surf["gains"],  targ["gains"]])

    print(f"LOS delay = {los['delays'][0]*1e9:.3f} ns, |LOS gain| = {abs(los['gains'][0]):.3e}")

    # WiFi-like OFDM 参数（简化）
    bw = 20e6
    n_fft = 64
    n_sym = 10
    cp_len = 16

    out = simulate_wifi_ofdm(
        bw=bw, n_fft=n_fft, n_sym=n_sym, cp_len=cp_len,
        delays=delays, gains=gains, seed=1
    )

    tx_time = out["tx_time"]
    rx_time = out["rx_time"]
    H = out["H"]
    f_offsets = out["f_offsets"]

    # Plot
    plt.figure()
    plt.plot(np.real(tx_time), label="Tx (real)")
    plt.plot(np.real(rx_time), label="Rx (real)", alpha=0.7)
    plt.legend()
    plt.title("OFDM time-domain baseband (real)")

    plt.figure()
    plt.plot(np.abs(tx_time), label="|Tx|")
    plt.plot(np.abs(rx_time), label="|Rx|", alpha=0.7)
    plt.legend()
    plt.title("OFDM time-domain baseband magnitude")

    plt.figure()
    plt.plot(f_offsets/1e6, 20*np.log10(np.abs(H)+1e-12))
    plt.xlabel("Subcarrier offset (MHz)")
    plt.ylabel("|H(f)| (dB, arb.)")
    plt.title("Baseband channel response across OFDM subcarriers")

    plt.show()


if __name__ == "__main__":
    main()
