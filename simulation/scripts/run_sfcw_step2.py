# scripts/run_sfcw_step2.py
import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt

# Ensure the repo root is on sys.path for in-place imports
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rfspsim.media.medium import Medium
from rfspsim.geometry.sampling import sample_ellipse_area, sample_surface_line
from rfspsim.propagation.channel_builders import build_surface_scatter_taps, build_target_reflection_taps
from rfspsim.simulators.sfcw_sim import simulate_sfcw
from rfspsim.propagation.channel_builders import (
    build_los_taps,
    build_surface_scatter_taps,
    build_target_reflection_taps,
)


def main():
    # ---------- 介质参数（无损） ----------
    # 你提到的 μ：如果你其实想表达“介电常数/慢速因子”，建议用 epsilon_r 来控制
    air = Medium("air", epsilon_r=1.0, mu_r=1.0)
    soil = Medium("soil", epsilon_r=4.0, mu_r=1.0)   # 例子：εr=4 => v ≈ 0.5 c0
    interface_z = 0.0

    # ---------- 场景 ----------
    tx = np.array([-0.5, -0.05])   # 空气中（z<0）
    rx = np.array([+0.5, -0.05])

    # 地表散射点（clutter）
    surface_pts, cell_len = sample_surface_line(x_min=-1.5, x_max=1.5, n_points=600, z=interface_z, method="uniform")

    # 红薯（地下目标）散射点
    # 以更接近实物的红薯尺寸/埋深：长轴≈36 cm，短轴≈12 cm，埋深≈20 cm
    center = (0.0, 0.20)   # z>0 在土里，20 cm 深
    a, b = 0.18, 0.06      # 半轴长度 (m): 0.18/0.06 => 全长 ~36 cm，厚度 ~12 cm
    target_pts = sample_ellipse_area(center=center, a=a, b=b, n_points=800, seed=0)
    cell_area = np.pi * a * b / len(target_pts)

    # ---------- 构建 taps ----------
    surf = build_surface_scatter_taps(
        tx, rx, surface_pts,
        air=air, soil=soil, interface_z=interface_z,
        per_point_length=cell_len,
        surface_reflectivity=0.15,  # 地表散射强度（经验系数）
        pol="avg",
        include_fresnel=True,
    )

    targ = build_target_reflection_taps(
        tx, rx, target_pts,
        air=air, soil=soil, interface_z=interface_z,
        per_point_area=cell_area,
        target_reflectivity=1.0 + 0j,  # 红薯反射系数（先当常数）
        pol="avg",
        include_fresnel=True,
    )


    los = build_los_taps(
        tx, rx, air=air,
        los_coupling=5.0 + 0j,   # 这个值你可以调：越大 LOS 越强
        model="inv_d2"
    )

    delays = np.concatenate([los["delays"], surf["delays"], targ["delays"]])
    gains  = np.concatenate([los["gains"],  surf["gains"],  targ["gains"]])

    print(f"LOS delay = {los['delays'][0]*1e9:.3f} ns, |LOS gain| = {abs(los['gains'][0]):.3e}")
    print(f"Air v={air.v:.2e} m/s, Soil v={soil.v:.2e} m/s")
    print(f"Surface taps: {len(surf['delays'])}, Target taps: {len(targ['delays'])}")
    print(f"Delay range: {delays.min()*1e9:.3f} ns ~ {delays.max()*1e9:.3f} ns")

    # ---------- SFCW 扫频 ----------
    res = simulate_sfcw(
        f_start=2e9,
        f_step=40e6,
        n_steps=51,
        delays=delays,
        gains=gains,
        baseband=True
    )

    freqs = res["freqs"]
    S = res["S"]
    t = res["t"]
    h = res["h"]

    # ---------- Plot ----------
    plt.figure()
    plt.plot(freqs/1e9, 20*np.log10(np.abs(S)+1e-12))
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("|S(f)| (dB, arb.)")
    plt.title("SFCW magnitude with surface + refracted target paths")

    plt.figure()
    plt.plot(t*1e9, np.abs(h))
    plt.xlabel("Delay (ns)")
    plt.ylabel("|h(t)| (arb.)")
    plt.title("IFFT -> equivalent impulse response")

    plt.figure()
    plt.scatter(surface_pts[:,0], surface_pts[:,1], s=5, label="Surface scatterers")
    plt.scatter(target_pts[:,0], target_pts[:,1], s=5, label="Target scatterers")
    plt.scatter(tx[0], tx[1], marker="^", label="Tx")
    plt.scatter(rx[0], rx[1], marker="s", label="Rx")
    plt.gca().invert_yaxis()
    plt.xlabel("x (m)")
    plt.ylabel("z (m)")
    plt.legend()
    plt.title("Geometry (z=0 is soil interface)")

    plt.show()


if __name__ == "__main__":
    main()
