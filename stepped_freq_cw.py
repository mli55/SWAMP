import numpy as np
import matplotlib.pyplot as plt


def make_ellipse_area_scatterers(center, a, b, n_r=20, n_theta=36, seed=None):
    """
    在椭圆内部随机生成 2D 散射点：
      (x - x0)^2 / a^2 + (z - z0)^2 / b^2 <= 1
    """
    center = np.array(center, dtype=float)
    n_points = n_r * n_theta

    rng = np.random.default_rng(seed)
    rs = np.sqrt(rng.random(n_points))
    thetas = rng.random(n_points) * 2.0 * np.pi

    xs = center[0] + a * rs * np.cos(thetas)
    zs = center[1] + b * rs * np.sin(thetas)

    scatterers = np.column_stack((xs, zs))
    return scatterers


def build_channel(center=(0.0, 0.4), a=0.15, b=0.07,
                  tx=(-0.5, 0.0), rx=(0.5, 0.0),
                  n_r=20, n_theta=36, c=3e8, seed=None):
    """
    和 WiFi 版本一样：几何 + 多散射点通道参数 (A_i, tau_i)。
    """
    tx_arr = np.array(tx, dtype=float)
    rx_arr = np.array(rx, dtype=float)

    scatterers = make_ellipse_area_scatterers(center, a, b, n_r, n_theta, seed=seed)
    n_scatter = scatterers.shape[0]

    d_tx = np.linalg.norm(scatterers - tx_arr, axis=1)
    d_rx = np.linalg.norm(scatterers - rx_arr, axis=1)
    d = d_tx + d_rx
    tau = d / c

    cell_area = np.pi * a * b / n_scatter
    A = cell_area / (d**2 + 1e-6)

    return scatterers, A, tau


def simulate_sfcw_response(
    f_start=2e9,      # 起始频率（和文中一样 2 GHz）
    f_step=40e6,      # 步进（例如 2.000 → 2.040 → 2.080 ...）
    n_steps=51,       # 频点数量
    center=(0.0, 0.4),
    a=0.15, b=0.07,
    tx=(-0.5, 0.0), rx=(0.5, 0.0),
    n_r=20, n_theta=36,
    c=3e8, seed=None
):
    """
    仿真 stepped-frequency CW：

    对每个频点 fk:
        S(fk) = sum_i A_i * exp(-j 2π fk * tau_i)

    然后对 S(fk) 做 IFFT，得到一个“等效脉冲响应” h(t)，对应深度方向的反射分布。
    """
    scatterers, A, tau = build_channel(center, a, b, tx, rx, n_r, n_theta, c, seed)

    # 频率扫：f_start, f_start+f_step, ...
    freqs = f_start + np.arange(n_steps) * f_step

    # 为了做 IFFT，实际只关心 baseband 频率偏移 f_bb，从 0 开始
    f_bb = freqs - freqs[0]

    S = np.zeros(n_steps, dtype=complex)
    for k, fk in enumerate(f_bb):
        S[k] = np.sum(A * np.exp(-1j * 2.0 * np.pi * fk * tau))

    # IFFT -> 时域脉冲响应
    delta_f = f_step
    T_win = 1.0 / delta_f      # 最大无混叠时延窗口
    dt = T_win / n_steps       # 时域采样间隔
    t = np.arange(n_steps) * dt
    h = np.fft.ifft(S)

    return {
        "scatterers": scatterers,
        "A": A,
        "tau": tau,
        "freqs": freqs,    # 实际扫的载频
        "S": S,            # 每个频点的复响应
        "t": t,            # 时域（延时）
        "h": h,            # 时域脉冲响应
        "f_start": f_start,
        "f_step": f_step,
        "n_steps": n_steps,
    }


if __name__ == "__main__":
    # ------------ stepped-frequency CW 版本示例 ------------
    res = simulate_sfcw_response(
        f_start=2e9,
        f_step=40e6,
        n_steps=51,
        seed=0
    )

    freqs = res["freqs"]
    S = res["S"]
    t = res["t"]
    h = res["h"]
    scatterers = res["scatterers"]
    tau = res["tau"]

    print(f"扫频范围: {freqs[0]/1e9:.3f} GHz ~ {freqs[-1]/1e9:.3f} GHz")
    print(f"散射点数量: {len(scatterers)}")
    print(f"几何产生的真实时延范围: {tau.min()*1e9:.3f} ns ~ {tau.max()*1e9:.3f} ns")

    # 1) 频域响应幅度
    plt.figure()
    plt.plot(freqs / 1e9, 20*np.log10(np.abs(S) + 1e-12))
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("|S(f)| (dB, arbitrary)")
    plt.title("Stepped-frequency CW response magnitude")

    # 2) 时域脉冲响应幅度（近似“深度 profile”）
    plt.figure()
    plt.plot(t * 1e9, np.abs(h))
    plt.xlabel("Delay (ns)")
    plt.ylabel("|h(t)| (arb. unit)")
    plt.title("Time-domain impulse response via IFFT (SFCW)")

    # 可以按 c/2 * t 换成“等效距离”，这里先不转
    # 3) 几何示意图
    plt.figure()
    plt.scatter(scatterers[:, 0], scatterers[:, 1], s=5, label="Scatterers")
    plt.scatter(-0.5, 0.0, marker='^', label="Tx")
    plt.scatter(0.5, 0.0, marker='s', label="Rx")
    plt.gca().invert_yaxis()
    plt.xlabel("x (m)")
    plt.ylabel("z (m)")
    plt.legend()
    plt.title("Sweet potato scatterers (2D area, SFCW)")

    plt.show()