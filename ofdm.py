import numpy as np
import matplotlib.pyplot as plt


def make_ellipse_area_scatterers(center, a, b, n_r=20, n_theta=36, seed=None):
    """
    在椭圆内部随机生成 2D 散射点：
      (x - x0)^2 / a^2 + (z - z0)^2 / b^2 <= 1

    随机均匀采样面积，点数 = n_r * n_theta。
    """
    center = np.array(center, dtype=float)
    n_points = n_r * n_theta

    rng = np.random.default_rng(seed)
    # 随机半径用 sqrt 保证面积均匀，角度均匀分布 [0, 2π)
    rs = np.sqrt(rng.random(n_points))
    thetas = rng.random(n_points) * 2.0 * np.pi

    xs = center[0] + a * rs * np.cos(thetas)
    zs = center[1] + b * rs * np.sin(thetas)

    scatterers = np.column_stack((xs, zs))  # (N, 2)，N = n_r * n_theta
    return scatterers


def build_channel(center=(0.0, 0.4), a=0.15, b=0.07,
                  tx=(-0.5, 0.0), rx=(0.5, 0.0),
                  n_r=20, n_theta=36, c=3e8, seed=None):
    """
    根据几何生成散射点，并计算:
      - 每个点的总路径长度 d_i
      - 时延 tau_i
      - 振幅 A_i  (包含“单元面积/路径衰减”)
    """
    tx_arr = np.array(tx, dtype=float)
    rx_arr = np.array(rx, dtype=float)

    scatterers = make_ellipse_area_scatterers(center, a, b, n_r, n_theta, seed=seed)
    n_scatter = scatterers.shape[0]

    d_tx = np.linalg.norm(scatterers - tx_arr, axis=1)
    d_rx = np.linalg.norm(scatterers - rx_arr, axis=1)
    d = d_tx + d_rx
    tau = d / c

    # 每个散射单元的“面积 + 距离衰减”
    cell_area = np.pi * a * b / n_scatter
    A = cell_area / (d**2 + 1e-6)

    return scatterers, A, tau


def generate_wifi_like_baseband(bw=20e6, n_sub=64, n_sym=10, cp_ratio=0.25, seed=None):
    """
    生成一个“WiFi 风格”的 OFDM 基带复信号（不严格按 802.11，只是结构类似）:

      - 总带宽约为 bw
      - 子载波数 n_sub
      - OFDM 符号数 n_sym
      - 循环前缀长度比例 cp_ratio

    返回:
      t       : 时间轴 (s)
      s_bb    : 复基带信号 (长度 Ns)
      fs      : 采样率 (Hz)
      sym_len : 单个 OFDM 符号（不含 CP）的采样点数
    """
    rng = np.random.default_rng(seed)

    # 简化：采样率 fs ≈ 带宽 bw
    fs = bw
    delta_f = bw / n_sub          # 子载波间隔
    T_sym = 1.0 / delta_f         # 有效符号时长
    sym_len = n_sub               # 不含 CP 的采样点数
    dt = 1.0 / fs

    n_cp = int(sym_len * cp_ratio)
    total_len = n_sym * (sym_len + n_cp)

    s_bb = np.zeros(total_len, dtype=complex)

    # 子载波索引（概念用，不直接用 k 做别的）
    k = np.arange(-n_sub // 2, n_sub // 2)

    ptr = 0
    for _ in range(n_sym):
        # QPSK 符号
        bits = rng.integers(0, 4, size=n_sub)
        const = np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2.0)
        Xk = const[bits]

        # IFFT 生成时域 OFDM 符号（基带）
        x_time = np.fft.ifft(np.fft.ifftshift(Xk))  # 长度 n_sub

        # 加循环前缀
        x_cp = np.concatenate([x_time[-n_cp:], x_time])

        s_bb[ptr:ptr + len(x_cp)] = x_cp
        ptr += len(x_cp)

    t = np.arange(total_len) * dt
    return t, s_bb, fs, sym_len


def apply_multipath(tx_bb, A, tau, fs):
    """
    将基带信号 tx_bb 通过多径信道:
        h(t) = sum_i A_i delta(t - tau_i)

    用离散时移近似：n_i = round(tau_i * fs)
    返回 rx_bb（复基带）
    """
    tx_bb = np.asarray(tx_bb)
    n = len(tx_bb)
    delay_samples = np.round(tau * fs).astype(int)
    max_delay = int(delay_samples.max())
    rx_bb = np.zeros(n + max_delay, dtype=complex)

    for Ai, di in zip(A, delay_samples):
        rx_bb[di:di + n] += Ai * tx_bb

    return rx_bb


if __name__ == "__main__":
    # ------------ WiFi 风格版本示例（OFDM 基带 + 多径）------------
    fc = 2.4e9  # 载波中心频率（这里只是个“标签”，真正仿真在基带完成）

    # 1) 建立几何 + 信道（同一颗红薯）
    scatterers, A, tau = build_channel(seed=0)
    print(f"散射点数量: {len(scatterers)}")
    print(f"时延范围: {tau.min()*1e9:.3f} ns ~ {tau.max()*1e9:.3f} ns")

    # 2) 生成 WiFi 风格 OFDM 基带信号
    t_bb, tx_bb, fs, sym_len = generate_wifi_like_baseband(
        bw=20e6, n_sub=64, n_sym=10, seed=1
    )
    print(f"基带采样率 fs = {fs/1e6:.1f} MHz, 信号长度 {len(tx_bb)} 点")

    # 3) 通过多径信道
    rx_bb = apply_multipath(tx_bb, A, tau, fs)

    # 对齐时间轴
    t_rx = np.arange(len(rx_bb)) / fs

    # 4) 画图：基带实部/幅度对比
    plt.figure()
    # plt.plot(t_bb * 1e6, np.real(tx_bb), label="Tx baseband (real)")
    plt.plot(t_bb * 1e6, np.abs(tx_bb), label="|Tx baseband|")
    plt.plot(t_rx * 1e6, np.real(rx_bb), label="Rx baseband (real)", alpha=0.7)
    plt.xlabel("t (µs)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.title("WiFi-like OFDM baseband (real part)")

    plt.figure()
    plt.plot(t_rx * 1e6, np.abs(rx_bb), label="|Rx baseband|")
    plt.xlabel("t (µs)")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.title("Rx baseband magnitude (WiFi-like)")

    # 5) 几何示意图
    plt.figure()
    plt.scatter(scatterers[:, 0], scatterers[:, 1], s=5, label="Scatterers")
    plt.scatter(-0.5, 0.0, marker='^', label="Tx")
    plt.scatter(0.5, 0.0, marker='s', label="Rx")
    plt.gca().invert_yaxis()
    plt.xlabel("x (m)")
    plt.ylabel("z (m)")
    plt.legend()
    plt.title("Sweet potato scatterers (2D area)")

    plt.show()