import numpy as np
import matplotlib.pyplot as plt


def make_ellipse_area_scatterers(center, a, b, n_r=20, n_theta=36, seed=None):
    """
    Randomly generate 2D scatterers inside an ellipse:
      (x - x0)^2 / a^2 + (z - z0)^2 / b^2 <= 1

    Sample uniformly by area with n_r * n_theta points.
    """
    center = np.array(center, dtype=float)
    n_points = n_r * n_theta

    rng = np.random.default_rng(seed)
    # Use sqrt on radius for uniform area; angles uniform in [0, 2π).
    rs = np.sqrt(rng.random(n_points))
    thetas = rng.random(n_points) * 2.0 * np.pi

    xs = center[0] + a * rs * np.cos(thetas)
    zs = center[1] + b * rs * np.sin(thetas)

    scatterers = np.column_stack((xs, zs))  # (N, 2), N = n_r * n_theta
    return scatterers


def build_channel(center=(0.0, 0.4), a=0.15, b=0.07,
                  tx=(-0.5, 0.0), rx=(0.5, 0.0),
                  n_r=20, n_theta=36, c=3e8, seed=None):
    """
    From the geometry, generate scatterers and compute:
      - Total path length d_i for each point
      - Delay tau_i
      - Amplitude A_i (includes cell area and path attenuation)
    """
    tx_arr = np.array(tx, dtype=float)
    rx_arr = np.array(rx, dtype=float)

    scatterers = make_ellipse_area_scatterers(center, a, b, n_r, n_theta, seed=seed)
    n_scatter = scatterers.shape[0]

    d_tx = np.linalg.norm(scatterers - tx_arr, axis=1)
    d_rx = np.linalg.norm(scatterers - rx_arr, axis=1)
    d = d_tx + d_rx
    tau = d / c

    # Each scatter cell's area and distance attenuation
    cell_area = np.pi * a * b / n_scatter
    A = cell_area / (d**2 + 1e-6)

    return scatterers, A, tau


def generate_wifi_like_baseband(bw=20e6, n_sub=64, n_sym=10, cp_ratio=0.25, seed=None):
    """
    Generate a WiFi-style complex OFDM baseband (structure similar to 802.11, not exact):

      - Total bandwidth ≈ bw
      - Number of subcarriers n_sub
      - Number of OFDM symbols n_sym
      - Cyclic prefix length ratio cp_ratio

    Returns:
      t       : time axis (s)
      s_bb    : complex baseband signal (length Ns)
      fs      : sample rate (Hz)
      sym_len : samples per OFDM symbol without CP
    """
    rng = np.random.default_rng(seed)

    # Simplification: sample rate fs ≈ bandwidth bw
    fs = bw
    delta_f = bw / n_sub          # Subcarrier spacing
    T_sym = 1.0 / delta_f         # Useful symbol duration
    sym_len = n_sub               # Samples per symbol without CP
    dt = 1.0 / fs

    n_cp = int(sym_len * cp_ratio)
    total_len = n_sym * (sym_len + n_cp)

    s_bb = np.zeros(total_len, dtype=complex)

    # Conceptual subcarrier indices
    k = np.arange(-n_sub // 2, n_sub // 2)

    ptr = 0
    for _ in range(n_sym):
        # QPSK symbols
        bits = rng.integers(0, 4, size=n_sub)
        const = np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2.0)
        Xk = const[bits]

        # IFFT to produce the time-domain OFDM symbol (baseband)
        x_time = np.fft.ifft(np.fft.ifftshift(Xk))  # length n_sub

        # Add cyclic prefix
        x_cp = np.concatenate([x_time[-n_cp:], x_time])

        s_bb[ptr:ptr + len(x_cp)] = x_cp
        ptr += len(x_cp)

    t = np.arange(total_len) * dt
    return t, s_bb, fs, sym_len


def apply_multipath(tx_bb, A, tau, fs):
    """
    Pass baseband signal tx_bb through a multipath channel:
        h(t) = sum_i A_i delta(t - tau_i)

    Approximate with discrete shifts: n_i = round(tau_i * fs)
    Returns rx_bb (complex baseband)
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
    # ------------ WiFi-style example (OFDM baseband + multipath) ------------
    fc = 2.4e9  # Carrier center frequency (label only; simulation runs in baseband)

    # 1) Build geometry + channel (same sweet potato)
    scatterers, A, tau = build_channel(seed=0)
    print(f"Number of scatterers: {len(scatterers)}")
    print(f"Delay range: {tau.min()*1e9:.3f} ns ~ {tau.max()*1e9:.3f} ns")

    # 2) Generate WiFi-style OFDM baseband signal
    t_bb, tx_bb, fs, sym_len = generate_wifi_like_baseband(
        bw=20e6, n_sub=64, n_sym=10, seed=1
    )
    print(f"Baseband sample rate fs = {fs/1e6:.1f} MHz, signal length {len(tx_bb)} samples")

    # 3) Pass through multipath channel
    rx_bb = apply_multipath(tx_bb, A, tau, fs)

    # Align time axes
    t_rx = np.arange(len(rx_bb)) / fs

    # 4) Plot: baseband real/magnitude comparison
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

    # 5) Geometry visualization
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
