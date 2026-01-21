import numpy as np
import matplotlib.pyplot as plt


def make_ellipse_area_scatterers(center, a, b, n_r=20, n_theta=36, seed=None):
    """
    Randomly generate 2D scatterers inside an ellipse:
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
    Same as the WiFi version: geometry + multi-scatterer channel params (A_i, tau_i).
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
    f_start=2e9,      # Start frequency (e.g., 2 GHz as in the text)
    f_step=40e6,      # Step size (e.g., 2.000 → 2.040 → 2.080 ...)
    n_steps=51,       # Number of frequency points
    center=(0.0, 0.4),
    a=0.15, b=0.07,
    tx=(-0.5, 0.0), rx=(0.5, 0.0),
    n_r=20, n_theta=36,
    c=3e8, seed=None
):
    """
    Simulate stepped-frequency CW:

    For each tone fk:
        S(fk) = sum_i A_i * exp(-j 2π fk * tau_i)

    Then IFFT S(fk) to obtain an equivalent impulse response h(t), representing reflections vs. depth.
    """
    scatterers, A, tau = build_channel(center, a, b, tx, rx, n_r, n_theta, c, seed)

    # Frequency sweep: f_start, f_start+f_step, ...
    freqs = f_start + np.arange(n_steps) * f_step

    # For IFFT we only care about the baseband frequency offsets f_bb starting from 0
    f_bb = freqs - freqs[0]

    S = np.zeros(n_steps, dtype=complex)
    for k, fk in enumerate(f_bb):
        S[k] = np.sum(A * np.exp(-1j * 2.0 * np.pi * fk * tau))

    # IFFT -> time-domain impulse response
    delta_f = f_step
    T_win = 1.0 / delta_f      # Maximum unaliased delay window
    dt = T_win / n_steps       # Time-domain sampling interval
    t = np.arange(n_steps) * dt
    h = np.fft.ifft(S)

    return {
        "scatterers": scatterers,
        "A": A,
        "tau": tau,
        "freqs": freqs,    # Actual swept carrier freqs
        "S": S,            # Complex response at each tone
        "t": t,            # Time-domain (delay)
        "h": h,            # Time-domain impulse response
        "f_start": f_start,
        "f_step": f_step,
        "n_steps": n_steps,
    }


if __name__ == "__main__":
    # ------------ stepped-frequency CW example ------------
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

    print(f"Sweep range: {freqs[0]/1e9:.3f} GHz ~ {freqs[-1]/1e9:.3f} GHz")
    print(f"Number of scatterers: {len(scatterers)}")
    print(f"Delay range from geometry: {tau.min()*1e9:.3f} ns ~ {tau.max()*1e9:.3f} ns")

    # 1) Frequency-domain magnitude
    plt.figure()
    plt.plot(freqs / 1e9, 20*np.log10(np.abs(S) + 1e-12))
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("|S(f)| (dB, arbitrary)")
    plt.title("Stepped-frequency CW response magnitude")

    # 2) Time-domain impulse response magnitude (approx depth profile)
    plt.figure()
    plt.plot(t * 1e9, np.abs(h))
    plt.xlabel("Delay (ns)")
    plt.ylabel("|h(t)| (arb. unit)")
    plt.title("Time-domain impulse response via IFFT (SFCW)")

    # Could map to equivalent range via c/2 * t; keeping delay for now
    # 3) Geometry diagram
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
