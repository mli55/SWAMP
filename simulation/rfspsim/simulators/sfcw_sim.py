# src/rfspsim/simulators/sfcw_sim.py
from __future__ import annotations
import numpy as np
from typing import Dict, Optional


def simulate_sfcw(
    f_start: float,
    f_step: float,
    n_steps: int,
    delays: np.ndarray,
    gains: np.ndarray,
    baseband: bool = True,
    probe_tone_hz: Optional[float] = None,
    probe_duration: float = 5e-3,
    probe_fs: float = 48e3,
) -> Dict[str, np.ndarray | None]:
    """
    SFCW (stepped-frequency CW) sweep simulation.

    Complex response measured at each tone after downconverting to baseband:
      S(f_bb) = Σ gain_i * exp(-j 2π f_bb * delay_i)

    probe_tone_hz simulates sending a baseband sine at each tone (e.g., 1 kHz):
      - probe_tone_hz = 0 or None: equivalent to DC, giving the ideal CFR (original behavior)
      - probe_tone_hz > 0: synthesize a baseband sine of length probe_duration at probe_fs,
        multiply by the ideal CFR, then take the mean and the coherent-demod mean to match real TX/RX extraction.

    baseband=True means the measured response is already downconverted to f_start,
    so h(t) from the IFFT looks like a typical radar impulse response.

    Returns:
      freqs: absolute tones (Hz)
      f_bb : baseband frequency offsets (Hz)
      S    : complex response (if probe_tone_hz>0, after coherent demod)
      S_ideal: ideal CFR from taps
      S_probe_naive: mean without demod (tone≠0 should be near 0)
      probe_t: time axis of the synthesized probe, or None if not generated
      t    : delay axis for the IFFT
      h    : equivalent impulse response from IFFT
    """
    delays = np.asarray(delays, dtype=float)
    gains = np.asarray(gains, dtype=complex)

    freqs = f_start + np.arange(n_steps) * f_step
    f = freqs - freqs[0] if baseband else freqs

    # Frequency-domain response
    # S[k] = Σ g_i * exp(-j2π f[k] τ_i)
    exp_mat = np.exp(-1j * 2.0 * np.pi * f[:, None] * delays[None, :])
    S_ideal = exp_mat @ gains

    # Simulate "transmit 1 kHz tone + RX demod/average" chain
    tone = probe_tone_hz if probe_tone_hz is not None else 0.0
    probe_t = None
    S_probe_naive = None

    if tone != 0.0:
        n_samples = int(round(probe_duration * probe_fs))
        if n_samples <= 0:
            raise ValueError("probe_duration * probe_fs must yield at least 1 sample")

        probe_t = np.arange(n_samples) / probe_fs
        tx_probe = np.exp(1j * 2.0 * np.pi * tone * probe_t)
        demod = np.exp(-1j * 2.0 * np.pi * tone * probe_t)

        S_probe_naive = np.zeros(n_steps, dtype=complex)
        S = np.zeros(n_steps, dtype=complex)
        for k, h_k in enumerate(S_ideal):
            rx = h_k * tx_probe
            S_probe_naive[k] = rx.mean()
            S[k] = (rx * demod).mean()
    else:
        S = S_ideal

    # IFFT -> time domain
    # Δf = f_step => max unaliased delay window T = 1/Δf
    delta_f = f_step
    T_win = 1.0 / delta_f
    dt = T_win / n_steps
    t = np.arange(n_steps) * dt
    h = np.fft.ifft(S)

    return {
        "freqs": freqs,
        "f_bb": f,
        "S": S,
        "S_ideal": S_ideal,
        "S_probe_naive": S_probe_naive,
        "probe_t": probe_t,
        "t": t,
        "h": h,
    }
