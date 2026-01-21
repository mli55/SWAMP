# src/rfspsim/waveforms/ofdm_waveform.py
from __future__ import annotations
import numpy as np
from typing import Dict, Tuple, Optional


def _qpsk(n: int, rng: np.random.Generator) -> np.ndarray:
    const = np.array([1+1j, -1+1j, -1-1j, 1-1j], dtype=complex) / np.sqrt(2.0)
    idx = rng.integers(0, 4, size=n)
    return const[idx]


def generate_ofdm_grid(
    n_fft: int = 64,
    n_sym: int = 10,
    seed: Optional[int] = None,
    used_carriers: str = "all_except_dc",
) -> Dict[str, np.ndarray]:
    """
    Generate a WiFi-like OFDM frequency-domain grid (shifted with DC centered).
    Use QPSK only to simplify sensing simulations.

    Returns:
      X_shifted: (n_sym, n_fft) frequency-domain symbols, DC at index n_fft//2
      used_mask: (n_fft,) bool
    """
    rng = np.random.default_rng(seed)

    used_mask = np.ones(n_fft, dtype=bool)
    used_mask[n_fft // 2] = False  # Skip DC

    if used_carriers.lower() == "all_except_dc":
        pass
    else:
        raise ValueError("currently only support used_carriers='all_except_dc'")

    X = np.zeros((n_sym, n_fft), dtype=complex)
    for s in range(n_sym):
        X[s, used_mask] = _qpsk(int(used_mask.sum()), rng)

    return {"X_shifted": X, "used_mask": used_mask}


def ofdm_modulate(X_shifted: np.ndarray, cp_len: int) -> np.ndarray:
    """
    IFFT + CP to turn the frequency grid into a continuous complex baseband waveform.
    X_shifted has DC in the middle (fftshift order).
    """
    n_sym, n_fft = X_shifted.shape
    out = []
    for s in range(n_sym):
        x = np.fft.ifft(np.fft.ifftshift(X_shifted[s]))
        if cp_len > 0:
            x = np.concatenate([x[-cp_len:], x])
        out.append(x)
    return np.concatenate(out)


def subcarrier_frequencies_shifted(n_fft: int, bw: float) -> np.ndarray:
    """
    Return subcarrier frequency offsets (Hz) aligned with X_shifted, DC-centered:
      f[k] = (k - n_fft//2) * (bw / n_fft)
    """
    df = bw / n_fft
    k = np.arange(n_fft) - (n_fft // 2)
    return k * df
