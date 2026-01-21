# src/rfspsim/simulators/wifi_ofdm_sim.py
from __future__ import annotations
import numpy as np
from typing import Dict

from rfspsim.waveforms.ofdm_waveform import generate_ofdm_grid, ofdm_modulate, subcarrier_frequencies_shifted


def channel_H(f_offsets: np.ndarray, delays: np.ndarray, gains: np.ndarray) -> np.ndarray:
    """
    在一组频率偏移 f_offsets 上计算基带频域通道响应：
      H(f) = Σ g_i * exp(-j2π f τ_i)
    """
    delays = np.asarray(delays, dtype=float)
    gains = np.asarray(gains, dtype=complex)
    exp_mat = np.exp(-1j * 2.0 * np.pi * f_offsets[:, None] * delays[None, :])
    return exp_mat @ gains  # (n_fft,)


def simulate_wifi_ofdm(
    bw: float,
    n_fft: int,
    n_sym: int,
    cp_len: int,
    delays: np.ndarray,
    gains: np.ndarray,
    seed: int = 0
) -> Dict[str, np.ndarray]:
    """
    用“频域乘法”的方式把通道施加到 OFDM 上（更适合你现在这种 ns 级延时）。
    """
    grid = generate_ofdm_grid(n_fft=n_fft, n_sym=n_sym, seed=seed)
    X = grid["X_shifted"]

    f_offsets = subcarrier_frequencies_shifted(n_fft=n_fft, bw=bw)
    H = channel_H(f_offsets, delays, gains)  # (n_fft,)

    # 每个符号同一条静态信道
    Y = X * H[None, :]

    tx_time = ofdm_modulate(X, cp_len=cp_len)
    rx_time = ofdm_modulate(Y, cp_len=cp_len)

    return {
        "f_offsets": f_offsets,
        "H": H,
        "X": X,
        "Y": Y,
        "tx_time": tx_time,
        "rx_time": rx_time,
    }