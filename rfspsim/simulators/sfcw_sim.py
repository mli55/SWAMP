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
    SFCW（stepped-frequency CW）扫频仿真：

    在每个频点上测到的复响应（下变频到基带后的表示）：
      S(f_bb) = Σ gain_i * exp(-j 2π f_bb * delay_i)

    probe_tone_hz 用来模拟你真实链路在每个频点发一个基带正弦（例如 1 kHz）的情况：
      - probe_tone_hz = 0 或 None 时，等价于发 DC，直接得到理想 CFR（原来的行为）
      - probe_tone_hz > 0 时，会生成一段长度 probe_duration、采样率 probe_fs 的基带正弦，
        先乘以理想 CFR，再做“均值”和“同频解调后均值”，用来对齐真实 TX/RX 的提取方式。

    baseband=True 表示你“测到的”是已下变频到 f_start 的基带响应，
    这样 IFFT 得到的 h(t) 更像雷达常见的脉冲响应。

    返回:
      freqs: 绝对频点 (Hz)
      f_bb : 基带频率偏移 (Hz)
      S    : 频域复响应（若 probe_tone_hz>0，取同频解调后的值）
      S_ideal: 直接由 taps 计算的理想 CFR
      S_probe_naive: 不解调直接求均值的结果（tone≠0 时应接近 0）
      probe_t: 合成探测信号的时间轴，未生成则为 None
      t    : IFFT 对应的延时轴
      h    : IFFT 后的“等效脉冲响应”
    """
    delays = np.asarray(delays, dtype=float)
    gains = np.asarray(gains, dtype=complex)

    freqs = f_start + np.arange(n_steps) * f_step
    f = freqs - freqs[0] if baseband else freqs

    # 频域响应
    # S[k] = Σ g_i * exp(-j2π f[k] τ_i)
    exp_mat = np.exp(-1j * 2.0 * np.pi * f[:, None] * delays[None, :])
    S_ideal = exp_mat @ gains

    # 模拟“发 1 kHz 正弦 + RX 解调/均值”的链路
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

    # IFFT -> 时域
    # Δf = f_step => 最大无混叠延时窗口 T = 1/Δf
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
