# src/rfspsim/media/interface.py
from __future__ import annotations
import numpy as np
from typing import Optional, Tuple

from rfspsim.media.medium import Medium


def snell_theta_t(m1: Medium, m2: Medium, theta_i: float) -> Optional[float]:
    """
    Snell: n1 sin(theta_i) = n2 sin(theta_t)
    返回 theta_t（弧度）。若发生全反射（sin>1）则返回 None。
    """
    n1, n2 = m1.n, m2.n
    sin_t = (n1 / n2) * np.sin(theta_i)
    if np.abs(sin_t) > 1.0:
        return None
    return float(np.arcsin(sin_t))


def fresnel_rt(m1: Medium, m2: Medium, theta_i: float, pol: str = "avg") -> Tuple[complex, complex]:
    """
    无损 Fresnel 反射/透射系数（电场幅度系数）。

    pol:
      - "TE" : s-polarization
      - "TM" : p-polarization
      - "avg": 简单平均 TE 与 TM（不是严格功率平均，但够用做 Step2）

    返回:
      (Gamma, Tau)
      Gamma: 反射系数 (E_r / E_i)
      Tau  : 透射系数 (E_t / E_i)
    """
    theta_t = snell_theta_t(m1, m2, theta_i)

    # 全反射：这里简单处理为 |Gamma|=1, Tau=0
    if theta_t is None:
        return 1.0 + 0j, 0.0 + 0j

    eta1, eta2 = m1.eta, m2.eta
    ci, ct = np.cos(theta_i), np.cos(theta_t)

    pol = pol.upper()
    if pol == "TE":
        gamma = (eta2 * ci - eta1 * ct) / (eta2 * ci + eta1 * ct)
        tau = (2.0 * eta2 * ci) / (eta2 * ci + eta1 * ct)
        return complex(gamma), complex(tau)

    if pol == "TM":
        gamma = (eta2 * ct - eta1 * ci) / (eta2 * ct + eta1 * ci)
        tau = (2.0 * eta2 * ci) / (eta2 * ct + eta1 * ci)
        return complex(gamma), complex(tau)

    if pol == "AVG":
        g_te, t_te = fresnel_rt(m1, m2, theta_i, pol="TE")
        g_tm, t_tm = fresnel_rt(m1, m2, theta_i, pol="TM")
        return 0.5 * (g_te + g_tm), 0.5 * (t_te + t_tm)

    raise ValueError(f"Unknown pol={pol}. Use 'TE', 'TM', or 'avg'.")


def reflection_coeff(m1: Medium, m2: Medium, theta_i: float, pol: str = "avg") -> complex:
    """只取反射系数 Gamma"""
    g, _ = fresnel_rt(m1, m2, theta_i, pol=pol)
    return g


def transmission_coeff(m1: Medium, m2: Medium, theta_i: float, pol: str = "avg") -> complex:
    """只取透射系数 Tau"""
    _, t = fresnel_rt(m1, m2, theta_i, pol=pol)
    return t