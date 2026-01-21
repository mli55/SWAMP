# src/rfspsim/media/interface.py
from __future__ import annotations
import numpy as np
from typing import Optional, Tuple

from rfspsim.media.medium import Medium


def snell_theta_t(m1: Medium, m2: Medium, theta_i: float) -> Optional[float]:
    """
    Snell: n1 sin(theta_i) = n2 sin(theta_t)
    Return theta_t (radians). If total internal reflection occurs (sin>1), return None.
    """
    n1, n2 = m1.n, m2.n
    sin_t = (n1 / n2) * np.sin(theta_i)
    if np.abs(sin_t) > 1.0:
        return None
    return float(np.arcsin(sin_t))


def fresnel_rt(m1: Medium, m2: Medium, theta_i: float, pol: str = "avg") -> Tuple[complex, complex]:
    """
    Lossless Fresnel reflection/transmission coefficients (field amplitude).

    pol:
      - "TE" : s-polarization
      - "TM" : p-polarization
      - "avg": simple average of TE and TM (not a strict power average but sufficient for Step2)

    Returns:
      (Gamma, Tau)
      Gamma: reflection coefficient (E_r / E_i)
      Tau  : transmission coefficient (E_t / E_i)
    """
    theta_t = snell_theta_t(m1, m2, theta_i)

    # Total internal reflection: set |Gamma|=1, Tau=0
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
    """Return only reflection coefficient Gamma"""
    g, _ = fresnel_rt(m1, m2, theta_i, pol=pol)
    return g


def transmission_coeff(m1: Medium, m2: Medium, theta_i: float, pol: str = "avg") -> complex:
    """Return only transmission coefficient Tau"""
    _, t = fresnel_rt(m1, m2, theta_i, pol=pol)
    return t
