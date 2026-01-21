# src/rfspsim/propagation/planar.py
from __future__ import annotations
import numpy as np
from typing import Tuple


def _dist(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.linalg.norm(p - q))


def _incidence_angle_from_segment(p_from: np.ndarray, p_to: np.ndarray, interface_z: float) -> float:
    """
    Compute the incidence angle (vs. normal) for a ray from p_from -> p_to (p_to on the interface).
    The interface is the horizontal line z=interface_z with a vertical normal.
    """
    dx = float(p_to[0] - p_from[0])
    dz = float(interface_z - p_from[1])  # Vertical distance from point to interface (should be positive; abs also ok)
    L = float(np.hypot(dx, dz))
    if L < 1e-15:
        return 0.0
    sin_theta = abs(dx) / L
    sin_theta = min(1.0, max(0.0, sin_theta))
    return float(np.arcsin(sin_theta))


def snell_crossing_x(p1: np.ndarray, v1: float, p2: np.ndarray, v2: float,
                     interface_z: float = 0.0,
                     tol: float = 1e-12, max_iter: int = 200) -> float:
    """
    Find x* on the planar interface z=interface_z that minimizes travel time from p1 (medium 1) to p2 (medium 2).

    Solve for the minimum of travel time t(x) = L1(x)/v1 + L2(x)/v2 (equivalent to Snell).
    """
    x1, z1 = float(p1[0]), float(p1[1])
    x2, z2 = float(p2[0]), float(p2[1])

    h1 = abs(z1 - interface_z)
    h2 = abs(z2 - interface_z)

    # If a point is on the interface, the crossing x is that point's x
    if h1 < 1e-12:
        return x1
    if h2 < 1e-12:
        return x2

    def g(x: float) -> float:
        L1 = float(np.hypot(x - x1, h1))
        L2 = float(np.hypot(x - x2, h2))
        return (x - x1) / (v1 * L1) + (x - x2) / (v2 * L2)

    # First find a bracketing interval
    span = (abs(x2 - x1) + h1 + h2 + 1.0)
    lo = min(x1, x2) - span
    hi = max(x1, x2) + span
    glo, ghi = g(lo), g(hi)

    expand = 0
    while glo * ghi > 0 and expand < 60:
        span *= 2.0
        lo = min(x1, x2) - span
        hi = max(x1, x2) + span
        glo, ghi = g(lo), g(hi)
        expand += 1

    # Should always bracket; if not, fall back to midpoint (rare)
    if glo * ghi > 0:
        return 0.5 * (x1 + x2)

    # Bisection
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        gmid = g(mid)
        if abs(gmid) < tol:
            return mid
        if glo * gmid <= 0:
            hi, ghi = mid, gmid
        else:
            lo, glo = mid, gmid

    return 0.5 * (lo + hi)


def refracted_two_segment(p1: np.ndarray, p2: np.ndarray,
                          v1: float, v2: float,
                          interface_z: float = 0.0) -> Tuple[np.ndarray, float, float, float, float]:
    """
    Return geometric info for the refracted path across the interface:
      - cross: interface crossing point (2,)
      - L1: length in medium 1
      - L2: length in medium 2
      - theta_i: incidence angle in medium 1 (vs. normal)
      - theta_t: refracted angle in medium 2 (geometry angle from cross to p2)
    """
    x_cross = snell_crossing_x(p1, v1, p2, v2, interface_z=interface_z)
    cross = np.array([x_cross, interface_z], dtype=float)

    L1 = _dist(p1, cross)
    L2 = _dist(p2, cross)

    theta_i = _incidence_angle_from_segment(p1, cross, interface_z)
    # Compute refracted angle geometrically (reasonable for lossless Snell case)
    # theta_t is on medium-2 side, angle from cross -> p2
    dx2 = float(p2[0] - cross[0])
    dz2 = float(p2[1] - interface_z)
    L2_safe = float(np.hypot(dx2, dz2))
    if L2_safe < 1e-15:
        theta_t = 0.0
    else:
        sin_theta_t = abs(dx2) / L2_safe
        sin_theta_t = min(1.0, max(0.0, sin_theta_t))
        theta_t = float(np.arcsin(sin_theta_t))

    return cross, float(L1), float(L2), float(theta_i), float(theta_t)
