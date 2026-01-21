# src/rfspsim/propagation/planar.py
from __future__ import annotations
import numpy as np
from typing import Tuple


def _dist(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.linalg.norm(p - q))


def _incidence_angle_from_segment(p_from: np.ndarray, p_to: np.ndarray, interface_z: float) -> float:
    """
    计算射线从 p_from -> p_to（p_to 在界面上）的入射角（相对于法线）。
    界面是水平线 z=interface_z，法线方向是竖直方向。
    """
    dx = float(p_to[0] - p_from[0])
    dz = float(interface_z - p_from[1])  # 从点到界面（竖直距离，应该为正，取绝对也行）
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
    求 p1（介质1一侧）-> p2（介质2一侧）跨越平面界面 z=interface_z 的最短时间路径交点 x*。

    通过求解 travel time t(x) = L1(x)/v1 + L2(x)/v2 的最小值（等价 Snell）。
    """
    x1, z1 = float(p1[0]), float(p1[1])
    x2, z2 = float(p2[0]), float(p2[1])

    h1 = abs(z1 - interface_z)
    h2 = abs(z2 - interface_z)

    # 如果某个点就在界面上，交点就是该点的 x
    if h1 < 1e-12:
        return x1
    if h2 < 1e-12:
        return x2

    def g(x: float) -> float:
        L1 = float(np.hypot(x - x1, h1))
        L2 = float(np.hypot(x - x2, h2))
        return (x - x1) / (v1 * L1) + (x - x2) / (v2 * L2)

    # 先找一个能夹住根的区间
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

    # 理论上应该总能夹住；如果没夹住，退化返回中点（极少发生）
    if glo * ghi > 0:
        return 0.5 * (x1 + x2)

    # 二分
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
    返回跨界面折射路径的几何信息：
      - cross: 界面交点 (2,)
      - L1: 介质1段长度
      - L2: 介质2段长度
      - theta_i: 介质1入射角（相对于法线）
      - theta_t: 介质2折射角（几何上从交点到 p2 的角）
    """
    x_cross = snell_crossing_x(p1, v1, p2, v2, interface_z=interface_z)
    cross = np.array([x_cross, interface_z], dtype=float)

    L1 = _dist(p1, cross)
    L2 = _dist(p2, cross)

    theta_i = _incidence_angle_from_segment(p1, cross, interface_z)
    # 折射角用几何计算（对无损 Snell 情况是合理的）
    # 这里 theta_t 是介质2侧，cross -> p2 的角
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