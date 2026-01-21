# src/rfspsim/propagation/channel_builders.py
from __future__ import annotations
import numpy as np
from typing import Dict, Optional

from rfspsim.media.medium import Medium
from rfspsim.media.interface import reflection_coeff, transmission_coeff
from rfspsim.propagation.planar import refracted_two_segment

def build_los_taps(
    tx: np.ndarray,
    rx: np.ndarray,
    air: Medium,
    los_coupling: complex = 1.0 + 0j,
    model: str = "inv_d2",
) -> Dict[str, np.ndarray]:
    """
    LOS / Tx->Rx 直达（都在空气中）：
      路径: Tx(air) -> Rx(air)

    delay = |Tx-Rx| / v_air

    gain:
      - model="inv_d2": ~ los_coupling / d^2
      - model="inv_d" : ~ los_coupling / d
      - model="const" : ~ los_coupling
    """
    tx = np.asarray(tx, dtype=float)
    rx = np.asarray(rx, dtype=float)

    d = float(np.linalg.norm(rx - tx))
    tau = d / air.v

    model = model.lower()
    if model == "inv_d2":
        g = los_coupling / (d**2 + 1e-12)
    elif model == "inv_d":
        g = los_coupling / (d + 1e-12)
    elif model == "const":
        g = los_coupling
    else:
        raise ValueError("model must be 'inv_d2', 'inv_d', or 'const'")

    return {
        "delays": np.array([tau], dtype=float),
        "gains": np.array([g], dtype=complex),
    }

def build_surface_scatter_taps(
    tx: np.ndarray,
    rx: np.ndarray,
    surface_points: np.ndarray,
    air: Medium,
    soil: Medium,
    interface_z: float = 0.0,
    per_point_length: float = 1.0,
    surface_reflectivity: float = 1.0,
    pol: str = "avg",
    include_fresnel: bool = True,
) -> Dict[str, np.ndarray]:
    """
    地表散射（clutter）：
      每个 surface point 当作一个点散射体，
      路径: Tx(air) -> S(interface) -> Rx(air)

    gain ~ per_point_length / (L_total^2) * surface_reflectivity * Gamma(air->soil)

    返回 dict:
      points (N,2), delays (N,), gains (N,)
    """
    tx = np.asarray(tx, dtype=float)
    rx = np.asarray(rx, dtype=float)
    pts = np.asarray(surface_points, dtype=float)

    v_air = air.v

    delays = np.zeros(len(pts), dtype=float)
    gains = np.zeros(len(pts), dtype=complex)

    for i, s in enumerate(pts):
        L1 = float(np.linalg.norm(s - tx))
        L2 = float(np.linalg.norm(rx - s))
        Ltot = L1 + L2
        delays[i] = Ltot / v_air

        # 入射角近似用 Tx->S 的几何角
        if L1 < 1e-12:
            theta_i = 0.0
        else:
            sin_theta = abs(s[0] - tx[0]) / L1
            sin_theta = min(1.0, max(0.0, sin_theta))
            theta_i = float(np.arcsin(sin_theta))

        Gamma = 1.0 + 0j
        if include_fresnel:
            Gamma = reflection_coeff(air, soil, theta_i, pol=pol)

        gains[i] = (per_point_length * surface_reflectivity * Gamma) / (Ltot**2 + 1e-12)

    return {"points": pts, "delays": delays, "gains": gains}


def build_target_reflection_taps(
    tx: np.ndarray,
    rx: np.ndarray,
    target_points: np.ndarray,
    air: Medium,
    soil: Medium,
    interface_z: float = 0.0,
    per_point_area: float = 1.0,
    target_reflectivity: complex = 1.0 + 0j,
    pol: str = "avg",
    include_fresnel: bool = True,
) -> Dict[str, np.ndarray]:
    """
    地下目标（红薯）散射点回波：
      路径: Tx(air) -> (折射入土) -> target_point(soil) -> (折射出土) -> Rx(air)

    delay = (L_air_in/va + L_soil_in/vs) + (L_soil_out/vs + L_air_out/va)

    gain  ~ per_point_area/(L_total^2) * target_reflectivity * T_in(air->soil)*T_out(soil->air)
    """
    tx = np.asarray(tx, dtype=float)
    rx = np.asarray(rx, dtype=float)
    pts = np.asarray(target_points, dtype=float)

    delays = np.zeros(len(pts), dtype=float)
    gains = np.zeros(len(pts), dtype=complex)

    entry_points = np.zeros_like(pts)
    exit_points = np.zeros_like(pts)

    va, vs = air.v, soil.v

    for i, p in enumerate(pts):
        # 入土段：tx(air) -> p(soil)
        cross_in, L_air_in, L_soil_in, theta_air_i, _ = refracted_two_segment(tx, p, va, vs, interface_z=interface_z)

        # 出土段：p(soil) -> rx(air)
        cross_out, L_soil_out, L_air_out, theta_soil_i, _ = refracted_two_segment(p, rx, vs, va, interface_z=interface_z)

        entry_points[i] = cross_in
        exit_points[i] = cross_out

        tau = (L_air_in / va + L_soil_in / vs) + (L_soil_out / vs + L_air_out / va)
        delays[i] = tau

        Tin = 1.0 + 0j
        Tout = 1.0 + 0j
        if include_fresnel:
            Tin = transmission_coeff(air, soil, theta_air_i, pol=pol)
            Tout = transmission_coeff(soil, air, theta_soil_i, pol=pol)

        Ltot = L_air_in + L_soil_in + L_soil_out + L_air_out
        gains[i] = (per_point_area * target_reflectivity * Tin * Tout) / (Ltot**2 + 1e-12)

    return {
        "points": pts,
        "entry_points": entry_points,
        "exit_points": exit_points,
        "delays": delays,
        "gains": gains,
    }