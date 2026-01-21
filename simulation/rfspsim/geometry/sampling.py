# src/rfspsim/geometry/sampling.py
from __future__ import annotations
import numpy as np
from typing import Tuple, Optional


def sample_ellipse_area(center, a: float, b: float, n_points: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Sample points uniformly by area inside an ellipse:
      (x-x0)^2/a^2 + (z-z0)^2/b^2 <= 1
    """
    center = np.array(center, dtype=float)
    rng = np.random.default_rng(seed)

    r = np.sqrt(rng.random(n_points))            # sqrt keeps area sampling uniform
    theta = rng.random(n_points) * 2.0 * np.pi

    x = center[0] + a * r * np.cos(theta)
    z = center[1] + b * r * np.sin(theta)
    return np.column_stack([x, z])


def sample_surface_line(x_min: float, x_max: float, n_points: int,
                        z: float = 0.0, seed: Optional[int] = None,
                        method: str = "uniform") -> Tuple[np.ndarray, float]:
    """
    Generate scatterers along the surface segment [x_min, x_max] (z fixed at interface_z).

    Returns:
      points: (N,2)
      cell_len: length weight each point represents ≈ (x_max - x_min)/N
    """
    if n_points <= 0:
        raise ValueError("n_points must be > 0")
    if x_max <= x_min:
        raise ValueError("x_max must be > x_min")

    cell_len = (x_max - x_min) / n_points
    method = method.lower()

    if method == "uniform":
        xs = x_min + (np.arange(n_points) + 0.5) * cell_len  # midpoint of each cell
    elif method == "random":
        rng = np.random.default_rng(seed)
        xs = rng.uniform(x_min, x_max, size=n_points)
    else:
        raise ValueError("method must be 'uniform' or 'random'")

    zs = np.full_like(xs, fill_value=z, dtype=float)
    return np.column_stack([xs, zs]), float(cell_len)
