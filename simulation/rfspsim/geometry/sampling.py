# src/rfspsim/geometry/sampling.py
from __future__ import annotations
import numpy as np
from typing import Tuple, Optional


def sample_ellipse_area(center, a: float, b: float, n_points: int, seed: Optional[int] = None) -> np.ndarray:
    """
    在椭圆内部均匀（面积意义上）随机采样点：
      (x-x0)^2/a^2 + (z-z0)^2/b^2 <= 1
    """
    center = np.array(center, dtype=float)
    rng = np.random.default_rng(seed)

    r = np.sqrt(rng.random(n_points))            # sqrt 保证面积均匀
    theta = rng.random(n_points) * 2.0 * np.pi

    x = center[0] + a * r * np.cos(theta)
    z = center[1] + b * r * np.sin(theta)
    return np.column_stack([x, z])


def sample_surface_line(x_min: float, x_max: float, n_points: int,
                        z: float = 0.0, seed: Optional[int] = None,
                        method: str = "uniform") -> Tuple[np.ndarray, float]:
    """
    在地表线段 [x_min, x_max] 上生成散射点（z固定为 interface_z）。

    返回:
      points: (N,2)
      cell_len: 每个点代表的“线段长度权重”≈(x_max-x_min)/N
    """
    if n_points <= 0:
        raise ValueError("n_points must be > 0")
    if x_max <= x_min:
        raise ValueError("x_max must be > x_min")

    cell_len = (x_max - x_min) / n_points
    method = method.lower()

    if method == "uniform":
        xs = x_min + (np.arange(n_points) + 0.5) * cell_len  # 取每格中点
    elif method == "random":
        rng = np.random.default_rng(seed)
        xs = rng.uniform(x_min, x_max, size=n_points)
    else:
        raise ValueError("method must be 'uniform' or 'random'")

    zs = np.full_like(xs, fill_value=z, dtype=float)
    return np.column_stack([xs, zs]), float(cell_len)