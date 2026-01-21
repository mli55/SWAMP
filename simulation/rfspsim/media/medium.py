# src/rfspsim/media/medium.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from rfspsim.constants import C0, ETA0


@dataclass(frozen=True)
class Medium:
    """
    Lossless medium (currently only epsilon_r, mu_r).
    """
    name: str
    epsilon_r: float = 1.0
    mu_r: float = 1.0

    @property
    def n(self) -> float:
        """Index/slowness factor n = sqrt(epsilon_r * mu_r)"""
        return float(np.sqrt(self.epsilon_r * self.mu_r))

    @property
    def v(self) -> float:
        """Phase velocity v = c0 / n"""
        return float(C0 / self.n)

    @property
    def eta(self) -> float:
        """Wave impedance eta = eta0 * sqrt(mu_r / epsilon_r)"""
        return float(ETA0 * np.sqrt(self.mu_r / self.epsilon_r))
