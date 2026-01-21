# src/rfspsim/constants.py
import numpy as np

# Vacuum speed of light (more precise value)
C0 = 3e8  # m/s

# Vacuum permittivity and permeability
EPS0 = 8.8541878128e-12  # Permittivity (F/m)
MU0 = 4.0 * np.pi * 1e-7  # Permeability (H/m)

# Vacuum wave impedance
ETA0 = float(np.sqrt(MU0 / EPS0))  # ~ 376.73 ohm
