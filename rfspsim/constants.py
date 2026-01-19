# src/rfspsim/constants.py
import numpy as np

# 更精确的真空光速
C0 = 3e8  # m/s

# 真空介电常数、磁导率
EPS0 = 8.8541878128e-12  # 介电常数（F/m）
MU0 = 4.0 * np.pi * 1e-7  # 磁导率（H/m）

# 真空波阻抗
ETA0 = float(np.sqrt(MU0 / EPS0))  # ~ 376.73 ohm