"""
Hardcoded paper fixtures from Cont & Kokholm (2013).

Parameters sourced from Table 2 of:
  R. Cont, T. Kokholm — "A consistent pricing model for index options
  and volatility derivatives", Mathematical Finance 23(2), 2013.
"""
import numpy as np
from finpricing.parameters import VIXMertonModelParameters, VIXKouModelParameters


# ---------------------------------------------------------------------------
# Tenor structure used in the paper
# ---------------------------------------------------------------------------

TENOR_DATES: np.ndarray = np.array([0, 1, 2, 3, 4, 7, 10]) / 12  # in years

# Initial forward variance swap rates V^i_0 (Table 1 / calibration target)
VI_0: np.ndarray = np.array([0.041, 0.052, 0.056, 0.059, 0.062, 0.061])

# ---------------------------------------------------------------------------
# Merton model — Gaussian jumps
# ---------------------------------------------------------------------------

MERTON_PARAMS = VIXMertonModelParameters(
    rho=-0.45,
    omega=2.04,
    lmbd=3.52,
    k1=21.9,
    k2=2.07,
    m=0.54,
    delta=0.25,
)

# b_i coefficients linking variance swap dynamics to index jump exposure
MERTON_B_I: np.ndarray = np.array([-0.140, -0.161, -0.162, -0.187, -0.198, -0.199])

# ---------------------------------------------------------------------------
# Kou model — double-exponential jumps
# ---------------------------------------------------------------------------

KOU_PARAMS = VIXKouModelParameters(
    rho=-0.45,
    omega=1.98,
    lmbd=13.6,
    k1=22.3,
    k2=2.20,
    p=0.86,
    alpha_plus=4.25,
    alpha_minus=19.9,
)

KOU_B_I: np.ndarray = np.array([-0.141, -0.159, -0.158, -0.187, -0.195, -0.192])
