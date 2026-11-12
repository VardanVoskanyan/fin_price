"""
Heston characteristic function in the "little Heston trap" form
(Albrecher, Mayer, Schoutens & Tistaert, 2007), which avoids the branch-cut
discontinuity of the original formulation for long maturities.
"""
import numpy as np

from finpricing.parameters import HestonParameters


def heston_cf(u: np.ndarray | complex, T: float, params: HestonParameters) -> np.ndarray:
    """
    Characteristic function E[exp(iu X_T)] of X_T = ln(S_T / S_0) - (r - q) T.

    X_T is the martingale part of the log-price, so heston_cf(-1j, T, p) == 1.
    ``u`` may be complex (Fourier pricers evaluate it off the real axis).
    """
    u = np.asarray(u, dtype=complex)
    kappa, theta, xi, rho, v0 = params.kappa, params.theta, params.xi, params.rho, params.v0

    iu = 1j * u
    beta = kappa - rho * xi * iu
    d = np.sqrt(beta ** 2 + xi ** 2 * (iu + u ** 2))
    g = (beta - d) / (beta + d)
    edt = np.exp(-d * T)

    C = kappa * theta / xi ** 2 * ((beta - d) * T - 2 * np.log((1 - g * edt) / (1 - g)))
    D = (beta - d) / xi ** 2 * (1 - edt) / (1 - g * edt)
    return np.exp(C + D * v0)
