"""
Heston characteristic function in the "little Heston trap" form
(Albrecher, Mayer, Schoutens & Tistaert, 2007), which avoids the branch-cut
discontinuity of the original formulation for long maturities.
"""
import numpy as np

from finpricing.parameters import HestonParameters


def _log1p(x: np.ndarray) -> np.ndarray:
    """ln(1 + x) for complex x, accurate for small |x| (numpy's complex log1p is not)."""
    re, im = x.real, x.imag
    return 0.5 * np.log1p(re * (2 + re) + im * im) + 1j * np.arctan2(im, 1 + re)


def heston_cf(u: np.ndarray | complex, T: float, params: HestonParameters) -> np.ndarray:
    """
    Characteristic function E[exp(iu X_T)] of X_T = ln(S_T / S_0) - (r - q) T.

    X_T is the martingale part of the log-price, so heston_cf(-1j, T, p) == 1.
    ``u`` may be complex (Fourier pricers evaluate it off the real axis).

    beta - d and beta + d multiply to -xi^2 (iu + u^2), so the smaller of the two is
    computed from that product rather than by subtraction. This keeps full precision
    as xi -> 0, where beta - d = O(xi^2) is later divided by xi^2.
    """
    u = np.asarray(u, dtype=complex)
    kappa, theta, xi, rho, v0 = params.kappa, params.theta, params.xi, params.rho, params.v0

    iu = 1j * u
    w = iu + u ** 2
    beta = kappa - rho * xi * iu
    d = np.sqrt(beta ** 2 + xi ** 2 * w)
    bpd = beta + d
    flip = np.real(beta * np.conj(d)) < 0                      # |beta + d| < |beta - d|
    if np.any(flip):
        with np.errstate(divide='ignore', invalid='ignore'):
            bpd = np.where(flip, -xi ** 2 * w / (beta - d), bpd)
            bmd_xi2 = np.where(w == 0, 0.0, -w / bpd)        # u = 0 or u = -i: phi = 1
    else:
        bmd_xi2 = -w / bpd                                    # (beta - d) / xi^2

    # g = (beta - d) / (beta + d) and 1 - g = 2d / (beta + d). With e = exp(-dT):
    # (1 - g e) / (1 - g) = 1 + x,  x = g (1 - e) / (1 - g) = (beta - d)(1 - e) / (2d).
    one_minus_e = -np.expm1(-d * T)
    h = one_minus_e / (2 * d)
    x = xi ** 2 * bmd_xi2 * h

    C = kappa * theta * (bmd_xi2 * T - 2 * _log1p(x) / xi ** 2)
    D = -w * h / (1 + x)                                       # (beta - d)(1 - e) / (xi^2 (1 - g e))
    return np.exp(C + D * v0)
