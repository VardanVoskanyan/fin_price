"""
Fourier-Based Call Option Pricing
"""
from collections.abc import Callable
from enum import Enum

import numpy as np
from scipy.fft import fft
from scipy.stats import norm

from finpricing.utils.characteristic_functions import black_scholes_characteristic_function
from finpricing.utils.math_utils import compute_weights, generate_grid


class FourierMethod(Enum):
    CONT_TANKOV = "Cont&Tankov"
    KAR_MADAN = "Kar&Madan"


def cont_tankov_pricer(
        chr_func: Callable[[complex], complex],
        T: float,
        r: float,
        s_0: float,
        N: int,
        Delta: float,
        sigma: float,
        log_forward: float | None = None) -> np.ndarray:
    """
    Fourier pricing using Cont & Tankov method.

    Returns C(k) - C_BS(k): the transform is applied to the difference from a
    Black-Scholes price with volatility ``sigma`` and the same forward as the model.
    ``log_forward`` is ln E[S_T / S_0]; it defaults to r T (no dividends).
    """
    mu = r * T if log_forward is None else log_forward
    # Integration grid
    d = 2 * np.pi / (Delta * N)
    A = Delta * (N - 1)
    xs = generate_grid(-0.5 * A, Delta, N)
    us = generate_grid(-N * d / 2 + s_0, d, N)

    # Quadrature weights
    w = compute_weights(N)

    # Modified characteristic function
    zeta_T = np.exp(-r * T) * chr_func(xs - 1j)
    zeta_T -= np.exp(-r * T + 1j * (xs - 1j) * mu) * black_scholes_characteristic_function(xs - 1j, T, sigma)
    zeta_T /= (1j * xs - xs**2)

    # Inverse Fourier Transform
    zs = fft(zeta_T * w * np.exp(-1j * (s_0 - N * d / 2) * (xs + 0.5 * A)))
    zs = Delta * np.exp(1j * A * us / 2) * zs / (2 * np.pi)

    return zs.real


def kar_madan_pricer(chr_func: Callable[[complex], complex], T: float, r: float, s_0: float, N: int, Delta: float, alpha: float) -> np.ndarray:
    """Fourier pricing using Kar & Madan method."""
    d = 2 * np.pi / (N * Delta)  # output grid step (dual to integration step Delta)

    xs = generate_grid(0, Delta, N)               # integration frequencies
    us = generate_grid(-N * d / 2 + s_0, d, N)   # log-moneyness output grid

    # Modified characteristic function
    zeta_T = np.exp(-r * T) * chr_func(xs - (alpha + 1) * 1j)
    zeta_T /= alpha**2 + alpha - xs**2 + 1j * (2 * alpha + 1) * xs

    # Quadrature weights (Simpson's rule)
    w = compute_weights(N, method="simpson")

    # Inverse Fourier Transform
    zs = fft(zeta_T * np.exp((N * d / 2 - s_0) * 1j * xs) * w) * Delta
    return np.exp(-alpha * us) * zs.real / np.pi


def fourier_call_pricer(
    chr_func: Callable[[complex], complex],
    T: float,
    r: float,
    s_0: float = 0.0,
    method: FourierMethod = FourierMethod.CONT_TANKOV,
    **kwargs
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fourier-based call option pricer.

    ``chr_func`` is the characteristic function of ln(S_T / S_0), drift included.
    Returns (log-strikes, call prices) for S_0 = 1.
    """
    if method == FourierMethod.CONT_TANKOV:
        N = kwargs.get("N", 1024)
        Delta = kwargs.get("Delta", 0.17)
        sigma = kwargs.get("sigma", 0.3575)
        # The reference Black-Scholes price must share the model's forward,
        # E[S_T / S_0] = chr_func(-i), or the transform picks up a constant error.
        mu = float(np.log(np.real(chr_func(np.array([-1j]))[0])))
        zs = cont_tankov_pricer(chr_func, T, r, s_0, N, Delta, sigma, log_forward=mu)
        # Add back the reference Black-Scholes price.
        d = 2 * np.pi / (N * Delta)
        log_k = generate_grid(-N * d / 2 + s_0, d, N)
        d1 = (mu - log_k + sigma**2 * T / 2) / (sigma * np.sqrt(T))
        zs = zs + np.exp(-r * T) * (np.exp(mu) * norm.cdf(d1)
                                    - np.exp(log_k) * norm.cdf(d1 - sigma * np.sqrt(T)))
    
    elif method == FourierMethod.KAR_MADAN:
        N = kwargs.get("N", 1024)
        Delta = kwargs.get("Delta", 0.17)
        alpha = kwargs.get("alpha", 0.75)
        zs = kar_madan_pricer(chr_func, T, r, s_0, N, Delta, alpha)
    
    else:
        raise ValueError(f"Unsupported Fourier pricing method: {method}")

    d = 2 * np.pi / (N * Delta)
    us = generate_grid(-N * d / 2 + s_0, d, N)
    return us, zs
