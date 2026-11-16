"""
Monte Carlo simulation of the Heston model with Andersen's (2008)
Quadratic-Exponential scheme and martingale correction (QE-M).
"""
import numpy as np

from finpricing.parameters import HestonParameters

_PSI_CRITICAL = 1.5


def _qe_variance_step(v: np.ndarray, dt: float, p: HestonParameters, z: np.ndarray,
                      uniform: np.ndarray, A: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """
    One QE step of the variance.

    Returns ``(v_next, log_mgf)``, where ``log_mgf = ln E[exp(A v_next) | v]`` under the
    sampled (quadratic or exponential) distribution, NaN where that expectation is
    infinite: ``A >= 1/(2a)`` in the quadratic branch, ``A >= beta`` in the exponential one.
    """
    ekt = np.exp(-p.kappa * dt)
    m = p.theta + (v - p.theta) * ekt
    s2 = (v * p.xi ** 2 * ekt * (1 - ekt) / p.kappa
          + p.theta * p.xi ** 2 * (1 - ekt) ** 2 / (2 * p.kappa))
    psi = s2 / m ** 2

    v_next = np.empty_like(v)
    log_mgf = np.full_like(v, np.nan)

    quad = psi <= _PSI_CRITICAL           # moment-matched squared Gaussian
    inv = 2 / psi[quad]
    b2 = inv - 1 + np.sqrt(inv) * np.sqrt(inv - 1)
    a = m[quad] / (1 + b2)
    v_next[quad] = a * (np.sqrt(b2) + z[quad]) ** 2
    one_minus = 1 - 2 * A * a
    ok = one_minus > 0
    log_mgf[np.flatnonzero(quad)[ok]] = (A * b2[ok] * a[ok] / one_minus[ok]
                                         - 0.5 * np.log(one_minus[ok]))

    expo = ~quad                          # point mass at 0 + exponential tail
    pr = (psi[expo] - 1) / (psi[expo] + 1)
    beta = (1 - pr) / m[expo]
    u = uniform[expo]
    v_next[expo] = np.where(u <= pr, 0.0, np.log((1 - pr) / np.maximum(1 - u, 1e-300)) / beta)
    ok = beta > A
    log_mgf[np.flatnonzero(expo)[ok]] = np.log(pr[ok] + beta[ok] * (1 - pr[ok]) / (beta[ok] - A))
    return v_next, log_mgf


def simulate_heston(
    S0: float,
    T: float,
    r: float,
    params: HestonParameters,
    q: float = 0.0,
    n_steps: int = 100,
    n_paths: int = 10_000,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate Heston paths with the QE-M scheme.

    Args:
        S0: Spot price.
        T: Horizon in years.
        r: Risk-free rate.
        params: Heston parameters.
        q: Dividend yield.
        n_steps: Number of time steps.
        n_paths: Number of paths.
        seed: Random seed.

    Returns:
        (times, S, v): times has shape (n_steps + 1,); S and v have shape
        (n_paths, n_steps + 1).
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    p = params
    gamma1 = gamma2 = 0.5
    k0 = -p.rho * p.kappa * p.theta * dt / p.xi
    k1 = gamma1 * dt * (p.kappa * p.rho / p.xi - 0.5) - p.rho / p.xi
    k2 = gamma2 * dt * (p.kappa * p.rho / p.xi - 0.5) + p.rho / p.xi
    k3 = gamma1 * dt * (1 - p.rho ** 2)
    k4 = gamma2 * dt * (1 - p.rho ** 2)
    A = k2 + 0.5 * k4

    log_s = np.empty((n_paths, n_steps + 1))
    v = np.empty((n_paths, n_steps + 1))
    log_s[:, 0] = np.log(S0)
    v[:, 0] = p.v0

    for i in range(n_steps):
        z_v, z_s = rng.standard_normal((2, n_paths))
        uniform = rng.random(n_paths)
        v[:, i + 1], log_mgf = _qe_variance_step(v[:, i], dt, p, z_v, uniform, A)
        # Martingale correction: K0* makes E[S_{t+dt} | S_t, v_t] = S_t exp((r - q) dt)
        # exactly. Where E[exp(A v_{t+dt})] is infinite, keep the uncorrected K0.
        k0_star = np.where(np.isnan(log_mgf), k0, -log_mgf - (k1 + 0.5 * k3) * v[:, i])
        log_s[:, i + 1] = (
            log_s[:, i] + (r - q) * dt + k0_star + k1 * v[:, i] + k2 * v[:, i + 1]
            + np.sqrt(np.maximum(k3 * v[:, i] + k4 * v[:, i + 1], 0.0)) * z_s
        )

    return np.linspace(0.0, T, n_steps + 1), np.exp(log_s), v
