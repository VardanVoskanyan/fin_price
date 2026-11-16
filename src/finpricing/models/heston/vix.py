"""
VIX under Heston.

The VIX is the 30-day expected realised variance, which under Heston is affine in v_t:

    VIX_t^2 = a v_t + b,   a = (1 - e^{-kappa tau}) / (kappa tau),   b = theta (1 - a),

with tau = 30/365. VIX levels are in decimal volatility units (0.20, not 20).
v_T given v_0 is a scaled non-central chi-squared variable, so VIX futures and
options reduce to one-dimensional integrals.
"""
import numpy as np
from scipy import integrate, stats

from finpricing.parameters import HestonParameters

VIX_TAU = 30 / 365

# Probability mass left out at each end of the variance law.
_TAIL = 1e-12
# Above this non-centrality v_T is treated as normal. The skewness of the non-central chi^2
# is about 3 / sqrt(nc), so the error is far below quadrature tolerance, and scipy's ncx2
# loses its quantiles and density somewhere above 1e10.
_NORMAL_NC = 1e8


def vix_coefficients(params: HestonParameters, tau: float = VIX_TAU) -> tuple[float, float]:
    """(a, b) such that VIX_t^2 = a v_t + b."""
    a = (1 - np.exp(-params.kappa * tau)) / (params.kappa * tau)
    return a, params.theta * (1 - a)


def vix_spot(params: HestonParameters) -> float:
    """Model VIX today, sqrt(a v0 + b)."""
    a, b = vix_coefficients(params)
    return float(np.sqrt(a * params.v0 + b))


def _variance_law(T: float, p: HestonParameters):
    """v_T = c * X with X ~ non-central chi^2(df, nc)."""
    c = p.xi ** 2 * -np.expm1(-p.kappa * T) / (4 * p.kappa)
    df = 4 * p.kappa * p.theta / p.xi ** 2
    nc = p.v0 * np.exp(-p.kappa * T) / c
    return c, stats.ncx2(df, nc)


def _variance_moments(T: float, p: HestonParameters) -> tuple[float, float]:
    """Mean and standard deviation of v_T, computed directly so they stay finite as T -> 0."""
    e = np.exp(-p.kappa * T)
    one_minus_e = -np.expm1(-p.kappa * T)
    mean = p.v0 * e + p.theta * one_minus_e
    var = p.xi ** 2 * one_minus_e * (p.v0 * e + p.theta * one_minus_e / 2) / p.kappa
    return mean, np.sqrt(var)


def _quad(f, lo: float, hi: float, mid: float) -> float:
    points = [mid] if lo < mid < hi else None
    value, _ = integrate.quad(f, lo, hi, points=points, limit=500, epsabs=1e-12, epsrel=1e-10)
    return value


def _expect(payoff, T: float, p: HestonParameters, lower_vix: float = 0.0) -> float:
    """
    E[payoff(VIX_T)] over the law of v_T, restricted to VIX_T >= lower_vix.

    The integral runs over the bulk of the law, between its 1e-12 and 1 - 1e-12 quantiles,
    with a breakpoint at the mean: for large non-centrality (small xi or short T) the
    density is a narrow spike that quad would miss over [0, inf), and scipy's density is
    NaN far out in its left tail. Very large non-centrality uses a normal law for v_T, and
    a degenerate law falls back to the deterministic VIX.
    """
    a, b = vix_coefficients(p)

    def vix(v):
        return np.sqrt(max(a * v + b, 0.0))

    v_lo = (lower_vix ** 2 - b) / a
    mean, sd = _variance_moments(T, p)
    if not sd > 0:
        return payoff(vix(mean)) if mean >= v_lo else 0.0

    c, law = _variance_law(T, p)
    _, nc = law.args
    if nc <= _NORMAL_NC:
        x_lo = max(v_lo / c, law.ppf(_TAIL), 0.0)
        x_hi = law.ppf(1 - _TAIL)
        if np.isfinite(x_lo) and np.isfinite(x_hi):
            if x_lo >= x_hi:
                return 0.0

            def integrand(x):
                density = law.pdf(x)
                return payoff(vix(c * x)) * density if np.isfinite(density) else 0.0

            return _quad(integrand, x_lo, x_hi, law.mean())

    # Normal approximation, in standard units z = (v - mean) / sd.
    z_hi = stats.norm.isf(_TAIL)
    z_lo = max((v_lo - mean) / sd, -z_hi)
    if z_lo >= z_hi:
        return 0.0
    return _quad(lambda z: payoff(vix(mean + sd * z)) * stats.norm.pdf(z), z_lo, z_hi, 0.0)


def vix_future(T: float, params: HestonParameters) -> float:
    """VIX futures price E[VIX_T] (decimal units)."""
    if T < 0:
        raise ValueError("Maturity must be non-negative.")
    if T == 0:
        return vix_spot(params)
    return _expect(lambda vix: vix, T, params)


def vix_option_price(
    K: float | np.ndarray,
    T: float,
    r: float,
    params: HestonParameters,
    option_type: str = 'call',
) -> float | np.ndarray:
    """
    European VIX option price, e^{-rT} E[(VIX_T - K)^+] for calls.
    Puts via parity with the VIX future: P = C - e^{-rT} (F - K).
    """
    if T <= 0:
        raise ValueError("Maturity must be positive.")
    strikes = np.atleast_1d(np.asarray(K, dtype=float))
    df = np.exp(-r * T)
    calls = np.array([df * _expect(lambda vix, k=k: vix - k, T, params, lower_vix=k)
                      for k in strikes])
    match option_type:
        case 'call':
            prices = calls
        case 'put':
            prices = calls - df * (vix_future(T, params) - strikes)
        case _:
            raise ValueError(f"Unknown option type: {option_type!r}")
    return float(prices[0]) if np.ndim(K) == 0 else prices
