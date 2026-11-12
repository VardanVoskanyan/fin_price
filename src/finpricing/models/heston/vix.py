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
    c = p.xi ** 2 * (1 - np.exp(-p.kappa * T)) / (4 * p.kappa)
    df = 4 * p.kappa * p.theta / p.xi ** 2
    nc = p.v0 * np.exp(-p.kappa * T) / c
    return c, stats.ncx2(df, nc)


def _expect(payoff, T: float, p: HestonParameters, lower_vix: float = 0.0) -> float:
    """E[payoff(VIX_T)] integrated over the non-central chi^2 law, from VIX_T >= lower_vix."""
    a, b = vix_coefficients(p)
    c, law = _variance_law(T, p)
    x_lo = max((lower_vix ** 2 - b) / (a * c), 0.0)
    x_hi = law.ppf(1 - 1e-12)
    if x_lo >= x_hi:
        return 0.0
    mid = min(max(law.mean(), x_lo), x_hi)
    points = [mid] if x_lo < mid < x_hi else None

    def integrand(x):
        return payoff(np.sqrt(a * c * x + b)) * law.pdf(x)

    value, _ = integrate.quad(integrand, x_lo, x_hi, points=points, limit=500,
                              epsabs=1e-12, epsrel=1e-10)
    return value


def vix_future(T: float, params: HestonParameters) -> float:
    """VIX futures price E[VIX_T] (decimal units)."""
    if T <= 0:
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
