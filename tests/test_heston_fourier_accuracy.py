"""Heston Fourier pricing at short maturities and extreme parameters (#14, #18)."""
import numpy as np
import pytest
from scipy import integrate

from finpricing import HestonParameters
from finpricing.models.heston import heston_cf, heston_price
from finpricing.utils.bs_utils import bs_price

BENCH = HestonParameters(v0=0.0175, kappa=1.5768, theta=0.0398, xi=0.5751, rho=-0.5711)
BASE = HestonParameters(v0=0.04, kappa=1.5, theta=0.05, xi=0.6, rho=-0.7)
S0, R, Q = 100.0, 0.03, 0.01


def lewis_by_quad(S, K, T, r, q, p):
    """Lewis (2001) call price with adaptive quadrature on [0, inf): independent of the
    fixed Gauss-Legendre grid and integration cutoff that heston_price uses."""
    log_fk = np.log(S / K) + (r - q) * T

    def integrand(u):
        return np.real(np.exp(1j * u * log_fk) * heston_cf(np.array([u - 0.5j]), T, p)[0]) / (u ** 2 + 0.25)

    integral, _ = integrate.quad(integrand, 0, np.inf, limit=5000, epsabs=1e-13, epsrel=1e-12)
    return S * np.exp(-q * T) - np.sqrt(S * K) * np.exp(-0.5 * (r + q) * T) * integral / np.pi


# ── #14: FFT pricers at short maturities ─────────────────────────────────────────────
@pytest.mark.xfail(strict=True, reason="#14: fixed FFT grid cannot resolve short-dated smiles")
@pytest.mark.parametrize("method", ["fft", "fft-ct"])
@pytest.mark.parametrize("T", [1 / 365, 1 / 52])
def test_fft_matches_lewis_at_short_maturities(method, T):
    K = S0 * np.array([0.98, 0.99, 1.0, 1.01, 1.02])
    lewis = heston_price(S0, K, T, R, BENCH, q=Q)
    fft = heston_price(S0, K, T, R, BENCH, q=Q, method=method)
    np.testing.assert_allclose(fft, lewis, rtol=1e-3, atol=1e-5)


# ── #18: Lewis integral at extreme parameters ────────────────────────────────────────
CORNER = HestonParameters(v0=1e-4, kappa=0.5, theta=1e-4, xi=3.0, rho=-0.99)   # calibrator bound corner


@pytest.mark.xfail(strict=True, reason="#18: Lewis integral truncated at u = 1e4")
def test_lewis_not_truncated_at_bound_corner():
    K = np.array([99.0, 100.0, 101.0])
    reference = np.array([lewis_by_quad(S0, k, 1 / 12, 0.0, 0.0, CORNER) for k in K])
    np.testing.assert_allclose(heston_price(S0, K, 1 / 12, 0.0, CORNER), reference, atol=1e-6)


@pytest.mark.xfail(strict=True, reason="#18: characteristic function loses precision as xi -> 0")
@pytest.mark.parametrize("xi", [1e-6, 1e-7, 1e-8])
def test_black_scholes_limit_for_tiny_xi(xi):
    p = HestonParameters(v0=0.04, theta=0.04, kappa=1.5, xi=xi, rho=0.0)
    price = heston_price(S0, 100.0, 1.0, 0.0, p)
    assert price == pytest.approx(float(bs_price(S0, 100.0, 1.0, 0.0, 0.2)), abs=1e-8)


@pytest.mark.xfail(strict=True, reason="#18: short-dated far-OTM Lewis prices can be about -1e-9")
def test_short_dated_far_otm_prices_are_not_negative():
    T = 1 / 12
    forward = S0 * np.exp((R - Q) * T)
    K = forward * np.exp(np.linspace(0.2, 0.4, 41))
    assert np.all(heston_price(S0, K, T, R, BASE, q=Q) >= 0)


@pytest.mark.xfail(strict=True, reason="#18: invalid spot and strikes return NaN instead of raising")
@pytest.mark.parametrize("spot, strike", [(0.0, 100.0), (-1.0, 100.0), (100.0, np.nan)])
def test_invalid_spot_or_strike_raises(spot, strike):
    with pytest.raises(ValueError):
        heston_price(spot, strike, 1.0, R, BASE)


@pytest.mark.xfail(strict=True, reason="#18: unknown keyword arguments are silently ignored")
@pytest.mark.parametrize("method, kwargs", [("lewis", {"N": 4096}), ("fft", {"n": 4096})])
def test_unknown_keyword_arguments_raise(method, kwargs):
    with pytest.raises(TypeError):
        heston_price(S0, 100.0, 1.0, R, BASE, method=method, **kwargs)
