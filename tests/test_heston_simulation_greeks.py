"""Heston QE martingale correction and Greeks in the Black-Scholes limit (#20)."""
from datetime import datetime, timedelta

import numpy as np
import pytest
from scipy import integrate
from scipy.stats import norm

from finpricing import HestonModel, HestonParameters, MarketData, OptionParams
from finpricing.models.heston.simulation import _qe_variance_step

NOW = datetime(2026, 1, 1)
SIGMA = 0.2
BS_LIMIT = HestonModel(HestonParameters(v0=SIGMA**2, theta=SIGMA**2, kappa=1.5, xi=1e-4, rho=0.0))


def test_forward_unbiased_with_coarse_steps():
    model = HestonModel(HestonParameters(v0=0.09, kappa=1.0, theta=0.06, xi=1.5, rho=-0.9))
    _, S, _ = model.simulate(S0=100, T=5.0, r=0.0, n_steps=4, n_paths=200_000, seed=0)
    ST = S[:, -1]
    z = (ST.mean() - 100.0) / (ST.std() / np.sqrt(len(ST)))
    assert abs(z) < 3, f"E[S_T] = {ST.mean():.3f}, z = {z:.1f}"


@pytest.mark.parametrize("v, dt", [(0.2, 0.05), (0.001, 0.05), (0.04, 1.25)])
@pytest.mark.parametrize("A", [-2.0, 0.5])
def test_qe_log_mgf_matches_quadrature(v, dt, A):
    """The QE-M correction uses ln E[exp(A v_next)] in closed form; check it by quadrature
    against the sampled distribution, in both the quadratic and the exponential branch."""
    p = HestonParameters(v0=0.04, kappa=0.5, theta=0.04, xi=1.0, rho=0.5)
    vv = np.array([v])

    def v_next(z, u):
        return _qe_variance_step(vv, dt, p, np.array([z]), np.array([u]), A)[0][0]

    _, log_mgf = _qe_variance_step(vv, dt, p, np.zeros(1), np.full(1, 0.5), A)
    if v_next(0.3, 0.5) != v_next(-0.3, 0.5):       # quadratic branch: v_next depends on z
        numeric = integrate.quad(lambda z: np.exp(A * v_next(z, 0.5)) * norm.pdf(z), -40, 40,
                                 limit=400)[0]
    else:                                            # exponential branch: depends on u
        numeric = integrate.quad(lambda u: np.exp(A * v_next(0.0, u)), 0, 1, limit=400)[0]
    assert np.exp(log_mgf[0]) == pytest.approx(numeric, rel=1e-7)


def test_qe_log_mgf_nan_where_infinite():
    p = HestonParameters(v0=0.04, kappa=0.5, theta=0.04, xi=1.0, rho=0.5)
    _, log_mgf = _qe_variance_step(np.array([0.04]), 1.25, p, np.zeros(1), np.full(1, 0.5), 3.0)
    assert np.isnan(log_mgf[0])


def _bs_greeks(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return {
        "delta": norm.cdf(d1),
        "gamma": norm.pdf(d1) / (S * sigma * np.sqrt(T)),
        "vega": S * norm.pdf(d1) * np.sqrt(T),
        "theta": -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2),
        "rho": K * T * np.exp(-r * T) * norm.cdf(d2),
    }


def _greeks(days, strike=100.0, r=0.03):
    market = MarketData(spot_price=100.0, risk_free_rate=r, timestamp=NOW)
    option = OptionParams(symbol="X", expiry=NOW + timedelta(days=days), strike_price=strike,
                          option_type="call")
    T = days / 365.25
    return BS_LIMIT.greeks(market, option), _bs_greeks(100.0, strike, T, r, SIGMA)


@pytest.mark.parametrize("name", ["delta", "gamma", "rho"])
@pytest.mark.parametrize("days", [365.25, 30.0])
def test_greeks_in_black_scholes_limit(name, days):
    heston, bs = _greeks(days)
    assert heston[name] == pytest.approx(bs[name], rel=1e-3)


@pytest.mark.parametrize("days", [365.25, 30.0])
def test_vega_is_with_respect_to_instantaneous_volatility(days):
    """Heston vega bumps sqrt(v0) with theta fixed. With v0 = theta the effective variance is
    theta + (v0 - theta)(1 - exp(-kappa T)) / (kappa T), so vega = BS vega * (1 - e^{-kT}) / (kT)."""
    heston, bs = _greeks(days)
    kT = BS_LIMIT.params.kappa * days / 365.25
    assert heston["vega"] == pytest.approx(bs["vega"] * (1 - np.exp(-kT)) / kT, rel=1e-3)


@pytest.mark.parametrize("days", [365.25, 30.0])
def test_theta_in_black_scholes_limit(days):
    heston, bs = _greeks(days)
    assert heston["theta"] == pytest.approx(bs["theta"], rel=1e-3)


def test_theta_one_day_to_expiry():
    heston, bs = _greeks(1.0)
    assert heston["theta"] == pytest.approx(bs["theta"], rel=5e-3)
