import math
from datetime import datetime, timedelta

import numpy as np
import pytest

from finpricing import BlackScholesModel, MarketData, ModelConfig, OptionParams
from finpricing.models.black_scholes.implied_volatility import (
    ImpliedVolatilityCalculator,
    ImpliedVolatilityParams,
    Method,
)
from finpricing.utils.bs_utils import BlackScholesCalculator, bs_price

S, K, T, R = 100.0, 100.0, 1.0, 0.05


def call(s=S, k=K, t=T, r=R, sigma=0.2) -> float:
    return BlackScholesCalculator.call_price(s, k, t, r, sigma)


def put(s=S, k=K, t=T, r=R, sigma=0.2) -> float:
    return BlackScholesCalculator.put_price(s, k, t, r, sigma)


class TestCallPutParity:
    """C - P = S - K * exp(-rT)"""

    @pytest.mark.parametrize("s,k", [(100, 100), (120, 100), (80, 100), (100, 80)])
    def test_parity(self, s, k):
        c = call(s=s, k=k)
        p = put(s=s, k=k)
        expected = s - k * math.exp(-R * T)
        assert abs(c - p - expected) < 1e-10

    @pytest.mark.parametrize("sigma", [0.1, 0.2, 0.5])
    def test_parity_across_vols(self, sigma):
        c = call(sigma=sigma)
        p = put(sigma=sigma)
        expected = S - K * math.exp(-R * T)
        assert abs(c - p - expected) < 1e-10

    def test_parity_short_expiry(self):
        c = call(t=0.01)
        p = put(t=0.01)
        expected = S - K * math.exp(-R * 0.01)
        assert abs(c - p - expected) < 1e-10


class TestIVRoundTrip:
    """price → IV → price recovers original price."""

    @pytest.mark.parametrize("sigma", [0.1, 0.2, 0.4, 0.6])
    def test_call_round_trip(self, sigma):
        target = call(sigma=sigma)
        params = ImpliedVolatilityParams(call_value=target, s=S, k=K, t=T, r=R)
        iv = ImpliedVolatilityCalculator.implied_volatility(params)
        recovered = call(sigma=iv)
        assert abs(recovered - target) < 1e-6

    @pytest.mark.parametrize("method", [Method.BRENTQ, Method.NEWTON, Method.ITERATIVE])
    def test_methods_agree(self, method):
        target = call(sigma=0.25)
        params = ImpliedVolatilityParams(call_value=target, s=S, k=K, t=T, r=R, method=method)
        iv = ImpliedVolatilityCalculator.implied_volatility(params)
        assert abs(iv - 0.25) < 1e-5

    def test_otm_call(self):
        target = call(s=90.0, k=110.0, sigma=0.3)
        params = ImpliedVolatilityParams(call_value=target, s=90.0, k=110.0, t=T, r=R)
        iv = ImpliedVolatilityCalculator.implied_volatility(params)
        recovered = call(s=90.0, k=110.0, sigma=iv)
        assert abs(recovered - target) < 1e-6


class TestGreeks:
    """Finite-difference verification of first-order Greeks."""

    H = 1e-4
    TOL = 1e-4

    def test_delta_call(self):
        _, d2 = BlackScholesCalculator.calculate_d1_d2(S, K, T, R, 0.2)
        from scipy.stats import norm
        d1, _ = BlackScholesCalculator.calculate_d1_d2(S, K, T, R, 0.2)
        analytic_delta = norm.cdf(d1)
        fd_delta = (call(s=S + self.H) - call(s=S - self.H)) / (2 * self.H)
        assert abs(analytic_delta - fd_delta) < self.TOL

    def test_delta_put(self):
        from scipy.stats import norm
        d1, _ = BlackScholesCalculator.calculate_d1_d2(S, K, T, R, 0.2)
        analytic_delta = norm.cdf(d1) - 1
        fd_delta = (put(s=S + self.H) - put(s=S - self.H)) / (2 * self.H)
        assert abs(analytic_delta - fd_delta) < self.TOL

    def test_vega(self):
        analytic_vega = BlackScholesCalculator.vega(S, K, T, R, 0.2)
        fd_vega = (call(sigma=0.2 + self.H) - call(sigma=0.2 - self.H)) / (2 * self.H)
        assert abs(analytic_vega - fd_vega) < self.TOL

    def test_call_put_gamma_equal(self):
        h = 0.5
        gamma_call = (call(s=S + h) - 2 * call() + call(s=S - h)) / h ** 2
        gamma_put = (put(s=S + h) - 2 * put() + put(s=S - h)) / h ** 2
        assert abs(gamma_call - gamma_put) < 1e-4


class TestVectorisedTails:
    """Deep out-of-the-money prices must keep their relative precision."""

    @pytest.mark.parametrize("K", [75.0, 70.0, 65.0])
    def test_deep_otm_put_matches_direct_formula(self, K):
        from scipy.stats import norm

        from finpricing.utils.bs_utils import bs_price
        S, T, r, q, sigma = 100.0, 0.05, 0.02, 0.01, 0.25
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        direct = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
        price = bs_price(S, K, T, r, sigma, q, option_type='put')
        assert price == pytest.approx(direct, rel=1e-9, abs=0)

    def test_deep_otm_put_implied_vol_round_trip(self):
        from finpricing.utils.bs_utils import bs_price, implied_volatility
        K = np.array([60.0, 65.0, 70.0])
        prices = bs_price(100.0, K, 0.05, 0.02, 0.25, option_type='put')
        iv = implied_volatility(prices, 100.0, K, 0.05, 0.02, option_type='put')
        np.testing.assert_allclose(iv, 0.25, atol=1e-7)


class TestBlackScholesModel:
    """The object interface must agree with bs_price, dividends and puts included."""

    NOW = datetime(2026, 1, 1)
    SIGMA = 0.2

    def model(self, sigma=SIGMA):
        return BlackScholesModel(ModelConfig(model_type="bs", numerical_params={"volatility": sigma}))

    def market(self, spot=100.0, r=0.03, q=0.0, now=NOW):
        return MarketData(spot_price=spot, risk_free_rate=r, dividend_yield=q, timestamp=now)

    def option(self, kind, days=365.25, strike=95.0):
        return OptionParams(symbol="X", expiry=self.NOW + timedelta(days=days),
                            strike_price=strike, option_type=kind)

    @pytest.mark.parametrize("kind", ["call", "put"])
    @pytest.mark.parametrize("q", [0.0, 0.05])
    def test_price_matches_bs_price(self, kind, q):
        price = self.model().price(self.market(q=q), self.option(kind))
        expected = float(bs_price(100.0, 95.0, 1.0, 0.03, self.SIGMA, q=q, option_type=kind))
        assert price == pytest.approx(expected, rel=1e-12)

    @pytest.mark.parametrize("kind", ["call", "put"])
    @pytest.mark.parametrize("q", [0.0, 0.05])
    @pytest.mark.parametrize("days", [365.25, 0.5])
    def test_greeks_match_finite_differences(self, kind, q, days):
        model, opt = self.model(), self.option(kind, days=days)
        greeks = model.greeks(self.market(q=q), opt)
        T = days / 365.25

        def px(spot=100.0, r=0.03, sigma=self.SIGMA, now=self.NOW):
            return self.model(sigma).price(self.market(spot=spot, r=r, q=q, now=now), opt)

        hs, hr, hv = 1e-3, 1e-5, 1e-5
        ht = timedelta(seconds=T * 365.25 * 86400 * 1e-4)          # 0.01% of the option's life
        dt = ht.total_seconds() / (365.25 * 86400)
        fd = {
            "delta": (px(spot=100 + hs) - px(spot=100 - hs)) / (2 * hs),
            "gamma": (px(spot=100 + hs) - 2 * px() + px(spot=100 - hs)) / hs**2,
            "vega": (px(sigma=self.SIGMA + hv) - px(sigma=self.SIGMA - hv)) / (2 * hv),
            "rho": (px(r=0.03 + hr) - px(r=0.03 - hr)) / (2 * hr),
            # theta = dV/dt in calendar time: moving "now" forward shortens the option
            "theta": (px(now=self.NOW + ht) - px(now=self.NOW - ht)) / (2 * dt),
        }
        for name, value in fd.items():
            assert greeks[name] == pytest.approx(value, rel=1e-4, abs=1e-6), name


class TestInitialGuesses:
    """Corrado-Miller (1996) is accurate to a few percent within half a standard deviation of the forward."""

    @pytest.mark.parametrize("z", [-0.5, 0.0, 0.5])          # strike = forward * exp(z sigma sqrt(T))
    @pytest.mark.parametrize("t", [0.25, 1.0, 2.0])
    @pytest.mark.parametrize("sigma", [0.1, 0.3, 0.5])
    def test_corrado_miller_near_the_money(self, z, t, sigma):
        k = 100.0 * math.exp(R * t + z * sigma * math.sqrt(t))
        params = ImpliedVolatilityParams(call_value=call(k=k, t=t, sigma=sigma), s=100.0, k=k, t=t, r=R)
        assert ImpliedVolatilityCalculator.corrado_miller_guess(params) == pytest.approx(sigma, rel=0.05)
