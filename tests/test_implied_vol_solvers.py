"""Implied-volatility solvers: one failure mode, puts, dividends, deep ITM, broadcasting (#21)."""
import numpy as np
import pytest

from finpricing import ImpliedVolatilityCalculator, ImpliedVolatilityParams, Method
from finpricing.utils.bs_utils import BlackScholesCalculator, bs_price, implied_volatility

METHODS = list(Method)


def _solve(price, method, **kw):
    params = ImpliedVolatilityParams(call_value=price, method=method, **kw)
    return ImpliedVolatilityCalculator.implied_volatility(params)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("price", [120.0, 0.0])          # above S, and below intrinsic (S - K e^{-rT} > 0)
def test_out_of_bounds_quote_raises_value_error(method, price):
    with pytest.raises(ValueError, match="no-arbitrage"):
        _solve(price, method, s=100.0, k=90.0, t=1.0, r=0.03)


@pytest.mark.xfail(strict=True, reason="#21: deep in-the-money calls fail or return nonsense")
@pytest.mark.parametrize("method", METHODS)
def test_deep_in_the_money_call_round_trip(method):
    price = float(bs_price(100.0, 60.0, 0.1, 0.05, 0.2))
    assert _solve(price, method, s=100.0, k=60.0, t=0.1, r=0.05) == pytest.approx(0.2, abs=1e-6)


@pytest.mark.parametrize("method", METHODS)
def test_dividend_yield_round_trip(method):
    price = float(bs_price(100.0, 105.0, 0.5, 0.03, 0.25, q=0.04))
    assert _solve(price, method, s=100.0, k=105.0, t=0.5, r=0.03, q=0.04) == pytest.approx(0.25, abs=1e-6)


@pytest.mark.parametrize("method", METHODS)
def test_put_round_trip(method):
    price = float(bs_price(100.0, 90.0, 0.5, 0.03, 0.3, option_type="put"))
    iv = _solve(price, method, s=100.0, k=90.0, t=0.5, r=0.03, option_type="put")
    assert iv == pytest.approx(0.3, abs=1e-6)


@pytest.mark.parametrize("option_type", ["Put", "p", "CALL"])
def test_unknown_option_type_raises(option_type):
    with pytest.raises(ValueError):
        bs_price(100.0, 100.0, 1.0, 0.0, 0.2, option_type=option_type)
    with pytest.raises(ValueError):
        implied_volatility(8.0, 100.0, 100.0, 1.0, 0.0, option_type=option_type)


def test_rates_and_dividends_broadcast():
    r, q = np.array([0.0, 0.03, 0.05]), np.array([0.0, 0.01, 0.02])
    prices = bs_price(100.0, 100.0, 1.0, r, 0.2, q=q)
    np.testing.assert_allclose(implied_volatility(prices, 100.0, 100.0, 1.0, r, q), 0.2, atol=1e-8)


def test_scalar_calculator_accepts_arrays():
    k = np.array([90.0, 100.0, 110.0])
    np.testing.assert_allclose(BlackScholesCalculator.call_price(100.0, k, 1.0, 0.03, 0.2),
                               bs_price(100.0, k, 1.0, 0.03, 0.2), rtol=1e-12)


def test_high_volatility_round_trip():
    price = float(bs_price(100.0, 100.0, 0.01, 0.0, 6.0))
    assert implied_volatility(price, 100.0, 100.0, 0.01, 0.0) == pytest.approx(6.0, abs=1e-6)
