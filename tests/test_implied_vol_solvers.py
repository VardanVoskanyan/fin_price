"""Implied-volatility solvers: one failure mode, puts, dividends, deep ITM, broadcasting (#21)."""
import numpy as np
import pytest

from finpricing import ImpliedVolatilityCalculator, ImpliedVolatilityParams, Method
from finpricing.utils.bs_utils import BlackScholesCalculator, bs_price, implied_volatility

METHODS = list(Method)


def _solve(price, method, **kw):
    params = ImpliedVolatilityParams(call_value=price, method=method, **kw)
    return ImpliedVolatilityCalculator.implied_volatility(params)


@pytest.mark.xfail(strict=True, reason="#21: out-of-bounds quotes fail in five different ways")
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("price", [120.0, 0.0])          # above S, and below intrinsic (S - K e^{-rT} > 0)
def test_out_of_bounds_quote_raises_value_error(method, price):
    with pytest.raises(ValueError, match="no-arbitrage"):
        _solve(price, method, s=100.0, k=90.0, t=1.0, r=0.03)


@pytest.mark.xfail(strict=True, reason="#21: a quote at the lower bound has no positive volatility")
@pytest.mark.parametrize("method", METHODS)
def test_quote_at_the_lower_bound_raises_value_error(method):
    # At K=60, T=0.1 the time value (~1e-16) is below one ulp of the price, so the
    # sigma=0.2 call price rounds to exactly S - K e^{-rT}.
    price = float(bs_price(100.0, 60.0, 0.1, 0.05, 0.2))
    with pytest.raises(ValueError, match="no-arbitrage"):
        _solve(price, method, s=100.0, k=60.0, t=0.1, r=0.05)


# Deep ITM quotes whose time value is still resolvable in float64; marked per method
# because some solvers already recover them.
_DEEP_ITM_FAILING = {
    (70.0, 0.1): {Method.LM, Method.ITERATIVE},
    (60.0, 0.25): {Method.ITERATIVE},
}
_DEEP_ITM_CASES = [
    pytest.param(
        k, t, method,
        marks=[pytest.mark.xfail(strict=True, reason="#21: deep in-the-money calls return nonsense")]
        if method in failing else [],
        id=f"K{k:g}-T{t:g}-{method.name}",
    )
    for (k, t), failing in _DEEP_ITM_FAILING.items()
    for method in METHODS
]


@pytest.mark.parametrize(("k", "t", "method"), _DEEP_ITM_CASES)
def test_deep_in_the_money_call_round_trip(k, t, method):
    price = float(bs_price(100.0, k, t, 0.05, 0.2))
    assert _solve(price, method, s=100.0, k=k, t=t, r=0.05) == pytest.approx(0.2, abs=1e-6)


@pytest.mark.xfail(strict=True, reason="#21: no dividend yield")
@pytest.mark.parametrize("method", METHODS)
def test_dividend_yield_round_trip(method):
    price = float(bs_price(100.0, 105.0, 0.5, 0.03, 0.25, q=0.04))
    assert _solve(price, method, s=100.0, k=105.0, t=0.5, r=0.03, q=0.04) == pytest.approx(0.25, abs=1e-6)


@pytest.mark.xfail(strict=True, reason="#21: calls only")
@pytest.mark.parametrize("method", METHODS)
def test_put_round_trip(method):
    price = float(bs_price(100.0, 90.0, 0.5, 0.03, 0.3, option_type="put"))
    iv = _solve(price, method, s=100.0, k=90.0, t=0.5, r=0.03, option_type="put")
    assert iv == pytest.approx(0.3, abs=1e-6)


@pytest.mark.xfail(strict=True, reason="#21: unknown option types are priced as calls")
@pytest.mark.parametrize("option_type", ["Put", "p", "CALL"])
def test_unknown_option_type_raises(option_type):
    with pytest.raises(ValueError):
        bs_price(100.0, 100.0, 1.0, 0.0, 0.2, option_type=option_type)
    with pytest.raises(ValueError):
        implied_volatility(8.0, 100.0, 100.0, 1.0, 0.0, option_type=option_type)


@pytest.mark.xfail(strict=True, reason="#21: implied_volatility does not broadcast r and q")
def test_rates_and_dividends_broadcast():
    r, q = np.array([0.0, 0.03, 0.05]), np.array([0.0, 0.01, 0.02])
    prices = bs_price(100.0, 100.0, 1.0, r, 0.2, q=q)
    np.testing.assert_allclose(implied_volatility(prices, 100.0, 100.0, 1.0, r, q), 0.2, atol=1e-8)


@pytest.mark.xfail(strict=True, reason="#21: the scalar calculator fails on arrays")
def test_scalar_calculator_accepts_arrays():
    k = np.array([90.0, 100.0, 110.0])
    np.testing.assert_allclose(BlackScholesCalculator.call_price(100.0, k, 1.0, 0.03, 0.2),
                               bs_price(100.0, k, 1.0, 0.03, 0.2), rtol=1e-12)


@pytest.mark.xfail(strict=True, reason="#21: volatilities above 5 return NaN")
def test_high_volatility_round_trip():
    price = float(bs_price(100.0, 100.0, 0.01, 0.0, 6.0))
    assert implied_volatility(price, 100.0, 100.0, 0.01, 0.0) == pytest.approx(6.0, abs=1e-6)
