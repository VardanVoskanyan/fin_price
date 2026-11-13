import pytest
import numpy as np
from finpricing.models.vix.model import VixModel
from finpricing.data.fixtures import TENOR_DATES


V0 = 0.20   # representative VIX level
R = 0.02    # risk-free rate


@pytest.fixture(scope="module")
def merton():
    return VixModel(model_type="Merton")


@pytest.fixture(scope="module")
def kou():
    return VixModel(model_type="Kou")


class TestVixOptionPricerSmoke:
    """vix_option_pricer (Fourier) returns a finite non-negative float."""

    @pytest.mark.parametrize("model_name", ["Merton", "Kou"])
    @pytest.mark.parametrize("option_type", ["call", "put"])
    def test_nonnegative(self, model_name, option_type):
        model = VixModel(model_type=model_name)
        price = model.vix_option_pricer(V0, K=0.20, T=TENOR_DATES[1], r=R, option_type=option_type)
        assert np.isfinite(price), "price must be finite"
        assert price >= 0.0, f"option price must be non-negative: {price}"

    @pytest.mark.parametrize("model_name", ["Merton", "Kou"])
    def test_deep_otm_near_zero(self, model_name):
        model = VixModel(model_type=model_name)
        price = model.vix_option_pricer(V0, K=1.00, T=TENOR_DATES[1], r=R, option_type="call")
        assert price < 0.01, "deep OTM call should be near zero"


class TestVixCallPutParity:
    """
    The Fourier put is defined as: P = exp(-rT)*K + C - V0
    So C - P = V0 - exp(-rT)*K exactly (no Monte Carlo variance).
    """

    @pytest.mark.parametrize("model_name", ["Merton", "Kou"])
    @pytest.mark.parametrize("K", [0.15, 0.20, 0.25])
    def test_parity(self, model_name, K):
        T = TENOR_DATES[2]
        model = VixModel(model_type=model_name)
        call = model.vix_option_pricer(V0, K=K, T=T, r=R, option_type="call")
        put = model.vix_option_pricer(V0, K=K, T=T, r=R, option_type="put")
        expected = V0 - np.exp(-R * T) * K
        assert abs(call - put - expected) < 1e-10, (
            f"C - P = {call - put:.6f}, expected {expected:.6f}"
        )


class TestIndexOptionPricerSmoke:
    """index_option_pricer (MC) returns non-negative prices for each strike."""

    def test_merton_call_prices(self, merton):
        merton.store_tenor_data(num_paths=2_000)
        strikes = np.array([90.0, 100.0, 110.0])
        prices = merton.index_option_pricer(S0=100.0, strikes=strikes, tenor_index=1, r=R)
        assert prices.shape == (3,)
        assert np.all(prices >= 0), "call prices must be non-negative"
        assert np.all(np.isfinite(prices)), "all prices must be finite"
        assert prices[0] >= prices[1] >= prices[2], "calls must be monotone decreasing in strike"

    def test_kou_call_prices(self, kou):
        kou.store_tenor_data(num_paths=2_000)
        strikes = np.array([90.0, 100.0, 110.0])
        prices = kou.index_option_pricer(S0=100.0, strikes=strikes, tenor_index=1, r=R)
        assert prices.shape == (3,)
        assert np.all(prices >= 0), "call prices must be non-negative"
        assert np.all(np.isfinite(prices)), "all prices must be finite"


@pytest.mark.parametrize("model_name", ["Merton", "Kou"])
@pytest.mark.parametrize("r", [0.0, 0.03])
def test_fourier_methods_agree(model_name, r):
    """Kar-Madan and Cont-Tankov must give the same VIX option prices."""
    from finpricing.data.fixtures import KOU_PARAMS, MERTON_PARAMS
    from finpricing.models.vix.pricing import price_vix_options

    params = MERTON_PARAMS if model_name == "Merton" else KOU_PARAMS
    for K in [0.16, 0.20, 0.26]:
        km = price_vix_options(V0, K, TENOR_DATES[2], r, params, 'Kar&Madan')
        ct = price_vix_options(V0, K, TENOR_DATES[2], r, params, 'Cont&Tankov')
        assert ct == pytest.approx(km, abs=1e-6)
