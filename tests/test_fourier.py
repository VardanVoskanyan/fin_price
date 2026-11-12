import numpy as np
import pytest

from finpricing.utils.bs_utils import bs_price, implied_volatility
from finpricing.utils.characteristic_functions import black_scholes_characteristic_function
from finpricing.utils.fourier import FourierMethod, fourier_call_pricer


@pytest.mark.parametrize("method", list(FourierMethod))
@pytest.mark.parametrize("r", [0.0, 0.04])
@pytest.mark.parametrize("q", [0.0, 0.02, 0.04])
@pytest.mark.parametrize("s_0", [0.0, 0.2])
def test_fft_pricers_reproduce_black_scholes(method, r, q, s_0):
    """
    Regressions for Cont-Tankov: it returned C - C_BS(sigma_ref) instead of C, and
    assumed the forward is e^{rT} (wrong with dividends, or for VIX futures, where q = r).
    """
    T, sigma = 0.5, 0.25

    def cf(u):
        return np.exp(1j * u * (r - q) * T) * black_scholes_characteristic_function(u, T, sigma)

    log_k, prices = fourier_call_pricer(cf, T, r, s_0=s_0, method=method)
    near = np.abs(log_k - s_0) < 0.5
    expected = bs_price(1.0, np.exp(log_k[near]), T, r, sigma, q=q)
    np.testing.assert_allclose(prices[near], expected, atol=1e-6)


def test_implied_volatility_round_trip():
    K = np.array([70, 90, 100, 110, 140.0])
    sig = np.array([0.35, 0.25, 0.2, 0.18, 0.22])
    types = np.array(['put', 'put', 'call', 'call', 'call'])
    prices = bs_price(100, K, 0.75, 0.03, sig, q=0.01, option_type=types)
    np.testing.assert_allclose(
        implied_volatility(prices, 100, K, 0.75, 0.03, q=0.01, option_type=types), sig, atol=1e-10)


def test_implied_volatility_nan_outside_bounds():
    assert np.isnan(implied_volatility(150.0, 100, 100, 1.0, 0.0))
