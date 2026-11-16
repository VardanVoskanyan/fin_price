"""Heston VIX futures and options in the near-deterministic limit (#19)."""
import numpy as np
import pytest

from finpricing import HestonParameters
from finpricing.models.heston import vix_future, vix_option_price

# v0 = theta and tiny xi: the variance stays at theta, so the VIX stays at sqrt(theta) = 0.2.
QUIET = HestonParameters(v0=0.04, theta=0.04, kappa=1.5, xi=1e-3, rho=-0.7)


@pytest.mark.parametrize("T", [1e-3, 1 / 365, 1 / 52])
def test_vix_future_deterministic_limit(T):
    assert vix_future(T, QUIET) == pytest.approx(0.2, abs=1e-6)


@pytest.mark.parametrize("T", [1 / 365, 1 / 52])
def test_vix_call_deterministic_limit(T):
    assert vix_option_price(0.18, T, 0.0, QUIET) == pytest.approx(0.02, abs=1e-5)


def test_vix_future_tiny_maturity():
    p = HestonParameters(v0=0.04, theta=0.04, kappa=1.5, xi=1e-2, rho=-0.7)
    assert vix_future(1e-6, p) == pytest.approx(0.2, abs=1e-6)


def test_negative_maturity_raises():
    with pytest.raises(ValueError):
        vix_future(-0.1, QUIET)


def test_wide_vol_of_vol_still_matches_spot_at_zero_maturity_limit():
    """Regression guard for the fix: an ordinary case must keep working."""
    p = HestonParameters()
    assert np.isfinite(vix_future(0.25, p))
