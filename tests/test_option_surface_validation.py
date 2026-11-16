"""OptionSurface must reject quotes a calibrator cannot use (#17)."""
import numpy as np
import pytest

from finpricing.calibration import OptionSurface

GOOD = dict(strike=[90.0, 100.0, 110.0], maturity=[1.0, 1.0, 1.0], price=[13.0, 6.0, 2.0],
            spot=100.0, r=0.03)

BAD = {
    "NaN maturity": dict(maturity=[np.nan, 1.0, 1.0]),
    "infinite strike": dict(strike=[90.0, 100.0, np.inf]),
    "infinite spot": dict(spot=np.inf),
    "NaN rate": dict(r=np.nan),
    "NaN dividend yield": dict(q=np.nan),
    "NaN weight": dict(weight=[1.0, np.nan, 1.0]),
    "all weights zero": dict(weight=0.0),
    "duplicate quote with two prices": dict(strike=[100.0, 100.0, 110.0], price=[6.0, 7.0, 2.0]),
    "call above spot": dict(price=[150.0, 6.0, 2.0]),
    "call below intrinsic": dict(strike=[50.0, 100.0, 110.0], price=[40.0, 6.0, 2.0]),
    "put above discounted strike": dict(option_type="put", price=[95.0, 6.0, 2.0]),
}


def test_good_surface_is_accepted():
    OptionSurface(**GOOD)


@pytest.mark.parametrize("spot", [0.0, -1.0, np.nan])
def test_bad_spot_is_rejected(spot):
    with pytest.raises(ValueError):
        OptionSurface(**{**GOOD, "spot": spot})


@pytest.mark.xfail(strict=True, reason="#17: OptionSurface validation gaps")
@pytest.mark.parametrize("case", list(BAD))
def test_bad_surface_is_rejected(case):
    with pytest.raises(ValueError):
        OptionSurface(**{**GOOD, **BAD[case]})


def test_zero_price_far_out_of_the_money_is_allowed():
    """A zero bid is a legitimate quote; the bounds check must not reject it."""
    OptionSurface(**{**GOOD, "strike": [90.0, 100.0, 400.0], "price": [13.0, 6.0, 0.0]})
