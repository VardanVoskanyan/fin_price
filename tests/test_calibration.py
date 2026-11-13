import numpy as np
import pytest

from finpricing.calibration import ContKokholmCalibrator, OptionSurface
from finpricing.calibration.cont_kokholm import _KOU_NAMES, _MERTON_NAMES
from finpricing.data.fixtures import KOU_PARAMS, MERTON_PARAMS
from finpricing.models.vix.pricing import price_vix_options

V0 = 0.20
STRIKES = [0.16, 0.18, 0.20, 0.22, 0.25]
MATURITIES = [1 / 12, 3 / 12]

CASES = {
    "Merton": (MERTON_PARAMS, _MERTON_NAMES),
    "Kou": (KOU_PARAMS, _KOU_NAMES),
}


def _surface(params, V0=V0, r=0.0):
    K, T = np.meshgrid(STRIKES, MATURITIES)
    K, T = K.ravel(), T.ravel()
    prices = [price_vix_options(V0, k, t, r, params) for k, t in zip(K, T, strict=True)]
    return OptionSurface(strike=K, maturity=T, price=prices, spot=V0, r=r)


def _vector(params, names):
    return np.array([getattr(params, n) for n in names])


class TestOptionSurface:
    def test_broadcasts_scalars(self):
        s = OptionSurface(strike=[90, 100], maturity=0.5, price=[12.0, 5.0], spot=100)
        assert len(s) == 2
        assert list(s.option_type) == ['call', 'call']
        np.testing.assert_array_equal(s.maturity, [0.5, 0.5])
        np.testing.assert_array_equal(s.weight, [1.0, 1.0])

    def test_from_columns_dict(self):
        s = OptionSurface.from_columns(
            {'strike': [90, 110], 'maturity': [0.5, 1.0], 'price': [12.0, 3.0],
             'option_type': ['put', 'call']},
            spot=100, r=0.02)
        assert list(s.option_type) == ['put', 'call'] and s.r == 0.02

    def test_from_columns_missing(self):
        with pytest.raises(ValueError, match="Missing required columns"):
            OptionSurface.from_columns({'strike': [100], 'price': [5.0]}, spot=100)

    @pytest.mark.parametrize("kwargs", [
        dict(strike=[100, 110], maturity=[1.0], price=[5.0, 2.0]),     # length mismatch
        dict(strike=[-100], maturity=1.0, price=5.0),                  # negative strike
        dict(strike=[100], maturity=0.0, price=5.0),                   # zero maturity
        dict(strike=[100], maturity=1.0, price=np.nan),                # bad price
        dict(strike=[100], maturity=1.0, price=5.0, option_type='straddle'),
    ])
    def test_rejects_bad_input(self, kwargs):
        with pytest.raises(ValueError):
            OptionSurface(spot=100, **kwargs)

    def test_immutable(self):
        s = OptionSurface(strike=[100], maturity=1.0, price=5.0, spot=100)
        with pytest.raises(ValueError):
            s.price[0] = 1.0


@pytest.mark.parametrize("model_type", ["Merton", "Kou"])
class TestObjective:
    def test_zero_at_true_params(self, model_type):
        params, names = CASES[model_type]
        cal = ContKokholmCalibrator(model_type)
        assert cal.objective(_vector(params, names), _surface(params)) < 1e-20

    def test_positive_away_from_true_params(self, model_type):
        params, names = CASES[model_type]
        cal = ContKokholmCalibrator(model_type)
        x = _vector(params, names)
        x[0] *= 1.5  # perturb omega
        assert cal.objective(x, _surface(params)) > 1e-8

    def test_uses_surface_spot_and_r(self, model_type):
        """Regression: the calibrator used to price every row with V0=K and r=0 (always ATM)."""
        params, names = CASES[model_type]
        cal = ContKokholmCalibrator(model_type)
        surface = _surface(params, V0=0.25, r=0.02)
        assert cal.objective(_vector(params, names), surface) < 1e-20


@pytest.mark.slow
def test_calibrate_improves_fit():
    surface = _surface(MERTON_PARAMS)
    cal = ContKokholmCalibrator("Merton", de_maxiter=10, de_popsize=5)
    fitted, _ = cal.calibrate(surface)
    assert cal.objective(_vector(fitted, _MERTON_NAMES), surface) < 1e-4
