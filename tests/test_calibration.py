import numpy as np
import pandas as pd
import pytest

from finpricing.calibration import ContKokholmCalibrator
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
    rows = [
        (K, T, price_vix_options(V0, K, T, r, params))
        for K in STRIKES
        for T in MATURITIES
    ]
    df = pd.DataFrame(rows, columns=["strike", "maturity", "price"])
    df["V0"] = V0
    df["r"] = r
    return df


def _vector(params, names):
    return np.array([getattr(params, n) for n in names])


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

    def test_uses_surface_V0_and_r(self, model_type):
        """Regression: the calibrator used to price every row with V0=K and r=0 (always ATM)."""
        params, names = CASES[model_type]
        cal = ContKokholmCalibrator(model_type)
        surface = _surface(params, V0=0.25, r=0.02)
        assert cal.objective(_vector(params, names), surface) < 1e-20


def test_missing_columns_raise():
    cal = ContKokholmCalibrator("Merton")
    with pytest.raises(ValueError, match="missing required columns"):
        cal.calibrate(pd.DataFrame({"strike": [0.2], "price": [0.01]}))


@pytest.mark.slow
def test_calibrate_improves_fit():
    surface = _surface(MERTON_PARAMS)
    cal = ContKokholmCalibrator("Merton", de_maxiter=10, de_popsize=5)
    fitted, _ = cal.calibrate(surface)
    loss = cal.objective(_vector(fitted, _MERTON_NAMES), surface)
    assert loss < 1e-4
