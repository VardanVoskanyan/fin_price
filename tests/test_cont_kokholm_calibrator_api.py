"""ContKokholmCalibrator: validated options, seeded, rho fixed, per-maturity pricing, report (#13, #23)."""
import numpy as np
import pytest

from finpricing.calibration import OptionSurface
from finpricing.calibration import cont_kokholm as ck_module
from finpricing.calibration.cont_kokholm import ContKokholmCalibrator
from finpricing.data.fixtures import MERTON_PARAMS
from finpricing.models.vix.pricing import price_vix_options
from finpricing.utils import fourier

REPORT_KEYS = {"success", "message", "loss", "n_quotes", "at_bounds"}
# Merton vector without rho, which VIX options do not depend on (#23)
MERTON_X = np.array([getattr(MERTON_PARAMS, n) for n in ["omega", "lmbd", "k1", "k2", "m", "delta"]])


def _surface(n_maturities=3):
    K, T = np.meshgrid([0.18, 0.2, 0.22, 0.25, 0.3], [1 / 12, 0.25, 0.5][:n_maturities])
    K, T = K.ravel(), T.ravel()
    prices = [price_vix_options(0.2, k, t, 0.0, MERTON_PARAMS) for k, t in zip(K, T, strict=True)]
    return OptionSurface(strike=K, maturity=T, price=prices, spot=0.2)


def _capture_de(monkeypatch):
    """Replace the optimisers with stubs: these tests check wiring, not convergence."""
    seen = {}

    class Result:
        def __init__(self, x):
            self.x, self.fun = np.asarray(x, dtype=float), 0.0

    def fake_de(func, bounds, **kwargs):
        seen["bounds"], seen["kwargs"] = list(bounds), kwargs
        x = [0.5 * (lo + hi) for lo, hi in bounds]
        func(np.asarray(x))
        return Result(x)

    def fake_minimize(func, x0, **kwargs):
        return Result(x0)

    monkeypatch.setattr(ck_module, "differential_evolution", fake_de)
    monkeypatch.setattr(ck_module, "minimize", fake_minimize, raising=False)   # local refinement
    return seen


@pytest.mark.xfail(strict=True, reason="#23: unknown options are not validated")
@pytest.mark.parametrize("kwargs", [{"pricing_method": "Carr-Madan"}, {"model_type": "Heston"}])
def test_invalid_options_raise(kwargs):
    with pytest.raises(ValueError):
        ContKokholmCalibrator(**kwargs)


@pytest.mark.xfail(strict=True, reason="#23: calibrate swallows unknown keyword arguments")
def test_unknown_calibrate_arguments_raise(monkeypatch):
    _capture_de(monkeypatch)
    with pytest.raises(TypeError):
        ContKokholmCalibrator().calibrate(_surface(), typo=1)


@pytest.mark.xfail(strict=True, reason="#23: the seed is hard-coded to 42")
def test_seed_reaches_the_optimiser(monkeypatch):
    seen = _capture_de(monkeypatch)
    ContKokholmCalibrator(seed=7).calibrate(_surface())
    assert seen["kwargs"]["seed"] == 7


@pytest.mark.xfail(strict=True, reason="#23: rho is fitted although VIX options do not depend on it")
@pytest.mark.parametrize("model_type", ["Merton", "Kou"])
def test_rho_is_fixed_not_fitted(monkeypatch, model_type):
    seen = _capture_de(monkeypatch)
    params, _ = ContKokholmCalibrator(model_type, rho=-0.6).calibrate(_surface())
    assert len(seen["bounds"]) == {"Merton": 6, "Kou": 7}[model_type]
    assert params.rho == -0.6


@pytest.mark.xfail(strict=True, reason="#23: the objective reprices the FFT grid once per quote")
def test_objective_prices_once_per_maturity(monkeypatch):
    """Every FFT pricer call goes through utils.fourier; count them."""
    calls = {"n": 0}
    for name in ["kar_madan_pricer", "cont_tankov_pricer"]:
        original = getattr(fourier, name)

        def counting(*args, _original=original, **kwargs):
            calls["n"] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(fourier, name, counting)
    value = ContKokholmCalibrator("Merton").objective(MERTON_X, _surface(n_maturities=3))
    assert value < 1e-12
    assert calls["n"] == 3


@pytest.mark.xfail(strict=True, reason="#13: the Cont-Kokholm calibrator returns info=None")
def test_report_has_every_field(monkeypatch):
    _capture_de(monkeypatch)
    _, info = ContKokholmCalibrator().calibrate(_surface())
    assert set(info) >= REPORT_KEYS
    assert info["n_quotes"] == 15


@pytest.mark.xfail(strict=True, reason="#13: programming errors inside the objective are swallowed")
def test_programming_errors_propagate(monkeypatch):
    surface = _surface()                     # priced before the pricer is broken

    def broken(*args, **kwargs):
        raise TypeError("bug")

    monkeypatch.setattr(fourier, "kar_madan_pricer", broken)
    with pytest.raises(TypeError):
        ContKokholmCalibrator("Merton").objective(MERTON_X, surface)
