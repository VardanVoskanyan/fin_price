"""Calibrators must report how the fit went and never fail silently (#13)."""
import numpy as np
import pytest

from finpricing import HestonModel, HestonParameters
from finpricing.calibration import HestonCalibrator, OptionSurface
from finpricing.calibration import heston as heston_module

REPORT_KEYS = {"success", "message", "loss", "n_quotes", "at_bounds"}
HESTON_KEYS = REPORT_KEYS | {"iv_rmse", "max_iv_error", "n_iv_failed", "feller"}
FAST = dict(de_maxiter=3, de_popsize=5, seed=1)


TRUE = HestonParameters(v0=0.04, kappa=2.0, theta=0.05, xi=0.6, rho=-0.7)


def _surface(params=TRUE):
    K, T = np.meshgrid([85.0, 95.0, 100.0, 105.0, 115.0], [0.25, 0.5, 1.0])
    K, T = K.ravel(), T.ravel()
    prices = HestonModel(params).option_price(100.0, K, T, 0.03)
    return OptionSurface(strike=K, maturity=T, price=prices, spot=100.0, r=0.03)


@pytest.mark.xfail(strict=True, reason="#13: calibration report is missing fields")
def test_heston_report_has_every_field():
    _, info = HestonCalibrator(**FAST).calibrate(_surface())
    assert set(info) >= HESTON_KEYS
    assert info["n_quotes"] == 15
    assert isinstance(info["success"], bool) and isinstance(info["message"], str)
    assert info["n_iv_failed"] == 0
    assert info["max_iv_error"] >= info["iv_rmse"] >= 0


@pytest.mark.xfail(strict=True, reason="#13: every evaluation failing returns parameters silently")
def test_heston_raises_when_every_evaluation_fails(monkeypatch):
    def broken(*args, **kwargs):
        raise ValueError("pricer failed")

    monkeypatch.setattr(heston_module, "heston_price", broken)
    with pytest.raises(RuntimeError):
        HestonCalibrator(**FAST).calibrate(_surface())


@pytest.mark.xfail(strict=True, reason="#13: bounds are fixed and parameters stuck on a bound are not reported")
def test_heston_reports_parameters_at_bounds():
    """True kappa = 2 with kappa capped at 0.5: the fit must end on the cap and say so."""
    calibrator = HestonCalibrator(bounds={"kappa": (0.01, 0.5)}, **FAST)
    params, info = calibrator.calibrate(_surface())
    assert params.kappa == pytest.approx(0.5, rel=1e-3)
    assert "kappa" in info["at_bounds"]
    assert info["success"] is False


@pytest.mark.xfail(strict=True, reason="#13: unknown bound names are silently ignored")
def test_heston_rejects_unknown_bound_names():
    with pytest.raises(ValueError):
        HestonCalibrator(bounds={"kapa": (0.1, 5.0)})


@pytest.mark.xfail(strict=True, reason="#13: iv_rmse ignores quotes whose model IV is NaN")
def test_heston_counts_failed_implied_vols(monkeypatch):
    """If the model IV of some quotes cannot be computed, the report must say how many."""
    real = heston_module.implied_volatility
    calls = {"n": 0}

    def flaky(prices, *args, **kwargs):
        calls["n"] += 1
        iv = np.array(real(prices, *args, **kwargs), dtype=float)
        if calls["n"] > 1:                    # the first call prepares market IVs
            iv[:2] = np.nan
        return iv

    monkeypatch.setattr(heston_module, "implied_volatility", flaky)
    _, info = HestonCalibrator(**FAST).calibrate(_surface())
    assert info["n_iv_failed"] == 2
    assert info["success"] is False
