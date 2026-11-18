"""Cont-Kokholm parameters that make the model ill-defined must be rejected (#16)."""
import numpy as np
import pytest

from finpricing.calibration.cont_kokholm import _KOU_BOUNDS, _KOU_NAMES
from finpricing.data.fixtures import MERTON_B_I, TENOR_DATES, VI_0
from finpricing.models.jump_diffusion.jump_diffusion import simulate_tenor_diffusions
from finpricing.models.vix.model import VixModel
from finpricing.models.vix.pricing import price_index_options
from finpricing.parameters import VIXKouModelParameters, VIXMertonModelParameters


@pytest.mark.parametrize("alpha_plus", [0.8, 1.0])
def test_kou_requires_finite_exponential_moment(alpha_plus):
    with pytest.raises(ValueError):
        VIXKouModelParameters(alpha_plus=alpha_plus)


def test_kou_calibrator_bounds_exclude_infinite_moments():
    low, _ = _KOU_BOUNDS[_KOU_NAMES.index("alpha_plus")]
    assert low > 1.0


def test_negative_index_variance_raises():
    """lmbd=10, m=0.5, delta=0.3 makes V^i_0 - cf * b_i^2 < 0, i.e. sigma_i^2 < 0 on every path."""
    np.random.seed(0)
    model = VixModel(model_type="Merton")
    model.store_tenor_data(2_000)
    bad = VIXMertonModelParameters(lmbd=10, m=0.5, delta=0.3)
    with pytest.raises(ValueError, match="sigma"):
        price_index_options(100.0, VI_0, np.array([90.0, 100.0, 110.0]), 0.0, TENOR_DATES[:4],
                            model._stored_data, MERTON_B_I, bad)


def test_diffusions_finite_at_long_tenors():
    np.random.seed(0)
    ZV, dZ = simulate_tenor_diffusions(1_000, np.array([0.0, 1 / 12, 2.0, 6.5]), k1=60.0)
    assert np.all(np.isfinite(ZV)) and np.all(np.isfinite(dZ))


def test_diffusion_variance_unchanged_for_paper_tenors():
    """Guard for the rewrite: Var(ZV_i) = (1 - exp(-2 k1 T_i)) / (2 k1)."""
    np.random.seed(1)
    k1 = 21.9
    ZV, _ = simulate_tenor_diffusions(200_000, TENOR_DATES, k1=k1)
    expected = (1 - np.exp(-2 * k1 * TENOR_DATES[1:])) / (2 * k1)
    np.testing.assert_allclose(ZV.var(0), expected, rtol=0.02)


def test_vix_model_negative_index_variance_raises():
    """The issue's reproducer: VixModel must raise rather than return NaN index prices."""
    np.random.seed(0)
    model = VixModel(model_type="Merton")
    model.params = VIXMertonModelParameters(lmbd=10, m=0.5, delta=0.3)
    model.constraint_cf = model.params.lmbd * (model.params.m**2 + model.params.delta**2)
    model.store_tenor_data(2_000)
    with pytest.raises(ValueError, match="sigma"):
        model.index_option_pricer(S0=100, strikes=np.array([90.0, 100.0, 110.0]), tenor_index=3)
    with pytest.raises(ValueError, match="sigma"):
        model.get_index_prices(100.0)


@pytest.mark.parametrize("T", [1 / 12, 0.25, 0.5])
def test_fourier_pricers_agree_at_the_kou_calibrator_bound(T):
    """Kar-Madan's fixed damping is wrong near alpha_plus = 1 (#33); the calibrator must
    not search there."""
    from finpricing.models.vix.pricing import price_vix_options

    low, _ = _KOU_BOUNDS[_KOU_NAMES.index("alpha_plus")]
    params = VIXKouModelParameters(alpha_plus=low)
@pytest.mark.parametrize("alpha_plus", [1.02, 1.05, 1.1, 1.2])
@pytest.mark.parametrize("T", [1 / 12, 0.25, 0.5])
def test_kar_madan_accurate_near_the_kou_pole(alpha_plus, T):
    """Kar-Madan's shifted contour approaches the pole of the Kou jump transform as
    alpha_plus -> 1; the grid must still resolve the integrand (#33)."""
    from finpricing.models.vix.pricing import price_vix_options

    params = VIXKouModelParameters(alpha_plus=alpha_plus)
    for K in [0.8, 1.0, 1.3, 1.6]:
        km = price_vix_options(1.0, K, T, 0.0, params, "Kar&Madan")
        ct = price_vix_options(1.0, K, T, 0.0, params, "Cont&Tankov")
        assert km == pytest.approx(ct, abs=1e-4)
