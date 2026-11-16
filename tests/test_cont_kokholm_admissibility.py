"""Cont-Kokholm parameters that make the model ill-defined must be rejected (#16)."""
import numpy as np
import pytest

from finpricing.calibration.cont_kokholm import _KOU_BOUNDS, _KOU_NAMES
from finpricing.data.fixtures import MERTON_B_I, TENOR_DATES, VI_0
from finpricing.models.jump_diffusion.jump_diffusion import simulate_tenor_diffusions
from finpricing.models.vix.model import VixModel
from finpricing.models.vix.pricing import price_index_options
from finpricing.parameters import VIXKouModelParameters, VIXMertonModelParameters


@pytest.mark.xfail(strict=True, reason="#16: E[e^Y] is infinite for alpha_plus <= 1")
@pytest.mark.parametrize("alpha_plus", [0.8, 1.0])
def test_kou_requires_finite_exponential_moment(alpha_plus):
    with pytest.raises(ValueError):
        VIXKouModelParameters(alpha_plus=alpha_plus)


@pytest.mark.xfail(strict=True, reason="#16: calibrator bounds allow alpha_plus <= 1")
def test_kou_calibrator_bounds_exclude_infinite_moments():
    low, _ = _KOU_BOUNDS[_KOU_NAMES.index("alpha_plus")]
    assert low > 1.0


@pytest.mark.xfail(strict=True, reason="#16: negative sigma_i^2 gives NaN prices instead of an error")
def test_negative_index_variance_raises():
    """lmbd=10, m=0.5, delta=0.3 makes V^i_0 - cf * b_i^2 < 0, i.e. sigma_i^2 < 0 on every path."""
    np.random.seed(0)
    model = VixModel(model_type="Merton")
    model.store_tenor_data(2_000)
    bad = VIXMertonModelParameters(lmbd=10, m=0.5, delta=0.3)
    with pytest.raises(ValueError, match="sigma"):
        price_index_options(100.0, VI_0, np.array([90.0, 100.0, 110.0]), 0.0, TENOR_DATES[:4],
                            model._stored_data, MERTON_B_I, bad)


@pytest.mark.xfail(strict=True, reason="#16: exp(2 k1 T) overflows at long tenors")
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
