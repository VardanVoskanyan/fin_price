"""VixModel: user parameters, validated inputs, no stale state, real index paths (#15)."""
import numpy as np
import pytest

from finpricing.data.fixtures import MERTON_B_I, TENOR_DATES, VI_0
from finpricing.models.vix.model import VixModel
from finpricing.parameters import VIXKouModelParameters, VIXMertonModelParameters

STRIKES = np.array([90.0, 100.0, 110.0])


def test_constructor_takes_user_parameters():
    params = VIXMertonModelParameters(lmbd=2.0)
    vi_0, b_i = VI_0 * 1.1, MERTON_B_I * 0.9
    model = VixModel(model_type="Merton", params=params, Vi_0=vi_0, b_i=b_i)
    assert model.params.lmbd == 2.0
    np.testing.assert_array_equal(model.Vi_0, vi_0)
    np.testing.assert_array_equal(model.b_i, b_i)


@pytest.mark.parametrize("kwargs", [
    {"Vi_0": VI_0[:5]},                                           # one value short
    {"b_i": MERTON_B_I[:4]},
    {"tenor_dates": TENOR_DATES[:5]},                             # Vi_0 / b_i now too long
    {"model_type": "Kou", "params": VIXMertonModelParameters()},  # params of the wrong family
])
def test_inconsistent_inputs_raise(kwargs):
    with pytest.raises(ValueError):
        VixModel(**{"model_type": "Merton", **kwargs})


def test_kou_parameters_are_accepted():
    model = VixModel(model_type="Kou")
    assert isinstance(model.params, VIXKouModelParameters)


@pytest.mark.parametrize("name", ["params", "tenor_dates", "Vi_0", "b_i"])
def test_inputs_are_read_only(name):
    model = VixModel(model_type="Merton")
    with pytest.raises(AttributeError):
        setattr(model, name, getattr(model, name))


@pytest.mark.parametrize("tenor_index", [0, 7, 10, -1])
def test_tenor_index_out_of_range_raises(tenor_index):
    model = VixModel(model_type="Merton")
    model.store_tenor_data(1_000, rng=0)
    with pytest.raises(ValueError, match="tenor_index"):
        model.index_option_pricer(S0=100.0, strikes=STRIKES, tenor_index=tenor_index)


@pytest.mark.parametrize("r", [0.0, 0.03])
def test_index_paths_reproduce_option_prices(r):
    """Call prices from the simulated S_T must match the conditional Monte Carlo pricer."""
    model = VixModel(model_type="Merton")
    model.store_tenor_data(200_000, rng=5)
    tenor_index = 3
    T = model.tenor_dates[tenor_index]
    S_T = model.get_index_prices(100.0, r=r)[:, tenor_index - 1]
    payoff = np.exp(-r * T) * np.maximum(S_T[:, None] - STRIKES, 0.0)
    mc, se = payoff.mean(0), payoff.std(0) / np.sqrt(len(S_T))
    reference = model.index_option_pricer(S0=100.0, strikes=STRIKES, tenor_index=tenor_index, r=r)
    assert np.all(np.abs(mc - reference) < 4 * se), (mc, reference, se)
