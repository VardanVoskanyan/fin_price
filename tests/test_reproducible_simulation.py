"""Simulations take a seed or Generator and leave NumPy's global state alone (#22)."""
import numpy as np
import pytest

from finpricing.data.fixtures import MERTON_PARAMS, TENOR_DATES
from finpricing.models.jump_diffusion.jump_diffusion import (
    simul_tenor_jumps,
    simulate_tenor_diffusions,
)
from finpricing.models.vix.model import VixModel
from finpricing.utils.math_utils import generate_correlated_diffusions

pytestmark = pytest.mark.xfail(strict=True, reason="#22: simulations use NumPy's global RNG")


def _global_state():
    return np.random.get_state()[1].copy()


def _price(seed):
    model = VixModel(model_type="Merton")
    model.store_tenor_data(5_000, rng=seed)
    return model.index_option_pricer(S0=100.0, strikes=np.array([95.0, 100.0, 105.0]), tenor_index=2)


def test_same_seed_same_prices():
    np.testing.assert_array_equal(_price(1), _price(1))


def test_different_seeds_differ():
    assert not np.array_equal(_price(1), _price(2))


def test_generator_instances_are_accepted():
    a = _price(np.random.default_rng(3))
    b = _price(np.random.default_rng(3))
    np.testing.assert_array_equal(a, b)


def test_global_state_untouched():
    before = _global_state()
    _price(4)
    np.testing.assert_array_equal(before, _global_state())


@pytest.mark.parametrize("simulate", [
    lambda rng: simulate_tenor_diffusions(1_000, TENOR_DATES, k1=21.9, rng=rng),
    lambda rng: simul_tenor_jumps(MERTON_PARAMS, 1_000, TENOR_DATES, rng=rng),
    lambda rng: generate_correlated_diffusions(1_000, TENOR_DATES, rho=-0.5, rng=rng),
])
def test_low_level_simulators_are_seeded(simulate):
    before = _global_state()
    first, second = simulate(9), simulate(9)
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(before, _global_state())
