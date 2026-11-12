from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from scipy import integrate

from finpricing import HestonModel, HestonParameters, MarketData, OptionParams
from finpricing.calibration import HestonCalibrator
from finpricing.models.heston import (
    heston_cf,
    heston_price,
    simulate_heston,
    vix_future,
    vix_option_price,
    vix_spot,
)
from finpricing.models.heston.vix import _variance_law, vix_coefficients
from finpricing.utils.bs_utils import bs_price

# Fang & Oosterlee (2008) / Albrecher et al. (2007) benchmark: S=K=100, T=1, r=q=0
BENCH = HestonParameters(v0=0.0175, kappa=1.5768, theta=0.0398, xi=0.5751, rho=-0.5711)
BENCH_PRICE = 5.785155450

PARAM_SETS = [
    BENCH,
    HestonParameters(),                                              # Feller violated
    HestonParameters(v0=0.09, kappa=1.0, theta=0.06, xi=1.5, rho=-0.9),  # extreme skew
    HestonParameters(v0=0.01, kappa=3.0, theta=0.02, xi=0.1, rho=0.3),   # Feller satisfied
]
S0, R, Q = 100.0, 0.03, 0.01
STRIKES = np.array([60, 80, 90, 100, 110, 120, 150.0])


def _original_heston_call(S, K, T, r, p):
    """Heston (1993) P1/P2 formulation: an implementation independent of heston_cf."""
    def f(phi, j):
        u, b = (0.5, p.kappa - p.rho * p.xi) if j == 1 else (-0.5, p.kappa)
        x = p.rho * p.xi * phi * 1j
        d = np.sqrt((x - b) ** 2 - p.xi ** 2 * (2 * u * phi * 1j - phi ** 2))
        g = (b - x - d) / (b - x + d)
        e = np.exp(-d * T)
        D = (b - x - d) / p.xi ** 2 * (1 - e) / (1 - g * e)
        C = r * phi * 1j * T + p.kappa * p.theta / p.xi ** 2 * (
            (b - x - d) * T - 2 * np.log((1 - g * e) / (1 - g)))
        return np.exp(C + D * p.v0 + 1j * phi * np.log(S))

    def prob(j):
        val = integrate.quad(lambda ph: np.real(np.exp(-1j * ph * np.log(K)) * f(ph, j) / (1j * ph)),
                             1e-12, 1000, limit=2000, epsabs=1e-13)[0]
        return 0.5 + val / np.pi

    return S * prob(1) - K * np.exp(-r * T) * prob(2)


class TestCharacteristicFunction:
    @pytest.mark.parametrize("p", PARAM_SETS)
    @pytest.mark.parametrize("T", [0.1, 1.0, 10.0])
    def test_normalisation_and_martingale(self, p, T):
        assert heston_cf(0.0, T, p) == pytest.approx(1.0)
        assert heston_cf(-1j, T, p) == pytest.approx(1.0)  # E[S_T / F] = 1

    @pytest.mark.parametrize("T", [10.0, 30.0])
    def test_continuous_at_long_maturities(self, T):
        """The original formulation jumps across the log branch cut; this one must not."""
        u = np.linspace(0, 40, 4001)
        phi = heston_cf(u, T, HestonParameters(xi=1.0, rho=-0.9))
        assert np.max(np.abs(np.diff(phi))) < 0.05


class TestPricing:
    def test_literature_benchmark(self):
        assert heston_price(100, 100, 1.0, 0.0, BENCH) == pytest.approx(BENCH_PRICE, abs=1e-7)

    @pytest.mark.parametrize("p", PARAM_SETS[:3])
    @pytest.mark.parametrize("K", [80.0, 100.0, 125.0])
    def test_matches_original_formulation(self, p, K):
        assert heston_price(S0, K, 0.75, R, p) == pytest.approx(
            _original_heston_call(S0, K, 0.75, R, p), abs=1e-6)

    @pytest.mark.parametrize("method", ["fft", "fft-ct"])
    @pytest.mark.parametrize("T", [0.25, 1.0, 3.0])
    def test_fft_agrees_with_lewis(self, method, T):
        K = np.array([85, 95, 100, 105, 115.0])
        lewis = heston_price(S0, K, T, R, BENCH, q=Q)
        fft = heston_price(S0, K, T, R, BENCH, q=Q, method=method)
        np.testing.assert_allclose(fft, lewis, atol=5e-3)

    @pytest.mark.parametrize("p", PARAM_SETS)
    def test_put_call_parity(self, p):
        T = 0.5
        call = heston_price(S0, STRIKES, T, R, p, q=Q, option_type='call')
        put = heston_price(S0, STRIKES, T, R, p, q=Q, option_type='put')
        np.testing.assert_allclose(call - put, S0 * np.exp(-Q * T) - STRIKES * np.exp(-R * T),
                                   atol=1e-10)

    @pytest.mark.parametrize("p", PARAM_SETS)
    @pytest.mark.parametrize("T", [1 / 52, 0.5, 5.0])
    def test_no_arbitrage_shape(self, p, T):
        K = np.linspace(50, 200, 61)
        calls = heston_price(S0, K, T, R, p, q=Q)
        intrinsic = np.maximum(S0 * np.exp(-Q * T) - K * np.exp(-R * T), 0)
        assert np.all(calls >= intrinsic - 1e-8)
        assert np.all(np.diff(calls) <= 1e-8)              # decreasing in K
        assert np.all(np.diff(calls, 2) >= -1e-8)          # convex in K

    @pytest.mark.parametrize("T", [0.1, 1.0, 3.0])
    def test_black_scholes_limit(self, T):
        """xi -> 0 with v0 = theta: variance is constant, so Heston = Black-Scholes."""
        p = HestonParameters(v0=0.04, kappa=1.5, theta=0.04, xi=1e-3, rho=0.0)
        np.testing.assert_allclose(
            heston_price(S0, STRIKES, T, R, p, q=Q),
            bs_price(S0, STRIKES, T, R, 0.2, q=Q), atol=1e-4)

    def test_shapes(self):
        assert isinstance(heston_price(S0, 100, 1.0, R, BENCH), float)
        assert heston_price(S0, STRIKES, 1.0, R, BENCH).shape == STRIKES.shape
        grid = heston_price(S0, STRIKES[None, :], np.array([[0.5], [1.0]]), R, BENCH)
        assert grid.shape == (2, len(STRIKES))
        np.testing.assert_allclose(grid[1], heston_price(S0, STRIKES, 1.0, R, BENCH))

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            heston_price(S0, -1.0, 1.0, R, BENCH)
        with pytest.raises(ValueError):
            heston_price(S0, 100.0, 1.0, R, BENCH, option_type='straddle')
        with pytest.raises(ValueError):
            heston_price(S0, 100.0, 1.0, R, BENCH, method='cos')


class TestMonteCarlo:
    @pytest.fixture(scope="class", params=[BENCH, PARAM_SETS[2]], ids=["bench", "extreme"])
    def paths(self, request):
        p = request.param
        _, S, v = simulate_heston(S0, 1.0, R, p, q=Q, n_steps=50, n_paths=200_000, seed=7)
        return p, S, v

    def test_forward(self, paths):
        _, S, _ = paths
        ST = S[:, -1]
        se = ST.std() / np.sqrt(len(ST))
        assert abs(ST.mean() - S0 * np.exp((R - Q) * 1.0)) < 4 * se

    def test_variance_non_negative(self, paths):
        assert np.all(paths[2] >= 0)

    @pytest.mark.parametrize("K", [80.0, 100.0, 120.0])
    def test_matches_fourier(self, paths, K):
        p, S, _ = paths
        payoff = np.exp(-R) * np.maximum(S[:, -1] - K, 0)
        se = payoff.std() / np.sqrt(len(payoff))
        # 4 s.e. plus a small allowance for QE time-discretisation bias
        assert abs(payoff.mean() - heston_price(S0, K, 1.0, R, p, q=Q)) < 4 * se + 0.01


class TestVix:
    def test_spot(self):
        a, b = vix_coefficients(BENCH)
        assert vix_spot(BENCH) == pytest.approx(np.sqrt(a * BENCH.v0 + b))
        assert 0 < a < 1

    def test_deterministic_limit(self):
        """xi -> 0 with v0 = theta: VIX is constant at sqrt(theta)."""
        p = HestonParameters(v0=0.04, kappa=2.0, theta=0.04, xi=1e-3, rho=0.0)
        assert vix_future(0.5, p) == pytest.approx(0.2, abs=1e-5)
        assert vix_option_price(0.18, 0.5, 0.0, p) == pytest.approx(0.02, abs=1e-5)

    @pytest.mark.parametrize("p", [BENCH, HestonParameters()])
    def test_matches_exact_sampling(self, p):
        T = 0.25
        a, b = vix_coefficients(p)
        c, law = _variance_law(T, p)
        vix = np.sqrt(a * c * law.rvs(size=400_000, random_state=3) + b)
        se = vix.std() / np.sqrt(len(vix))
        assert abs(vix_future(T, p) - vix.mean()) < 4 * se
        for K in [0.15, 0.2, 0.25]:
            payoff = np.exp(-R * T) * np.maximum(vix - K, 0)
            assert abs(vix_option_price(K, T, R, p) - payoff.mean()) < 4 * payoff.std() / np.sqrt(len(vix))

    def test_put_call_parity(self):
        T, K = 0.5, np.array([0.12, 0.18, 0.25])
        call = vix_option_price(K, T, R, BENCH, 'call')
        put = vix_option_price(K, T, R, BENCH, 'put')
        np.testing.assert_allclose(call - put, np.exp(-R * T) * (vix_future(T, BENCH) - K))


class TestModel:
    def test_flat_smile_in_black_scholes_limit(self):
        m = HestonModel(HestonParameters(v0=0.04, kappa=1.5, theta=0.04, xi=1e-3, rho=0.0))
        np.testing.assert_allclose(m.implied_vol(S0, STRIKES[1:-1], 0.5, R, Q), 0.2, atol=1e-4)

    def test_negative_rho_gives_downward_skew(self):
        iv = HestonModel(BENCH).implied_vol(S0, [80, 100, 120], 0.5, R, Q)
        assert iv[0] > iv[1] > iv[2]

    def test_pricing_model_interface(self):
        m = HestonModel(BENCH)
        md = MarketData(spot_price=S0, risk_free_rate=R, dividend_yield=Q,
                        timestamp=datetime(2026, 1, 1))
        call = OptionParams(symbol="SPX", strike_price=100, option_type='call',
                            expiry=datetime(2027, 1, 1))
        T = 365 * 24 * 3600 / (365.25 * 24 * 3600)
        assert m.price(md, call) == pytest.approx(heston_price(S0, 100, T, R, BENCH, q=Q))
        g = m.greeks(md, call)
        assert 0 < g['delta'] < 1 and g['gamma'] > 0 and g['vega'] > 0 and g['rho'] > 0

    def test_greeks_match_black_scholes_limit(self):
        m = HestonModel(HestonParameters(v0=0.04, kappa=1.5, theta=0.04, xi=1e-3, rho=0.0))
        md = MarketData(spot_price=S0, risk_free_rate=R, timestamp=datetime(2026, 1, 1))
        op = OptionParams(symbol="X", strike_price=105, option_type='call',
                          expiry=datetime(2027, 1, 1))
        g = m.greeks(md, op)
        T = 365 / 365.25
        h = 1e-3 * S0
        bs = lambda s: float(bs_price(s, 105, T, R, 0.2))  # noqa: E731
        assert g['delta'] == pytest.approx((bs(S0 + h) - bs(S0 - h)) / (2 * h), rel=1e-3)


def _synthetic_surface(p):
    m = HestonModel(p)
    rows = []
    for T in [1 / 12, 0.25, 0.5, 1.0, 2.0]:
        K = np.array([75, 85, 95, 100, 105, 115, 125.0])
        types = np.where(S0 * np.exp((R - Q) * T) <= K, 'call', 'put')
        prices = m.option_price(S0, K, T, R, Q, option_type=types)
        rows += [dict(strike=k, maturity=T, price=pr, option_type=t)
                 for k, pr, t in zip(K, prices, types, strict=True)]
    df = pd.DataFrame(rows)
    df['S0'], df['r'], df['q'] = S0, R, Q
    return df


class TestCalibration:
    TRUE = HestonParameters(v0=0.03, kappa=2.0, theta=0.05, xi=0.6, rho=-0.65)

    def test_objective_zero_at_true_params(self):
        x = [self.TRUE.v0, self.TRUE.kappa, self.TRUE.theta, self.TRUE.xi, self.TRUE.rho]
        cal = HestonCalibrator()
        surface = _synthetic_surface(self.TRUE)
        assert cal.objective(x, surface) < 1e-16
        x[3] *= 1.3
        assert cal.objective(x, surface) > 1e-6

    def test_requires_spot(self):
        with pytest.raises(ValueError, match="S0"):
            HestonCalibrator().calibrate(pd.DataFrame({'strike': [100], 'maturity': [1], 'price': [5]}))

    def test_rejects_arbitrageable_quotes(self):
        df = pd.DataFrame({'strike': [100], 'maturity': [1.0], 'price': [150.0], 'S0': [100.0]})
        with pytest.raises(ValueError, match="no-arbitrage"):
            HestonCalibrator().calibrate(df)

    @pytest.mark.slow
    def test_recovers_parameters(self):
        fitted, info = HestonCalibrator().calibrate(_synthetic_surface(self.TRUE))
        for name in ['v0', 'kappa', 'theta', 'xi', 'rho']:
            assert getattr(fitted, name) == pytest.approx(getattr(self.TRUE, name), rel=1e-3)
        assert info['iv_rmse'] < 1e-5
