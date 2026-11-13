"""
Heston model: pricing, Greeks, simulation and VIX under one object.
"""
import numpy as np

from finpricing.base import Greeks, MarketData, OptionParams, PricingModel
from finpricing.models.heston import vix
from finpricing.models.heston.pricing import heston_price
from finpricing.models.heston.simulation import simulate_heston
from finpricing.parameters import HestonParameters
from finpricing.utils.bs_utils import implied_volatility


class HestonModel(PricingModel):
    """
    Heston (1993) stochastic volatility model.

    Example:
        >>> m = HestonModel(HestonParameters(v0=0.04, kappa=1.5, theta=0.04, xi=0.5, rho=-0.7))
        >>> m.option_price(S0=100, K=[90, 100, 110], T=0.5, r=0.03)
    """

    def __init__(self, params: HestonParameters | None = None):
        self.params = params or HestonParameters()

    # ── vanilla options ──────────────────────────────────────────────────────
    def option_price(self, S0, K, T, r, q=0.0, option_type='call', method='lewis'):
        """European option price(s); see ``heston_price`` for the methods."""
        return heston_price(S0, K, T, r, self.params, q=q, option_type=option_type, method=method)

    def implied_vol(self, S0, K, T, r, q=0.0):
        """Black-Scholes implied volatility of the model's OTM options (the model smile)."""
        K_arr, T_arr = np.broadcast_arrays(np.asarray(K, dtype=float), np.asarray(T, dtype=float))
        forward = S0 * np.exp((r - q) * T_arr)
        types = np.where(K_arr >= forward, 'call', 'put')
        prices = self.option_price(S0, K_arr, T_arr, r, q, option_type=types)
        return implied_volatility(prices, S0, K_arr, T_arr, r, q, option_type=types)

    # ── PricingModel interface ───────────────────────────────────────────────
    @staticmethod
    def _year_fraction(market_data: MarketData, instrument: OptionParams) -> float:
        if instrument.expiry is None:
            raise ValueError("Option expiry is required.")
        T = (instrument.expiry - market_data.timestamp).total_seconds() / (365.25 * 24 * 3600)
        if T <= 0:
            raise ValueError("Option has expired")
        return T

    def price(self, market_data: MarketData, instrument: OptionParams) -> float:
        T = self._year_fraction(market_data, instrument)
        return self.option_price(
            market_data.spot_price, instrument.strike_price, T,
            market_data.risk_free_rate, market_data.dividend_yield, instrument.option_type,
        )

    def greeks(self, market_data: MarketData, instrument: OptionParams) -> dict[str, float]:
        """
        Finite-difference Greeks. ``vega`` is the sensitivity to the instantaneous
        volatility sqrt(v0); ``theta`` is per year of calendar time.
        """
        S, K = market_data.spot_price, instrument.strike_price
        r, q, kind = market_data.risk_free_rate, market_data.dividend_yield, instrument.option_type
        T = self._year_fraction(market_data, instrument)

        def px(S_=S, T_=T, r_=r, params=self.params):
            return heston_price(S_, K, T_, r_, params, q=q, option_type=kind)

        h_s, h_t, h_r, h_vol = 1e-3 * S, min(1e-3, T / 2), 1e-4, 1e-4
        base = px()
        up, down = px(S_=S + h_s), px(S_=S - h_s)
        vol = np.sqrt(self.params.v0)
        bumped = [self.params.model_copy(update={'v0': (vol + s * h_vol) ** 2}) for s in (1, -1)]

        return Greeks(
            delta=(up - down) / (2 * h_s),
            gamma=(up - 2 * base + down) / h_s ** 2,
            vega=(px(params=bumped[0]) - px(params=bumped[1])) / (2 * h_vol),
            theta=-(px(T_=T + h_t) - px(T_=T - h_t)) / (2 * h_t),
            rho=(px(r_=r + h_r) - px(r_=r - h_r)) / (2 * h_r),
        ).model_dump()

    # ── simulation ───────────────────────────────────────────────────────────
    def simulate(self, S0, T, r, q=0.0, n_steps=100, n_paths=10_000, seed=None):
        """(times, S, v) paths via the Andersen QE scheme."""
        return simulate_heston(S0, T, r, self.params, q=q, n_steps=n_steps,
                               n_paths=n_paths, seed=seed)

    # ── VIX ──────────────────────────────────────────────────────────────────
    def vix_spot(self) -> float:
        return vix.vix_spot(self.params)

    def vix_future(self, T: float) -> float:
        return vix.vix_future(T, self.params)

    def vix_option_price(self, K, T, r, option_type='call'):
        return vix.vix_option_price(K, T, r, self.params, option_type)
