"""
Black Scholes model implementations
"""
from typing import Any

import numpy as np
from scipy.stats import norm

from finpricing.base import Greeks, MarketData, ModelConfig, OptionParams, PricingModel
from finpricing.utils.bs_utils import bs_price

_SECONDS_PER_YEAR = 365.25 * 24 * 3600


class BlackScholesModel(PricingModel):
    """Black-Scholes option pricing model with a continuous dividend yield."""

    def __init__(self, config: ModelConfig):
        self.config = config

    @property
    def sigma(self) -> float:
        return self.config.numerical_params.get('volatility', 0.2)

    @staticmethod
    def _year_fraction(market_data: MarketData, instrument: OptionParams) -> float:
        if not isinstance(instrument, OptionParams):
            raise ValueError("Black-Scholes model requires OptionParams")
        if instrument.expiry is None:
            raise ValueError("Option expiry is required.")
        T = (instrument.expiry - market_data.timestamp).total_seconds() / _SECONDS_PER_YEAR
        if T <= 0:
            raise ValueError("Option has expired")
        return T

    def price(self, market_data: MarketData, instrument: OptionParams) -> float:
        """
        Calculate the price of an option using the Black-Scholes model.

        Raises:
            ValueError: If the option has expired or invalid parameters are provided
        """
        T = self._year_fraction(market_data, instrument)
        return float(bs_price(
            market_data.spot_price, instrument.strike_price, T, market_data.risk_free_rate,
            self.sigma, q=market_data.dividend_yield, option_type=instrument.option_type,
        ))

    def greeks(self, market_data: MarketData, instrument: OptionParams) -> dict[str, Any]:
        """
        Analytic Greeks. ``vega`` and ``rho`` are per unit change of volatility and rate;
        ``theta`` is the change in value per year of calendar time.
        """
        T = self._year_fraction(market_data, instrument)
        S, K, sigma = market_data.spot_price, instrument.strike_price, self.sigma
        r, q = market_data.risk_free_rate, market_data.dividend_yield
        sqrt_t = np.sqrt(T)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_t)
        d2 = d1 - sigma * sqrt_t
        disc_s, disc_k = S * np.exp(-q * T), K * np.exp(-r * T)

        gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * sqrt_t)
        vega = disc_s * norm.pdf(d1) * sqrt_t
        decay = -disc_s * norm.pdf(d1) * sigma / (2 * sqrt_t)
        if instrument.option_type == 'call':
            delta = np.exp(-q * T) * norm.cdf(d1)
            theta = decay - r * disc_k * norm.cdf(d2) + q * disc_s * norm.cdf(d1)
            rho = T * disc_k * norm.cdf(d2)
        else:
            delta = -np.exp(-q * T) * norm.cdf(-d1)
            theta = decay + r * disc_k * norm.cdf(-d2) - q * disc_s * norm.cdf(-d1)
            rho = -T * disc_k * norm.cdf(-d2)

        return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho).model_dump()
