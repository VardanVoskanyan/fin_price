"""
Utility module for Black-Scholes calculations
"""
import numpy as np
from scipy.stats import norm


class BlackScholesCalculator:
    """Utility class for Black-Scholes calculations."""

    @staticmethod
    def calculate_d1_d2(s: float, k: float, t: float, r: float, sigma: float):
        """
        Calculate the d1 and d2 terms for the Black-Scholes model.
        """
        if np.any(np.asarray(s) <= 0) or k <= 0 or t <= 0 or np.any(np.asarray(sigma) <= 0):
            raise ValueError("Stock price, strike price, time to maturity, and volatility must be positive.")

        log_money = np.log(s / k)
        d1 = (log_money + (r + sigma ** 2 / 2) * t) / (sigma * np.sqrt(t))
        d2 = d1 - sigma * np.sqrt(t)
        return d1, d2

    @staticmethod
    def call_price(s: float, k: float, t: float, r: float, sigma: float) -> float:
        """Calculate Black-Scholes call price"""
        d1, d2 = BlackScholesCalculator.calculate_d1_d2(s, k, t, r, sigma)
        return s * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2)

    @staticmethod
    def put_price(s: float, k: float, t: float, r: float, sigma: float) -> float:
        """Calculate Black-Scholes put price"""
        d1, d2 = BlackScholesCalculator.calculate_d1_d2(s, k, t, r, sigma)
        return k * np.exp(-r * t) * norm.cdf(-d2) - s * norm.cdf(-d1)

    @staticmethod
    def vega(s: float, k: float, t: float, r: float, sigma: float) -> float:
        """Calculate Vega of the option"""
        d1, _ = BlackScholesCalculator.calculate_d1_d2(s, k, t, r, sigma)
        return s * norm.pdf(d1) * np.sqrt(t)


# Convenience functions for backward compatibility with notebook code
def bs_call(s: float, k: float, t: float, r: float, sigma: float) -> float:
    """
    Black-Scholes call option price.
    
    Args:
        s: Current stock price
        k: Strike price
        t: Time to maturity
        r: Risk-free rate
        sigma: Volatility
        
    Returns:
        Call option price
    """
    return BlackScholesCalculator.call_price(s, k, t, r, sigma)


def bs_put(s: float, k: float, t: float, r: float, sigma: float) -> float:
    """
    Black-Scholes put option price.
    
    Args:
        s: Current stock price
        k: Strike price  
        t: Time to maturity
        r: Risk-free rate
        sigma: Volatility
        
    Returns:
        Put option price
    """
    return BlackScholesCalculator.put_price(s, k, t, r, sigma)




def bs_price(s, k, t, r, sigma, q=0.0, option_type='call'):
    """Vectorised Black-Scholes price with a continuous dividend yield."""
    s, k, t, sigma = (np.asarray(x, dtype=float) for x in (s, k, t, sigma))
    sqrt_t = np.sqrt(t)
    d1 = (np.log(s / k) + (r - q + 0.5 * sigma ** 2) * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    call = s * np.exp(-q * t) * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2)
    put = call - s * np.exp(-q * t) + k * np.exp(-r * t)
    return np.where(np.asarray(option_type) == 'put', put, call)


def bs_vega(s, k, t, r, sigma, q=0.0):
    """Vectorised Black-Scholes vega (same for calls and puts)."""
    s, k, t, sigma = (np.asarray(x, dtype=float) for x in (s, k, t, sigma))
    d1 = (np.log(s / k) + (r - q + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    return s * np.exp(-q * t) * norm.pdf(d1) * np.sqrt(t)


def implied_volatility(price, s, k, t, r, q=0.0, option_type='call',
                       lo: float = 1e-4, hi: float = 5.0) -> np.ndarray:
    """
    Vectorised implied volatility (Brent's method per option).
    Returns NaN where the price is outside the no-arbitrage bounds.
    """
    from scipy.optimize import brentq

    price, s, k, t, types = np.broadcast_arrays(
        np.asarray(price, dtype=float), np.asarray(s, dtype=float),
        np.asarray(k, dtype=float), np.asarray(t, dtype=float), np.asarray(option_type),
    )
    out = np.full(price.shape, np.nan)
    for idx in np.ndindex(price.shape):
        args = (s[idx], k[idx], t[idx], r, q, types[idx])

        def diff(sig, p=price[idx], a=args):
            return float(bs_price(a[0], a[1], a[2], a[3], sig, a[4], a[5])) - p

        if diff(lo) <= 0 <= diff(hi):
            out[idx] = brentq(diff, lo, hi, xtol=1e-12)
    return out if out.ndim else float(out)
