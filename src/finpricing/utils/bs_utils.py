"""
Utility module for Black-Scholes calculations
"""
import math

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr
from scipy.stats import norm

OPTION_TYPES = ('call', 'put')

# Search limits for the implied-volatility bracket, in total volatility sigma * sqrt(T).
# Below the minimum every price that is not at the money rounds to zero; above the
# maximum every call price rounds to its upper bound S e^{-qT}.
_MIN_TOTAL_VOL = 1e-15
_MAX_TOTAL_VOL = 100.0
_TINY = 1e-300


def _check_option_type(option_type) -> np.ndarray:
    """Return ``option_type`` as an array, raising ValueError unless every entry is 'call' or 'put'."""
    types = np.asarray(option_type)
    if not np.all((types == 'call') | (types == 'put')):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}.")
    return types


class BlackScholesCalculator:
    """
    Black-Scholes prices and vega without dividends. Every argument may be a scalar
    or an array; arrays broadcast against each other.
    """

    @staticmethod
    def calculate_d1_d2(s, k, t, r, sigma):
        """
        Calculate the d1 and d2 terms for the Black-Scholes model.

        Raises:
            ValueError: If any stock price, strike, maturity or volatility is not positive.
        """
        if any(np.any(np.asarray(x) <= 0) for x in (s, k, t, sigma)):
            raise ValueError("Stock price, strike price, time to maturity, and volatility must be positive.")

        log_money = np.log(np.divide(s, k))
        d1 = (log_money + (r + np.square(sigma) / 2) * t) / (sigma * np.sqrt(t))
        d2 = d1 - sigma * np.sqrt(t)
        return d1, d2

    @staticmethod
    def call_price(s, k, t, r, sigma):
        """Calculate Black-Scholes call price"""
        d1, d2 = BlackScholesCalculator.calculate_d1_d2(s, k, t, r, sigma)
        return s * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2)

    @staticmethod
    def put_price(s, k, t, r, sigma):
        """Calculate Black-Scholes put price"""
        d1, d2 = BlackScholesCalculator.calculate_d1_d2(s, k, t, r, sigma)
        return k * np.exp(-r * t) * norm.cdf(-d2) - s * norm.cdf(-d1)

    @staticmethod
    def vega(s, k, t, r, sigma):
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
    """
    Vectorised Black-Scholes price with a continuous dividend yield. Every argument,
    ``option_type`` included, broadcasts.

    Raises:
        ValueError: If an ``option_type`` entry is not 'call' or 'put'.
    """
    types = _check_option_type(option_type)
    s, k, t, r, sigma, q = (np.asarray(x, dtype=float) for x in (s, k, t, r, sigma, q))
    sqrt_t = np.sqrt(t)
    d1 = (np.log(s / k) + (r - q + 0.5 * sigma ** 2) * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    disc_s, disc_k = s * np.exp(-q * t), k * np.exp(-r * t)
    # Puts from their own formula, not parity: parity cancels digits for deep OTM puts.
    call = disc_s * norm.cdf(d1) - disc_k * norm.cdf(d2)
    put = disc_k * norm.cdf(-d2) - disc_s * norm.cdf(-d1)
    return np.where(types == 'put', put, call)


def bs_vega(s, k, t, r, sigma, q=0.0):
    """Vectorised Black-Scholes vega (same for calls and puts)."""
    s, k, t, r, sigma, q = (np.asarray(x, dtype=float) for x in (s, k, t, r, sigma, q))
    d1 = (np.log(s / k) + (r - q + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    return s * np.exp(-q * t) * norm.pdf(d1) * np.sqrt(t)


class _OTMQuote:
    """
    One option quote restated as the out-of-the-money option at its strike: a call
    when K >= F, a put below the forward F = S e^{(r-q)T}. Put-call parity moves an
    in-the-money quote across; the OTM price carries the same time value without the
    intrinsic part, so its volatility can be recovered wherever the quote resolves it.

    Build it with :meth:`from_quote`, which returns None when the price is not
    strictly inside the no-arbitrage bounds.
    """

    def __init__(self, target, disc_s, disc_k, sqrt_t, drift, otm_type, rounding):
        self.target = target              # OTM price to match
        self.disc_s = disc_s              # S e^{-qT}
        self.disc_k = disc_k              # K e^{-rT}
        self.sqrt_t = sqrt_t
        self.drift = drift                # ln(F / K)
        self.otm_type = otm_type
        self.rounding = rounding          # absolute rounding error carried by target

    @classmethod
    def from_quote(cls, price, s, k, t, r, q, option_type):
        """
        Restate a quote as its OTM option. Returns None if the price is not strictly
        inside the no-arbitrage bounds max(+-(S e^{-qT} - K e^{-rT}), 0) < price <
        S e^{-qT} (calls) or K e^{-rT} (puts), or if S, K or T is not positive.
        """
        price, s, k, t, r, q = (float(x) for x in (price, s, k, t, r, q))
        if not (s > 0 and k > 0 and t > 0):
            return None
        # Same expressions as bs_price, so a price that rounds onto a bound is caught here.
        disc_s, disc_k = float(s * np.exp(-q * t)), float(k * np.exp(-r * t))
        parity = disc_s - disc_k                       # C - P
        if option_type == 'call':
            lower, upper = max(parity, 0.0), disc_s
        else:
            lower, upper = max(-parity, 0.0), disc_k
        if not lower < price < upper:                  # also rejects NaN
            return None
        otm_type = 'call' if disc_k >= disc_s else 'put'
        eps = np.finfo(float).eps
        if otm_type == option_type:
            target, rounding = price, 4 * eps * price
        else:
            # price > lower guarantees target > 0 in floating point
            target = price - parity if option_type == 'call' else price + parity
            rounding = 4 * eps * (price + disc_s + disc_k)
        return cls(target, disc_s, disc_k, math.sqrt(t), math.log(s / k) + (r - q) * t,
                   otm_type, rounding)

    def d1(self, sigma):
        w = sigma * self.sqrt_t
        return self.drift / w + 0.5 * w

    def price(self, sigma):
        """Price of the OTM option at volatility sigma."""
        w = sigma * self.sqrt_t
        d1 = self.drift / w + 0.5 * w
        d2 = d1 - w
        if self.otm_type == 'call':
            return float(self.disc_s * ndtr(d1) - self.disc_k * ndtr(d2))
        return float(self.disc_k * ndtr(-d2) - self.disc_s * ndtr(-d1))

    def vega(self, sigma):
        return float(self.disc_s * norm.pdf(self.d1(sigma)) * self.sqrt_t)

    def log_gap(self, log_sigma):
        """ln(model price) - ln(target) at sigma = exp(log_sigma): increasing, nearly linear."""
        sigma = math.exp(min(max(log_sigma, -100.0), 100.0))
        return math.log(max(self.price(sigma), _TINY)) - math.log(self.target)

    def bracket(self, lo=1e-4, hi=5.0):
        """
        Volatilities (lo, hi) with price(lo) <= target <= price(hi). Starts from the
        given pair and widens geometrically in sigma * sqrt(T) up to the search limits;
        returns None if the target cannot be bracketed (it is then too close to a
        bound to be resolved in floating point).
        """
        while self.price(lo) > self.target:
            if lo * self.sqrt_t < _MIN_TOTAL_VOL:
                return None
            lo, hi = lo / 10.0, lo
        while self.price(hi) < self.target:
            if hi * self.sqrt_t > _MAX_TOTAL_VOL:
                return None
            lo, hi = hi, hi * 2.0
        return lo, hi

    def residual_ok(self, sigma, tol):
        """True if sigma reprices the quote to within vega * tol plus rounding."""
        if not (np.isfinite(sigma) and sigma > 0):
            return False
        error = abs(self.price(sigma) - self.target)
        return error <= 10.0 * tol * sigma * self.vega(sigma) + 16.0 * self.rounding


def implied_volatility(price, s, k, t, r, q=0.0, option_type='call',
                       lo: float = 1e-4, hi: float = 5.0) -> np.ndarray:
    """
    Vectorised Black-Scholes implied volatility. Every argument broadcasts.

    Each quote is restated as the out-of-the-money option at its strike through
    put-call parity, then solved with Brent's method in ln(sigma). ``lo`` and ``hi``
    are the starting bracket only: it widens geometrically in sigma * sqrt(T) until it
    contains the root.

    Returns:
        Implied volatilities (a float for scalar input). NaN where the price is not
        strictly inside the no-arbitrage bounds, or is too close to one to be resolved.

    Raises:
        ValueError: If an ``option_type`` entry is not 'call' or 'put'.
    """
    price, s, k, t, r, q, types = np.broadcast_arrays(
        *(np.asarray(x, dtype=float) for x in (price, s, k, t, r, q)),
        _check_option_type(option_type),
    )
    out = np.full(price.shape, np.nan)
    for idx in np.ndindex(price.shape):
        quote = _OTMQuote.from_quote(price[idx], s[idx], k[idx], t[idx], r[idx], q[idx],
                                     str(types[idx]))
        if quote is None:
            continue
        bounds = quote.bracket(lo, hi)
        if bounds is None:
            continue
        log_sigma = brentq(quote.log_gap, math.log(bounds[0]), math.log(bounds[1]),
                           xtol=1e-14, rtol=4 * np.finfo(float).eps)
        out[idx] = math.exp(log_sigma)
    return out if out.ndim else float(out)
