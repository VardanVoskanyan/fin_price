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
_TINY = np.finfo(float).smallest_subnormal
_EPS = np.finfo(float).eps
_SQRT_2PI = math.sqrt(2 * math.pi)


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


def _discounted(s: float, k: float, t: float, r: float, q: float) -> tuple[float, float]:
    """S e^{-qT} and K e^{-rT}, computed exactly as bs_price does, so that a price which
    rounds onto a no-arbitrage bound compares equal to it."""
    return float(s * np.exp(-q * t)), float(k * np.exp(-r * t))


def no_arbitrage_bounds(s: float, k: float, t: float, r: float, q: float,
                        option_type: str) -> tuple[float, float]:
    """
    Bounds (lower, upper) for a European option price: a call lies in
    (max(S e^{-qT} - K e^{-rT}, 0), S e^{-qT}), a put in (max(K e^{-rT} - S e^{-qT}, 0), K e^{-rT}).
    Only prices strictly inside have an implied volatility.
    """
    disc_s, disc_k = _discounted(s, k, t, r, q)
    if option_type == 'call':
        return max(disc_s - disc_k, 0.0), disc_s
    return max(disc_k - disc_s, 0.0), disc_k


class _OTMQuote:
    """
    One option quote restated as the out-of-the-money option at its strike: a call
    when K >= F, a put below the forward F = S e^{(r-q)T}. Put-call parity moves an
    in-the-money quote across; the OTM price carries the same time value without the
    intrinsic part, so its volatility can be recovered wherever the quote resolves it.

    Build it with :meth:`from_quote`, which returns None when the price is not
    strictly inside the no-arbitrage bounds. ``rounding`` bounds the absolute error
    the target inherits from the quote and the parity conversion: a target within
    twice that of either bound of the OTM price, (0, S e^{-qT}) for a call and
    (0, K e^{-rT}) for a put, does not determine a volatility.
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
        disc_s, disc_k = _discounted(s, k, t, r, q)
        parity = disc_s - disc_k                       # C - P
        lower, upper = no_arbitrage_bounds(s, k, t, r, q, option_type)
        if not lower < price < upper:                  # also rejects NaN
            return None
        otm_type = 'call' if disc_k >= disc_s else 'put'
        if otm_type == option_type:
            target, rounding = price, 4 * _EPS * price
        else:
            # price > lower guarantees target > 0 in floating point
            target = price - parity if option_type == 'call' else price + parity
            rounding = 4 * _EPS * (price + disc_s + disc_k)
        return cls(target, disc_s, disc_k, math.sqrt(t), math.log(s / k) + (r - q) * t,
                   otm_type, rounding)

    def _d1_d2_w(self, sigma):
        w = sigma * self.sqrt_t
        d1 = self.drift / w + 0.5 * w
        return d1, d1 - w, w

    def price(self, sigma):
        """Price of the OTM option at volatility sigma."""
        d1, d2, _ = self._d1_d2_w(sigma)
        if self.otm_type == 'call':
            return float(self.disc_s * ndtr(d1) - self.disc_k * ndtr(d2))
        return float(self.disc_k * ndtr(-d2) - self.disc_s * ndtr(-d1))

    def vega(self, sigma):
        d1, _, _ = self._d1_d2_w(sigma)
        return self.disc_s * math.exp(-0.5 * d1 * d1) / _SQRT_2PI * self.sqrt_t

    def gap(self, sigma):
        """ln(model price) - ln(target): increasing in sigma and close to linear in ln(sigma)."""
        return math.log(max(self.price(sigma), _TINY)) - math.log(self.target)

    def _log_price(self, log_sigma):
        """
        (ln price, d ln price / d ln sigma) at sigma = exp(log_sigma). Where the price is
        below min(1e-250, 1e-10 * target), so far from the root and possibly underflowed,
        both come from the small-volatility asymptote price ~ S e^{-qT} phi(d1) w / (d1 d2),
        w = sigma sqrt(T), which keeps the function smooth for the unbracketed solvers.
        """
        sigma = math.exp(min(max(log_sigma, -100.0), 100.0))
        d1, d2, w = self._d1_d2_w(sigma)
        price = self.price(sigma)
        log_density = -0.5 * d1 * d1 - math.log(_SQRT_2PI)
        if price > min(1e-250, 1e-10 * self.target) or d1 * d2 <= 1.0:
            log_price = math.log(max(price, _TINY))
            return log_price, w * self.disc_s * math.exp(log_density - log_price)
        product = d1 * d2
        log_price = math.log(self.disc_s) + log_density + math.log(w) - math.log(product)
        return log_price, product + 1.0 + (d1 * d1 + d2 * d2) / product

    def log_gap(self, log_sigma):
        """ln(model price) - ln(target) as a function of ln(sigma), for the unbracketed solvers."""
        return self._log_price(log_sigma)[0] - math.log(self.target)

    def log_slope(self, log_sigma):
        """Derivative of :meth:`log_gap`."""
        return self._log_price(log_sigma)[1]

    def brent(self, lo, hi, rtol):
        """Brent's method on :meth:`gap` over a bracket from :meth:`bracket`."""
        return brentq(self.gap, lo, hi, xtol=_TINY, rtol=max(rtol, 4 * _EPS))

    def bracket(self, lo=1e-4, hi=5.0):
        """
        Volatilities (lo, hi) with price(lo) <= target <= price(hi). Starts from the
        given pair and widens geometrically in sigma * sqrt(T) up to the search limits.
        Returns None if the target is not :meth:`resolvable` or cannot be bracketed: it
        is then too close to a bound to be resolved in floating point.
        """
        if not self.resolvable():
            return None
        while self.price(lo) > self.target:
            if lo * self.sqrt_t < _MIN_TOTAL_VOL:
                return None
            lo, hi = lo / 10.0, lo
        while self.price(hi) < self.target:
            if hi * self.sqrt_t > _MAX_TOTAL_VOL:
                return None
            lo, hi = hi, hi * 2.0
        return lo, hi

    def resolvable(self):
        """False if the target is within rounding of a bound of the OTM price."""
        upper = self.disc_s if self.otm_type == 'call' else self.disc_k
        return min(self.target, upper - self.target) > 2.0 * self.rounding

    def residual_ok(self, sigma, tol):
        """
        True if sigma reprices the target to within the price change of a relative
        volatility move of tol (times 10), plus the rounding of the target and of the
        price formula.
        """
        if not (math.isfinite(sigma) and sigma > 0):
            return False
        d1, d2, _ = self._d1_d2_w(sigma)
        sign = 1.0 if self.otm_type == 'call' else -1.0
        terms = self.disc_s * ndtr(sign * d1) + self.disc_k * ndtr(sign * d2)
        allowed = 10.0 * tol * sigma * self.vega(sigma) + 2.0 * self.rounding + 4.0 * _EPS * terms
        return abs(self.price(sigma) - self.target) <= allowed


def implied_volatility(price, s, k, t, r, q=0.0, option_type='call',
                       lo: float = 1e-4, hi: float = 5.0) -> np.ndarray:
    """
    Vectorised Black-Scholes implied volatility. Every argument broadcasts.

    Each quote is restated as the out-of-the-money option at its strike through
    put-call parity, then solved with Brent's method on ln(price). ``lo`` and ``hi``
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
        out[idx] = quote.brent(*bounds, rtol=4 * _EPS)
    return out if out.ndim else float(out)
