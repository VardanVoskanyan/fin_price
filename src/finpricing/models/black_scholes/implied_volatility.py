"""
Module for Implied Volatility Calculations
"""
import math
from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.optimize import newton, root

from finpricing.utils.bs_utils import OPTION_TYPES, _OTMQuote, no_arbitrage_bounds


class Method(Enum):
    HYBR = 'hybr'
    LM = 'lm'
    BRENTQ = 'brentq'
    NEWTON = 'newton'
    ITERATIVE = 'iterative'


@dataclass
class ImpliedVolatilityParams:
    """
    One option quote and the solver settings.

    Attributes:
        call_value: Option price; a put price when ``option_type='put'`` (the field
            keeps its historical name).
        s, k, t, r: Spot, strike, time to maturity in years, continuous risk-free rate.
        tol: Relative tolerance on the volatility.
        method: Solver; see :class:`Method`.
        guess_method: Starting point for HYBR, LM and NEWTON: 'improved',
            'corrado_miller', or anything else for 0.25.
        q: Continuous dividend yield.
        option_type: 'call' or 'put'.
    """
    call_value: float
    s: float
    k: float
    t: float
    r: float
    tol: float = 1.0e-8
    method: Method = Method.ITERATIVE
    guess_method: str = 'improved'
    q: float = 0.0
    option_type: str = 'call'


class ImpliedVolatilityCalculator:
    """
    Black-Scholes implied volatility of a single call or put, with a dividend yield,
    by one of five solvers.

    Every method works on the same restated problem. The quote is checked against the
    no-arbitrage bounds first (ValueError mentioning "no-arbitrage" unless it is
    strictly inside), then restated by put-call parity as the out-of-the-money option
    at its strike, a put below the forward and a call above it. That option carries
    the same time value without the intrinsic part, so deep in-the-money quotes keep
    whatever volatility information survives rounding. The solvers work in ln(sigma)
    on ln(price), which is close to linear, inside a bracket that widens in
    sigma * sqrt(T) from (1e-4, 5) until it holds the root. Whatever a solver
    returns must reprice the quote to within vega * tol plus rounding; otherwise
    RuntimeError, so no method reports a false success.
    """

    ROOT_METHODS: list[Method] = [
        Method.HYBR,
        Method.LM,
    ]

    ALL_METHODS: list[Method] = ROOT_METHODS + [Method.BRENTQ, Method.NEWTON, Method.ITERATIVE]

    @staticmethod
    def _call_equivalent(params: ImpliedVolatilityParams) -> tuple[float, float, float]:
        """(call price, S e^{-qT}, K e^{-rT}) of the quote, a put converted by parity."""
        disc_s = params.s * np.exp(-params.q * params.t)
        disc_k = params.k * np.exp(-params.r * params.t)
        call = params.call_value
        if params.option_type == 'put':
            call = call + disc_s - disc_k
        return call, disc_s, disc_k

    @staticmethod
    def corrado_miller_guess(params: ImpliedVolatilityParams) -> float:
        """
        Calculate an initial guess for implied volatility using the Corrado-Miller (1996)
        approximation:

            sigma ~ sqrt(2 pi) / (S' + X) * [C - (S' - X)/2 + sqrt((C - (S' - X)/2)^2 - (S' - X)^2 / pi)] / sqrt(T)

        with S' = S exp(-qT), X = K exp(-rT) and C the call price (a put quote is
        converted by parity). The square-root argument is clipped at 0 far from the money.
        """
        call, disc_s, x = ImpliedVolatilityCalculator._call_equivalent(params)
        exp1 = disc_s - x
        exp2 = np.sqrt(2 * np.pi) / (disc_s + x)
        centred = call - exp1 / 2

        guess = exp2 * (centred + np.sqrt(max(0.0, centred ** 2 - exp1 ** 2 / np.pi))) / np.sqrt(params.t)

        return max(guess, 1e-8)

    @staticmethod
    def improved_guess(params: ImpliedVolatilityParams) -> float:
        """
        Calculate an initial guess for implied volatility using an improved approximation
        (Corrado-Miller with a 1.85 correction), with S exp(-qT) in place of S and the
        call price of the quote (a put is converted by parity).
        """
        call, disc_s, x = ImpliedVolatilityCalculator._call_equivalent(params)
        y = 2 * call + x - disc_s
        factor1 = np.sqrt(2 * np.pi) / (2 * (disc_s + x))
        expression = y ** 2 - 1.85 * (disc_s + x) * (x - disc_s) ** 2 / (np.pi * np.sqrt(x * disc_s))
        factor2 = y + np.sqrt(max(0, expression))

        guess = factor1 * factor2 / np.sqrt(params.t)
        return max(guess, 1e-8)

    @staticmethod
    def _prepare(params: ImpliedVolatilityParams) -> tuple[_OTMQuote, float, float]:
        """
        Validate the quote and restate it as its OTM option.

        Returns:
            (quote, lo, hi) with the volatility bracket [lo, hi].

        Raises:
            ValueError: On an unknown option type, non-positive S, K, T or tol, a price
                not strictly inside the no-arbitrage bounds, or one too close to a
                bound to resolve in floating point.
        """
        if params.option_type not in OPTION_TYPES:
            raise ValueError(f"option_type must be 'call' or 'put', got {params.option_type!r}.")
        if not (params.s > 0 and params.k > 0 and params.t > 0 and params.tol > 0):
            raise ValueError("Stock price, strike price, time to maturity and tol must be positive.")
        quote = _OTMQuote.from_quote(params.call_value, params.s, params.k, params.t,
                                     params.r, params.q, params.option_type)
        if quote is None:
            lower, upper = no_arbitrage_bounds(params.s, params.k, params.t, params.r,
                                               params.q, params.option_type)
            raise ValueError(
                f"{params.option_type} price {params.call_value!r} violates the no-arbitrage "
                f"bounds: it must lie strictly between {lower!r} and {upper!r}.")
        bounds = quote.bracket()
        if bounds is None:
            raise ValueError(
                f"{params.option_type} price {params.call_value!r} is too close to a "
                "no-arbitrage bound for its volatility to be resolved in floating point.")
        return quote, *bounds

    @staticmethod
    def _checked(quote: _OTMQuote, sigma: float, params: ImpliedVolatilityParams) -> float:
        """Return sigma if it reprices the quote, else raise RuntimeError."""
        if not quote.residual_ok(sigma, params.tol):
            raise RuntimeError(f"{params.method.name} did not converge (last volatility {sigma!r}).")
        return float(sigma)

    @staticmethod
    def iterative_method(params: ImpliedVolatilityParams) -> float:
        """
        Calculate implied volatility by bisection in ln(sigma) on the bracket, until it
        is narrower than tol (relative).
        """
        quote, lo, hi = ImpliedVolatilityCalculator._prepare(params)
        a, b = math.log(lo), math.log(hi)
        for _ in range(300):
            if b - a <= params.tol:
                break
            mid = 0.5 * (a + b)
            gap = quote.price(math.exp(mid)) - quote.target
            if gap == 0:
                a = b = mid
            elif gap > 0:
                b = mid
            else:
                a = mid
        return ImpliedVolatilityCalculator._checked(quote, math.exp(0.5 * (a + b)), params)

    @staticmethod
    def root_finding_method(params: ImpliedVolatilityParams) -> float:
        """
        Calculate implied volatility with a scipy solver on ln(price) - ln(target) as a
        function of ln(sigma): Brent on the bracket, or HYBR, LM and Newton from the
        initial guess (the geometric middle of the bracket if the guess falls outside).
        """
        quote, lo, hi = ImpliedVolatilityCalculator._prepare(params)

        match params.guess_method:
            case 'improved':
                x0 = ImpliedVolatilityCalculator.improved_guess(params)
            case 'corrado_miller':
                x0 = ImpliedVolatilityCalculator.corrado_miller_guess(params)
            case _:
                x0 = 0.25
        if not (np.isfinite(x0) and lo < x0 < hi):
            x0 = math.sqrt(lo * hi)
        u0 = math.log(x0)

        if params.method in ImpliedVolatilityCalculator.ROOT_METHODS:
            result = root(lambda u: [quote.log_gap(u[0])], [u0], tol=params.tol,
                          method=params.method.value)
            u = result.x[0]
        elif params.method == Method.BRENTQ:
            u = math.log(quote.brent(lo, hi, rtol=params.tol))
        elif params.method == Method.NEWTON:
            try:
                u = newton(quote.log_gap, u0, fprime=quote.log_slope, tol=params.tol, maxiter=100)
            except (RuntimeError, ArithmeticError) as exc:
                raise RuntimeError(f"NEWTON did not converge: {exc}") from exc
        else:
            raise ValueError(f"Method {params.method} not supported")
        return ImpliedVolatilityCalculator._checked(quote, math.exp(u), params)

    @classmethod
    def implied_volatility(cls, params: ImpliedVolatilityParams) -> float:
        """
        Implied volatility of the quote in ``params`` by the chosen method.

        Raises:
            ValueError: If the price is not strictly inside the no-arbitrage bounds (the
                message says "no-arbitrage"), the inputs are invalid, or the method is
                unknown.
            RuntimeError: If the solver stops at a volatility that does not reprice the
                quote.
        """
        if params.method == Method.ITERATIVE:
            return cls.iterative_method(params)
        elif params.method in cls.ALL_METHODS:
            return cls.root_finding_method(params)
        else:
            raise ValueError(f"Method {params.method} not supported")
