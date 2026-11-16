"""
Module for Implied Volatility Calculations
"""
from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.optimize import brentq, newton, root

from finpricing.utils.bs_utils import BlackScholesCalculator


class Method(Enum):
    HYBR = 'hybr'
    LM = 'lm'
    BRENTQ = 'brentq'
    NEWTON = 'newton'
    ITERATIVE = 'iterative'


@dataclass
class ImpliedVolatilityParams:
    call_value: float
    s: float
    k: float
    t: float
    r: float
    tol: float = 1.0e-8
    method: Method = Method.ITERATIVE
    guess_method: str = 'improved'


class ImpliedVolatilityCalculator:
    """
    A class to calculate the implied volatility of options using various methods.
    """

    ROOT_METHODS: list[Method] = [
        Method.HYBR,
        Method.LM,
    ]

    ALL_METHODS: list[Method] = ROOT_METHODS + [Method.BRENTQ, Method.NEWTON, Method.ITERATIVE]

    @staticmethod
    def corrado_miller_guess(params: ImpliedVolatilityParams) -> float:
        """
        Calculate an initial guess for implied volatility using the Corrado-Miller approximation.
        """
        exp1 = params.s - params.k * np.exp(-params.r * params.t)
        exp2 = np.sqrt(2 * np.pi) / (params.s + params.k * np.exp(-params.r * params.t))

        guess = (exp2 + (params.call_value - exp1 / 2) +
                 np.sqrt((params.call_value - exp1 / 2) ** 2 - exp1 ** 2 / np.pi)) / np.sqrt(params.t)

        return max(guess, 1e-8)

    @staticmethod
    def improved_guess(params: ImpliedVolatilityParams) -> float:
        """
        Calculate an initial guess for implied volatility using an improved approximation.
        """
        x = params.k * np.exp(-params.r * params.t)
        y = 2 * params.call_value + x - params.s
        factor1 = np.sqrt(2 * np.pi) / (2 * (params.s + x))
        expression = y ** 2 - 1.85 * (params.s + x) * (x - params.s) ** 2 / (np.pi * np.sqrt(x * params.s))
        factor2 = y + np.sqrt(max(0, expression))

        guess = factor1 * factor2 / np.sqrt(params.t)
        return max(guess, 1e-8)

    @staticmethod
    def iterative_method(params: ImpliedVolatilityParams) -> float:
        """
        Calculate implied volatility using an iterative method.
        """
        vol = 0.5
        max_iter = 300
        upper, lower = 5.0, 0.0

        for _ in range(max_iter):
            price = BlackScholesCalculator.call_price(params.s, params.k, params.t, params.r, vol)
            if abs(price - params.call_value) < max(price, params.call_value) * params.tol:
                return vol

            if price > params.call_value:
                upper = vol
                vol = (vol + lower) / 2
            else:
                lower = vol
                vol = (upper + vol) / 2

        raise ValueError("Iterative method failed to converge")

    @staticmethod
    def root_finding_method(params: ImpliedVolatilityParams) -> float:
        """
        Calculate implied volatility using a root-finding method.
        """
        def value_diff(vol: float) -> float:
            return BlackScholesCalculator.call_price(params.s, params.k, params.t, params.r, vol) - params.call_value

        match params.guess_method:
            case 'improved':
                x0 = ImpliedVolatilityCalculator.improved_guess(params)
            case 'corrado_miller':
                x0 = ImpliedVolatilityCalculator.corrado_miller_guess(params)
            case _:
                x0 = 0.25

        if params.method in ImpliedVolatilityCalculator.ROOT_METHODS:
            result = root(value_diff, x0, tol=params.tol, method=params.method.value)
            return result.x[0] if result.success else None
        elif params.method == Method.BRENTQ:
            return brentq(value_diff, 1e-12, 5.0, xtol=params.tol)
        elif params.method == Method.NEWTON:
            def fprime(vol: float) -> float:
                return max(BlackScholesCalculator.vega(params.s, params.k, params.t, params.r, vol), 1e-8)

            return newton(value_diff, x0, fprime=fprime, tol=params.tol)
        else:
            raise ValueError(f"Method {params.method} not supported")

    @classmethod
    def implied_volatility(cls, params: ImpliedVolatilityParams) -> float:
        """
        Main function to calculate implied volatility using the chosen method.
        """
        if params.method == Method.ITERATIVE:
            return cls.iterative_method(params)
        elif params.method in cls.ALL_METHODS:
            return cls.root_finding_method(params)
        else:
            raise ValueError(f"Method {params.method} not supported")
