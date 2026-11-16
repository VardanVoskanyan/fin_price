"""
Characteristic functions for various models
"""
import numpy as np
import scipy.integrate as integ

from finpricing.parameters import (
    VIXBlackScholesModelParameters,
    VIXKouModelParameters,
    VIXMertonModelParameters,
)


def black_scholes_characteristic_function(u: complex, T: float, sigma: float) -> complex:
    """Characteristic function for the Black-Scholes model."""
    return np.exp(-sigma**2 * T * (u**2 + 1j * u) / 2)


def compute_terms(u: complex, T: float, omega: float, k1: float, cf2: float):
    """
    Compute the first and second terms in characteristic function formulas.
    """
    first_term = -cf2 * omega**2 * 1j * u * (1 - np.exp(-2 * k1 * T)) / (2 * k1)
    second_term = -cf2 * omega**2 * u**2 * (1 - np.exp(-2 * k1 * T)) / (2 * k1)
    return first_term, second_term


def merton_characteristic_function(
    u: complex, T: float, params: VIXMertonModelParameters, cf1: float, cf2: float
) -> complex:
    """Characteristic function for the Merton model."""
    first_term, second_term = compute_terms(u, T, params.omega, params.k1, cf2)

    integral1 = integ.quad_vec(
        lambda s: np.exp(
            cf1 * np.exp(-params.k2 * (T - s)) * params.m
            + cf2 * np.exp(-2 * params.k2 * (T - s)) * params.delta**2
        )
        - 1,
        0,
        T,
    )[0]

    integral2 = integ.quad_vec(
        lambda s: np.exp(
            cf1 * 1j * u * np.exp(-params.k2 * (T - s)) * params.m
            - cf2 * u**2 * np.exp(-2 * params.k2 * (T - s)) * params.delta**2
        )
        - 1,
        0,
        T,
    )[0]

    return np.exp(first_term + second_term - 1j * u * params.lmbd * integral1 + params.lmbd * integral2)


def kou_characteristic_function(
    u: complex, T: float, params: VIXKouModelParameters, cf1: float, cf2: float
) -> complex:
    """Characteristic function for the Kou model."""
    first_term, second_term = compute_terms(u, T, params.omega, params.k1, cf2)

    integral1 = integ.quad_vec(
        lambda s: params.p * params.alpha_plus / (params.alpha_plus - cf1 * np.exp(-params.k2 * (T - s)))
        + (1 - params.p) * params.alpha_minus / (params.alpha_minus + cf1 * np.exp(-params.k2 * (T - s)))
        - 1,
        0,
        T,
    )[0]

    integral2 = integ.quad_vec(
        lambda s: params.p * params.alpha_plus / (params.alpha_plus - cf1 * 1j * u * np.exp(-params.k2 * (T - s)))
        + (1 - params.p) * params.alpha_minus / (params.alpha_minus + cf1 * 1j * u * np.exp(-params.k2 * (T - s)))
        - 1,
        0,
        T,
    )[0]

    return np.exp(first_term + second_term - 1j * u * params.lmbd * integral1 + params.lmbd * integral2)


def characteristic_function(
    u: complex,
    T: float,
    params: VIXBlackScholesModelParameters | VIXMertonModelParameters | VIXKouModelParameters,
    asset_type: str = 'VIX',
) -> complex:
    """
    General characteristic function dispatcher.

    Args:
        u: Complex frequency parameter.
        T: Time to maturity.
        params: Model parameters (VIX Black-Scholes, Merton, or Kou).
        asset_type: Asset type ('VIX' or 'var_swap').
    """
    match asset_type:
        case 'VIX':
            cf1, cf2 = 0.5, 1 / 8
        case 'var_swap':
            cf1, cf2 = 1.0, 0.5
        case _:
            raise ValueError(f"Unknown asset type: {asset_type}")

    match params:
        case VIXBlackScholesModelParameters():
            return black_scholes_characteristic_function(u, T, params.sigma)
        case VIXMertonModelParameters():
            return merton_characteristic_function(u, T, params, cf1, cf2)
        case VIXKouModelParameters():
            return kou_characteristic_function(u, T, params, cf1, cf2)
        case _:
            raise ValueError(f"Unsupported model type: {params.model_type}")
