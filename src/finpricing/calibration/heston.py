"""
Calibration of the Heston model to an index option surface.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize

from finpricing.calibration.base import Calibrator
from finpricing.models.heston.pricing import heston_price
from finpricing.parameters import HestonParameters
from finpricing.utils.bs_utils import bs_vega, implied_volatility

_NAMES = ['v0', 'kappa', 'theta', 'xi', 'rho']
_BOUNDS: list[tuple[float, float]] = [
    (1e-4, 1.0),     # v0
    (1e-2, 10.0),    # kappa
    (1e-4, 1.0),     # theta
    (1e-2, 3.0),     # xi
    (-0.99, 0.99),   # rho
]


def _build_params(x: np.ndarray) -> HestonParameters:
    return HestonParameters(**dict(zip(_NAMES, map(float, x), strict=True)))


class HestonCalibrator(Calibrator):
    """
    Calibrates Heston parameters to an option surface.

    The loss is the sum of squared price errors divided by the market Black-Scholes
    vega, a first-order approximation of implied-volatility errors that avoids
    inverting Black-Scholes at every iteration. Global differential evolution is
    followed by L-BFGS-B refinement.

    Args:
        de_maxiter: Max iterations for differential evolution.
        de_popsize: Population size multiplier for differential evolution.
        min_vega: Floor on the vega denominator, relative to spot (stops deep
            OTM quotes dominating the fit).
        seed: Random seed for differential evolution.
        verbose: Print progress.
    """

    def __init__(self, de_maxiter: int = 100, de_popsize: int = 15,
                 min_vega: float = 1e-3, seed: int = 42, verbose: bool = False):
        self.de_maxiter = de_maxiter
        self.de_popsize = de_popsize
        self.min_vega = min_vega
        self.seed = seed
        self.verbose = verbose

    def _prepare(self, option_surface: pd.DataFrame) -> dict:
        surface = self._validate_surface(option_surface)
        if 'S0' not in surface.columns:
            raise ValueError("option_surface needs an 'S0' column (spot price).")
        S0 = float(surface['S0'].iloc[0])
        r = float(surface['r'].iloc[0]) if 'r' in surface.columns else 0.0
        q = float(surface['q'].iloc[0]) if 'q' in surface.columns else 0.0
        K = surface['strike'].to_numpy(float)
        T = surface['maturity'].to_numpy(float)
        price = surface['price'].to_numpy(float)
        types = surface['option_type'].to_numpy(str)

        iv = implied_volatility(price, S0, K, T, r, q, option_type=types)
        if np.any(np.isnan(iv)):
            raise ValueError(f"{int(np.isnan(iv).sum())} quotes violate no-arbitrage bounds.")
        vega = np.maximum(bs_vega(S0, K, T, r, iv, q), self.min_vega * S0)
        return {'S0': S0, 'r': r, 'q': q, 'K': K, 'T': T, 'price': price, 'types': types,
                'iv': iv, 'scale': surface['weight'].to_numpy(float) / vega ** 2}

    def _loss(self, x: np.ndarray, d: dict) -> float:
        try:
            model = heston_price(d['S0'], d['K'], d['T'], d['r'], _build_params(x),
                                 q=d['q'], option_type=d['types'])
            value = float(np.sum(d['scale'] * (model - d['price']) ** 2))
            return value if np.isfinite(value) else 1e10
        except (ValueError, FloatingPointError):
            return 1e10

    def objective(self, x: np.ndarray, option_surface: pd.DataFrame) -> float:
        """Vega-weighted squared error for parameter vector (v0, kappa, theta, xi, rho)."""
        return self._loss(np.asarray(x, dtype=float), self._prepare(option_surface))

    def calibrate(self, option_surface: pd.DataFrame, **kwargs) -> tuple[HestonParameters, dict]:
        """
        Fit Heston parameters.

        Args:
            option_surface: DataFrame with columns 'strike', 'maturity', 'price', 'S0'.
                Optional: 'option_type' (default 'call'), 'weight' (default 1.0),
                'r' and 'q' (default 0.0).

        Returns:
            (params, info) where info has 'loss', 'iv_rmse' (model vs market implied
            vol, exact) and 'feller' (whether 2 kappa theta >= xi^2).
        """
        d = self._prepare(option_surface)
        with np.errstate(all='ignore'):
            de = differential_evolution(self._loss, _BOUNDS, args=(d,), maxiter=self.de_maxiter,
                                        popsize=self.de_popsize, seed=self.seed, tol=1e-8,
                                        polish=False, disp=self.verbose)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                local = minimize(self._loss, de.x, args=(d,), method='L-BFGS-B', bounds=_BOUNDS,
                                 options={'ftol': 1e-14, 'gtol': 1e-10, 'maxiter': 500})
        best = local if local.fun < de.fun else de
        params = _build_params(best.x)

        model_prices = heston_price(d['S0'], d['K'], d['T'], d['r'], params,
                                    q=d['q'], option_type=d['types'])
        model_iv = implied_volatility(model_prices, d['S0'], d['K'], d['T'], d['r'], d['q'],
                                      option_type=d['types'])
        info = {
            'loss': float(best.fun),
            'iv_rmse': float(np.sqrt(np.nanmean((model_iv - d['iv']) ** 2))),
            'feller': params.feller_satisfied,
        }
        if self.verbose:
            print(f"Heston calibration: loss={info['loss']:.3e}, IV RMSE={info['iv_rmse']:.2e}")
        return params, info
