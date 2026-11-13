"""
Calibration of the Heston model to an index option surface.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import differential_evolution, minimize

from finpricing.calibration.base import Calibrator
from finpricing.calibration.surface import OptionSurface
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

    def _prepare(self, surface: OptionSurface) -> dict:
        s = surface
        iv = implied_volatility(s.price, s.spot, s.strike, s.maturity, s.r, s.q,
                                option_type=s.option_type)
        if np.any(np.isnan(iv)):
            raise ValueError(f"{int(np.isnan(iv).sum())} quotes violate no-arbitrage bounds.")
        vega = np.maximum(bs_vega(s.spot, s.strike, s.maturity, s.r, iv, s.q),
                          self.min_vega * s.spot)
        return {'S0': s.spot, 'r': s.r, 'q': s.q, 'K': s.strike, 'T': s.maturity,
                'price': s.price, 'types': s.option_type, 'iv': iv,
                'scale': s.weight / vega ** 2}

    def _loss(self, x: np.ndarray, d: dict) -> float:
        try:
            model = heston_price(d['S0'], d['K'], d['T'], d['r'], _build_params(x),
                                 q=d['q'], option_type=d['types'])
            value = float(np.sum(d['scale'] * (model - d['price']) ** 2))
            return value if np.isfinite(value) else 1e10
        except (ValueError, FloatingPointError):
            return 1e10

    def objective(self, x: np.ndarray, surface: OptionSurface) -> float:
        """Vega-weighted squared error for parameter vector (v0, kappa, theta, xi, rho)."""
        return self._loss(np.asarray(x, dtype=float), self._prepare(surface))

    def calibrate(self, surface: OptionSurface, **kwargs) -> tuple[HestonParameters, dict]:
        """
        Fit Heston parameters.

        Args:
            surface: Index option quotes, with spot, rate and dividend yield.

        Returns:
            (params, info) where info has 'loss', 'iv_rmse' (model vs market implied
            vol, exact) and 'feller' (whether 2 kappa theta >= xi^2).
        """
        d = self._prepare(surface)
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
