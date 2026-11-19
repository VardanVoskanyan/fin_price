"""
Calibration of the Heston model to an index option surface.
"""
from __future__ import annotations

import warnings
from collections.abc import Mapping
from types import MappingProxyType

import numpy as np
from scipy.optimize import differential_evolution, minimize

from finpricing.calibration.base import Calibrator
from finpricing.calibration.surface import OptionSurface
from finpricing.models.heston.pricing import heston_price
from finpricing.parameters import HestonParameters
from finpricing.utils.bs_utils import bs_vega, implied_volatility

_NAMES = ['v0', 'kappa', 'theta', 'xi', 'rho']

DEFAULT_BOUNDS: Mapping[str, tuple[float, float]] = MappingProxyType({
    'v0': (1e-4, 1.0),
    'kappa': (1e-2, 10.0),
    'theta': (1e-4, 1.0),
    'xi': (1e-2, 3.0),
    'rho': (-0.99, 0.99),
})

# Loss returned when the pricer fails or returns non-finite prices: large and finite,
# so the optimisers steer away from such points instead of breaking on inf/NaN.
_FAILED_LOSS = 1e10

# A parameter within this fraction of its bound interval's width of a bound is "at" it.
_AT_BOUND_RTOL = 1e-4


def _build_params(x: np.ndarray) -> HestonParameters:
    return HestonParameters(**dict(zip(_NAMES, map(float, x), strict=True)))


def _resolve_bounds(overrides: Mapping[str, tuple[float, float]] | None
                    ) -> dict[str, tuple[float, float]]:
    """Merge user bounds into the defaults, checking names and values."""
    bounds = dict(DEFAULT_BOUNDS)
    for name, pair in (overrides or {}).items():
        if name not in bounds:
            raise ValueError(f"Unknown Heston parameter {name!r} in bounds; "
                             f"expected one of {_NAMES}.")
        try:
            lo, hi = map(float, pair)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Bounds for {name!r} must be a (lower, upper) pair, "
                             f"got {pair!r}.") from exc
        if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
            raise ValueError(f"Bounds for {name!r} must be finite with lower < upper, "
                             f"got ({lo}, {hi}).")
        for value in (lo, hi):
            try:
                HestonParameters(**{name: value})
            except ValueError as exc:
                raise ValueError(f"Bound {value} is not a valid value of {name!r}.") from exc
        bounds[name] = (lo, hi)
    return bounds


class HestonCalibrator(Calibrator):
    """
    Calibrates Heston parameters to an option surface.

    The loss is the sum of squared price errors divided by the market Black-Scholes
    vega, a first-order approximation of implied-volatility errors that avoids
    inverting Black-Scholes at every iteration. Global differential evolution is
    followed by L-BFGS-B refinement, both inside the box ``bounds``.

    Args:
        bounds: Search box, as ``{name: (lower, upper)}`` for any of 'v0', 'kappa',
            'theta', 'xi', 'rho'. Names left out keep their ``DEFAULT_BOUNDS``.
            Short-dated equity surfaces can need a larger upper bound on 'kappa' or
            'xi'; ``info['at_bounds']`` says when a bound is binding.
        de_maxiter: Max iterations for differential evolution.
        de_popsize: Population size multiplier for differential evolution.
        min_vega: Floor on the vega denominator, relative to spot (stops deep
            OTM quotes dominating the fit).
        seed: Random seed for differential evolution.
        verbose: Print progress.

    Raises:
        ValueError: If ``bounds`` names an unknown parameter, or a pair is not finite,
            not increasing, or outside the parameter's domain (e.g. a non-positive
            lower bound on 'kappa').
    """

    def __init__(self, bounds: Mapping[str, tuple[float, float]] | None = None,
                 de_maxiter: int = 100, de_popsize: int = 15,
                 min_vega: float = 1e-3, seed: int = 42, verbose: bool = False):
        self.bounds = _resolve_bounds(bounds)
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

    def _model_prices(self, x: np.ndarray, d: dict) -> np.ndarray:
        return heston_price(d['S0'], d['K'], d['T'], d['r'], _build_params(x),
                            q=d['q'], option_type=d['types'])

    def _loss(self, x: np.ndarray, d: dict) -> float:
        try:
            value = float(np.sum(d['scale'] * (self._model_prices(x, d) - d['price']) ** 2))
        except (ValueError, FloatingPointError):
            return _FAILED_LOSS
        return value if np.isfinite(value) else _FAILED_LOSS

    def objective(self, x: np.ndarray, surface: OptionSurface) -> float:
        """Vega-weighted squared error for parameter vector (v0, kappa, theta, xi, rho)."""
        return self._loss(np.asarray(x, dtype=float), self._prepare(surface))

    def calibrate(self, surface: OptionSurface) -> tuple[HestonParameters, dict]:
        """
        Fit Heston parameters.

        Args:
            surface: Index option quotes, with spot, rate and dividend yield.

        Returns:
            ``(params, info)``. Besides the keys every calibrator reports (see
            ``Calibrator.calibrate``), info has:

            - ``iv_rmse``, ``max_iv_error``: root-mean-square and largest absolute
              difference between model and market implied vol (exact Black-Scholes
              inversion), over the quotes whose model implied vol could be computed.
            - ``n_iv_failed``: number of quotes whose model implied vol could not be
              computed (model price outside the no-arbitrage bounds). Non-zero makes
              ``success`` False, since the two IV statistics leave those quotes out.
            - ``feller``: whether 2 kappa theta >= xi^2.

            ``success`` is False if L-BFGS-B did not report convergence, a parameter
            ended at a bound (``at_bounds``), or ``n_iv_failed > 0``; ``message``
            lists every reason.

        Raises:
            ValueError: If a market price has no implied volatility.
            RuntimeError: If the pricer fails, or returns non-finite prices, at the
                best parameters found (e.g. it failed at every point tried).
        """
        d = self._prepare(surface)
        box = [self.bounds[name] for name in _NAMES]
        with np.errstate(all='ignore'):
            de = differential_evolution(self._loss, box, args=(d,), maxiter=self.de_maxiter,
                                        popsize=self.de_popsize, seed=self.seed, tol=1e-8,
                                        polish=False, disp=self.verbose)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                local = minimize(self._loss, de.x, args=(d,), method='L-BFGS-B', bounds=box,
                                 options={'ftol': 1e-14, 'gtol': 1e-10, 'maxiter': 500})
        best = local if local.fun < de.fun else de
        params = _build_params(best.x)

        # Re-price at the optimum without the failure sentinel: the optimisers only
        # ever saw _FAILED_LOSS, so this is where a pricer failure must surface.
        try:
            model_prices = np.asarray(self._model_prices(best.x, d), dtype=float)
        except (ValueError, FloatingPointError) as exc:
            raise RuntimeError(f"Heston calibration failed: the pricer fails at the best "
                               f"parameters found ({params}).") from exc
        if not np.all(np.isfinite(model_prices)):
            raise RuntimeError(f"Heston calibration failed: non-finite model prices at the "
                               f"best parameters found ({params}).")
        loss = float(np.sum(d['scale'] * (model_prices - d['price']) ** 2))

        model_iv = np.asarray(implied_volatility(model_prices, d['S0'], d['K'], d['T'], d['r'],
                                                 d['q'], option_type=d['types']), dtype=float)
        iv_error = np.abs(model_iv - d['iv'])
        iv_ok = np.isfinite(iv_error)
        n_iv_failed = int((~iv_ok).sum())
        if iv_ok.any():
            iv_rmse = float(np.sqrt(np.mean(iv_error[iv_ok] ** 2)))
            max_iv_error = float(iv_error[iv_ok].max())
        else:
            iv_rmse = max_iv_error = float('nan')

        at_bounds = [name for name, value, (lo, hi) in zip(_NAMES, best.x, box, strict=True)
                     if min(value - lo, hi - value) <= _AT_BOUND_RTOL * (hi - lo)]

        problems = []
        if not local.success:
            problems.append(f"L-BFGS-B did not converge: {local.message}")
        if at_bounds:
            problems.append(f"parameters at a bound of the search box: {', '.join(at_bounds)} "
                            f"(widen them with HestonCalibrator(bounds=...))")
        if n_iv_failed:
            problems.append(f"model implied vol could not be computed for {n_iv_failed} of "
                            f"{len(surface)} quotes; iv_rmse and max_iv_error exclude them")

        info = {
            'success': not problems,
            'message': '; '.join(problems) if problems else 'Calibration converged.',
            'loss': loss,
            'n_quotes': len(surface),
            'at_bounds': at_bounds,
            'iv_rmse': iv_rmse,
            'max_iv_error': max_iv_error,
            'n_iv_failed': n_iv_failed,
            'feller': params.feller_satisfied,
        }
        if self.verbose:
            print(f"Heston calibration: loss={loss:.3e}, IV RMSE={iv_rmse:.2e}, "
                  f"{info['message']}")
        return params, info
