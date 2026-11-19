"""
Calibration of the Cont-Kokholm model to an observed VIX option surface.
"""
from __future__ import annotations

import warnings
from collections.abc import Mapping

import numpy as np
from scipy.optimize import differential_evolution, minimize

from finpricing.calibration.base import Calibrator
from finpricing.calibration.surface import OptionSurface
from finpricing.models.vix.pricing import price_vix_option_slice
from finpricing.parameters import VIXKouModelParameters, VIXMertonModelParameters
from finpricing.utils.fourier import FourierMethod

# Default search box: (lower, upper) for each fitted parameter. rho is not fitted: VIX
# options do not depend on it, so it is held at the value given to the calibrator.
_MERTON_BOUNDS: list[tuple[float, float]] = [
    (0.1,  10.0),   # omega
    (0.1,  50.0),   # lmbd
    (1.0,  100.0),  # k1
    (0.1,  50.0),   # k2
    (-2.0,  2.0),   # m
    (0.01,  3.0),   # delta
]
_MERTON_NAMES = ['omega', 'lmbd', 'k1', 'k2', 'm', 'delta']

_KOU_BOUNDS: list[tuple[float, float]] = [
    (0.1,  10.0),   # omega
    (0.1, 100.0),   # lmbd
    (1.0, 100.0),   # k1
    (0.1,  50.0),   # k2
    (0.01,  0.99),  # p
    (1.05, 50.0),   # alpha_plus: E[e^Y] < inf needs alpha_plus > 1
    (0.5, 100.0),   # alpha_minus
]
_KOU_NAMES = ['omega', 'lmbd', 'k1', 'k2', 'p', 'alpha_plus', 'alpha_minus']

_MODELS = {
    'Merton': (VIXMertonModelParameters, _MERTON_NAMES, _MERTON_BOUNDS),
    'Kou': (VIXKouModelParameters, _KOU_NAMES, _KOU_BOUNDS),
}
_PRICING_METHODS = [m.value for m in FourierMethod]

# Loss returned where the Fourier pricer gives non-finite prices (overflow at extreme
# parameters), so the optimisers move away. A fit whose best loss is this raises.
_PENALTY = 1e10

# A parameter within this fraction of its range from a bound is reported as at the bound.
_AT_BOUND_RTOL = 1e-3


class ContKokholmCalibrator(Calibrator):
    """
    Calibrates Cont-Kokholm model parameters to a VIX option surface.

    The loss is the weighted sum of squared price errors. Quotes are grouped by maturity
    and each maturity is priced with one Fourier transform. Differential evolution (without
    its built-in polish) is followed by L-BFGS-B refinement.

    VIX option prices do not depend on ``rho``, so it is not fitted: the returned
    parameters carry the ``rho`` given here.

    Args:
        model_type: 'Merton' or 'Kou'.
        pricing_method: Fourier pricing method ('Kar&Madan' or 'Cont&Tankov').
        rho: Index/variance correlation to put in the fitted parameters. Defaults to the
            parameter class default (the paper's -0.45).
        bounds: Search-box overrides, {parameter name: (lower, upper)}; other parameters keep
            the defaults (``_MERTON_BOUNDS`` / ``_KOU_BOUNDS``).
        de_maxiter: Max iterations for differential evolution.
        de_popsize: Population size multiplier for differential evolution.
        seed: Seed (or Generator) for differential evolution.
        verbose: Print calibration progress.

    Raises:
        ValueError: for an unknown ``model_type`` or ``pricing_method``, an invalid ``rho``, or
            bounds that name an unknown parameter or have lower >= upper.
    """

    def __init__(
        self,
        model_type: str = 'Merton',
        pricing_method: str = 'Kar&Madan',
        *,
        rho: float | None = None,
        bounds: Mapping[str, tuple[float, float]] | None = None,
        de_maxiter: int = 200,
        de_popsize: int = 10,
        seed: int | np.random.Generator | None = 42,
        verbose: bool = False,
    ):
        if model_type not in _MODELS:
            raise ValueError(f"Unknown model_type {model_type!r}; use 'Merton' or 'Kou'.")
        if pricing_method not in _PRICING_METHODS:
            raise ValueError(f"Unknown pricing_method {pricing_method!r}; use one of "
                             f"{_PRICING_METHODS}.")
        params_cls, names, default_bounds = _MODELS[model_type]
        if rho is None:
            rho = params_cls.model_fields['rho'].default
        if not -1 <= rho <= 1:
            raise ValueError(f"rho must be in [-1, 1], got {rho}.")

        box = dict(zip(names, default_bounds, strict=True))
        for name, (lo, hi) in (bounds or {}).items():
            if name not in box:
                raise ValueError(f"Unknown parameter {name!r} in bounds; {model_type} fits {names}.")
            if not lo < hi:
                raise ValueError(f"bounds[{name!r}] = ({lo}, {hi}) needs lower < upper.")
            box[name] = (float(lo), float(hi))

        self.model_type = model_type
        self.pricing_method = pricing_method
        self.rho = float(rho)
        self.names = list(names)
        self.bounds = [box[n] for n in names]
        self.de_maxiter = de_maxiter
        self.de_popsize = de_popsize
        self.seed = seed
        self.verbose = verbose

    def _build_params(self, x: np.ndarray) -> VIXMertonModelParameters | VIXKouModelParameters:
        if len(x) != len(self.names):
            raise ValueError(f"Expected {len(self.names)} values {self.names}, got {len(x)}.")
        params_cls = _MODELS[self.model_type][0]
        return params_cls(rho=self.rho, **dict(zip(self.names, map(float, x), strict=True)))

    @staticmethod
    def _prepare(surface: OptionSurface) -> dict:
        maturities, inverse = np.unique(surface.maturity, return_inverse=True)
        slices = [(T, np.flatnonzero(inverse == j)) for j, T in enumerate(maturities)]
        return {'surface': surface, 'slices': slices}

    def _model_prices(self, x: np.ndarray, d: dict) -> np.ndarray:
        s = d['surface']
        params = self._build_params(x)
        prices = np.empty(len(s))
        for T, idx in d['slices']:
            prices[idx] = price_vix_option_slice(s.spot, s.strike[idx], T, s.r, params,
                                                 self.pricing_method, s.option_type[idx])
        return prices

    def _loss(self, x: np.ndarray, d: dict) -> float:
        s = d['surface']
        with np.errstate(all='ignore'):
            value = float(np.sum(s.weight * (self._model_prices(x, d) - s.price) ** 2))
        return value if np.isfinite(value) else _PENALTY

    def objective(self, x: np.ndarray, surface: OptionSurface) -> float:
        """
        Weighted SSE between model and market prices for parameter vector ``x``.

        ``x`` holds the fitted parameters in the order of ``self.names`` (no ``rho``).
        Pricer errors propagate; non-finite model prices give the penalty 1e10.
        """
        return self._loss(np.asarray(x, dtype=float), self._prepare(surface))

    def calibrate(
        self, surface: OptionSurface
    ) -> tuple[VIXMertonModelParameters | VIXKouModelParameters, dict]:
        """
        Fit model parameters to the observed option surface.

        Args:
            surface: VIX option quotes; ``surface.spot`` is the VIX future.

        Returns:
            (params, info). ``params`` has ``rho`` set to the calibrator's ``rho``. ``info``:

            - ``success``: True unless a parameter ended on a bound of the search box;
            - ``message``: what happened, naming any parameters at a bound;
            - ``loss``: weighted sum of squared price errors at ``params``;
            - ``price_rmse`` / ``max_price_error``: unweighted, over every quote;
            - ``n_quotes`` / ``n_maturities``: size of the surface;
            - ``at_bounds``: names of the parameters within 0.1% of the range from a bound;
            - ``rho``: the fixed correlation.

        Raises:
            RuntimeError: if the model prices are non-finite at every parameter tried (the
                best loss is the penalty), so no fit was found.
        """
        d = self._prepare(surface)

        def loss(x: np.ndarray) -> float:
            return self._loss(x, d)

        if self.verbose:
            print(f"Starting differential evolution ({self.model_type} model)...")
        de = differential_evolution(loss, self.bounds, maxiter=self.de_maxiter,
                                    popsize=self.de_popsize, seed=self.seed, tol=1e-6,
                                    polish=False, disp=self.verbose)
        if self.verbose:
            print(f"DE loss: {de.fun:.6g}. Refining with L-BFGS-B...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            local = minimize(loss, de.x, method='L-BFGS-B', bounds=self.bounds,
                             options={'ftol': 1e-10, 'gtol': 1e-8, 'maxiter': 500})
        best_x = np.asarray(local.x if local.fun < de.fun else de.x, dtype=float)

        params = self._build_params(best_x)
        best_loss = loss(best_x)
        if not best_loss < _PENALTY:
            raise RuntimeError(
                f"Cont-Kokholm calibration failed: the model prices were not finite at the "
                f"best parameters found ({dict(zip(self.names, best_x.tolist(), strict=True))})."
            )

        errors = self._model_prices(best_x, d) - surface.price
        lo, hi = np.array(self.bounds).T
        tol = _AT_BOUND_RTOL * (hi - lo)
        at_bounds = [n for n, v, a, b, t in zip(self.names, best_x, lo, hi, tol, strict=True)
                     if v - a <= t or b - v <= t]
        message = ("Converged inside the search box." if not at_bounds else
                   f"Parameters at a bound of the search box: {at_bounds}. The fit is limited "
                   "by the box; widen it with `bounds` or check the quotes.")
        info = {
            'success': not at_bounds,
            'message': message,
            'loss': best_loss,
            'price_rmse': float(np.sqrt(np.mean(errors ** 2))),
            'max_price_error': float(np.max(np.abs(errors))),
            'n_quotes': len(surface),
            'n_maturities': len(d['slices']),
            'at_bounds': at_bounds,
            'rho': self.rho,
        }
        if self.verbose:
            print(f"Final loss: {best_loss:.6g}. {message}")
        return params, info
