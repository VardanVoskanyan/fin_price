"""
Calibration of the Cont-Kokholm model to an observed VIX option surface.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import differential_evolution, minimize

from finpricing.calibration.base import Calibrator
from finpricing.calibration.surface import OptionSurface
from finpricing.models.vix.pricing import price_vix_options
from finpricing.parameters import VIXKouModelParameters, VIXMertonModelParameters

# Parameter bounds: (lower, upper) for each free parameter
_MERTON_BOUNDS: list[tuple[float, float]] = [
    (0.1,  10.0),   # omega
    (0.1,  50.0),   # lmbd
    (1.0,  100.0),  # k1
    (0.1,  50.0),   # k2
    (-2.0,  2.0),   # m
    (0.01,  3.0),   # delta
    (-0.99, 0.0),   # rho
]
_MERTON_NAMES = ['omega', 'lmbd', 'k1', 'k2', 'm', 'delta', 'rho']

_KOU_BOUNDS: list[tuple[float, float]] = [
    (0.1,  10.0),   # omega
    (0.1, 100.0),   # lmbd
    (1.0, 100.0),   # k1
    (0.1,  50.0),   # k2
    (0.01,  0.99),  # p
    (0.5,  50.0),   # alpha_plus
    (0.5, 100.0),   # alpha_minus
    (-0.99,  0.0),  # rho
]
_KOU_NAMES = ['omega', 'lmbd', 'k1', 'k2', 'p', 'alpha_plus', 'alpha_minus', 'rho']


def _build_params(
    x: np.ndarray,
    model_type: str,
) -> VIXMertonModelParameters | VIXKouModelParameters:
    match model_type:
        case 'Merton':
            omega, lmbd, k1, k2, m, delta, rho = x
            return VIXMertonModelParameters(
                omega=omega, lmbd=lmbd, k1=k1, k2=k2, m=m, delta=delta, rho=rho,
            )
        case 'Kou':
            omega, lmbd, k1, k2, p, alpha_plus, alpha_minus, rho = x
            return VIXKouModelParameters(
                omega=omega, lmbd=lmbd, k1=k1, k2=k2,
                p=p, alpha_plus=alpha_plus, alpha_minus=alpha_minus, rho=rho,
            )
        case _:
            raise ValueError(f"Unknown model type: {model_type!r}")


class ContKokholmCalibrator(Calibrator):
    """
    Calibrates Cont-Kokholm model parameters to a VIX option surface.

    Uses global differential evolution followed by local L-BFGS-B refinement.

    Args:
        model_type: 'Merton' or 'Kou'.
        pricing_method: Fourier pricing method ('Kar&Madan' or 'Cont&Tankov').
        de_maxiter: Max iterations for differential evolution.
        de_popsize: Population size multiplier for differential evolution.
        verbose: Print calibration progress.
    """

    def __init__(
        self,
        model_type: str = 'Merton',
        pricing_method: str = 'Kar&Madan',
        de_maxiter: int = 200,
        de_popsize: int = 10,
        verbose: bool = False,
    ):
        self.model_type = model_type
        self.pricing_method = pricing_method
        self.de_maxiter = de_maxiter
        self.de_popsize = de_popsize
        self.verbose = verbose

    def calibrate(
        self,
        surface: OptionSurface,
        Vi_0: np.ndarray | None = None,
        **kwargs,
    ) -> tuple[VIXMertonModelParameters | VIXKouModelParameters, None]:
        """
        Fit model parameters to the observed option surface.

        Args:
            surface: VIX option quotes; ``surface.spot`` is the current VIX level.
            Vi_0: Initial variance swap rates. If None, uses paper defaults.

        Returns:
            (fitted_params, None) — the second element is reserved for future
            joint calibration of b_i coefficients.
        """
        bounds = _MERTON_BOUNDS if self.model_type == 'Merton' else _KOU_BOUNDS

        def loss(x: np.ndarray) -> float:
            return self._loss(x, surface)

        if self.verbose:
            print(f"Starting differential evolution ({self.model_type} model)...")

        de_result = differential_evolution(
            loss,
            bounds,
            maxiter=self.de_maxiter,
            popsize=self.de_popsize,
            seed=42,
            tol=1e-6,
            disp=self.verbose,
        )

        if self.verbose:
            print(f"DE loss: {de_result.fun:.6f}. Refining with L-BFGS-B...")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            local_result = minimize(
                loss,
                de_result.x,
                method='L-BFGS-B',
                bounds=bounds,
                options={'ftol': 1e-10, 'gtol': 1e-8, 'maxiter': 500},
            )

        best_x = local_result.x if local_result.fun < de_result.fun else de_result.x

        if self.verbose:
            print(f"Final loss: {min(local_result.fun, de_result.fun):.6f}")

        return _build_params(best_x, self.model_type), None

    def objective(self, x: np.ndarray, surface: OptionSurface) -> float:
        """Weighted SSE between model and market prices for parameter vector ``x``."""
        return self._loss(np.asarray(x, dtype=float), surface)

    def _loss(self, x: np.ndarray, surface: OptionSurface) -> float:
        try:
            params = _build_params(x, self.model_type)
            model_prices = np.array([
                price_vix_options(
                    V0=surface.spot,
                    K=K,
                    T=T,
                    r=surface.r,
                    model_params=params,
                    pricing_method=self.pricing_method,
                    option_type=opt,
                )
                for K, T, opt in zip(surface.strike, surface.maturity, surface.option_type,
                                     strict=True)
            ])
            return float(np.sum(surface.weight * (model_prices - surface.price) ** 2))
        except Exception:
            return 1e10
