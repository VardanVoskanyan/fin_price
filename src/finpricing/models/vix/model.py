"""
VIX Model implementation for simultaneous pricing of S&P and VIX derivatives
"""
import numpy as np
from finpricing.parameters import VIXMertonModelParameters, VIXKouModelParameters
from finpricing.models.jump_diffusion.jump_diffusion import simul_tenor_jumps, simulate_tenor_diffusions
from finpricing.data.fixtures import (
    TENOR_DATES, VI_0,
    MERTON_PARAMS, MERTON_B_I,
    KOU_PARAMS, KOU_B_I,
)


def get_vix_prices(
    Vi_0: np.ndarray,
    diff_V: np.ndarray,
    jump_V: np.ndarray,
    tenor_dates: np.ndarray,
    model_params: VIXMertonModelParameters | VIXKouModelParameters,
) -> np.ndarray:
    """
    Compute variance swap rates V^i_{T_i} at each tenor date.

    Args:
        Vi_0: Initial values V^i_0, shape (m,)
        diff_V: Diffusion terms up to each tenor, shape (num_paths, m-1)
        jump_V: Jump terms up to each tenor, shape (num_paths, m-1)
        tenor_dates: Sorted tenor dates, shape (m+1,)
        model_params: Model parameters

    Returns:
        VT_i: Values at tenor dates, shape (num_paths, m)
    """
    from finpricing.models.vix.utils import vix_drift_coeff

    m = len(tenor_dates) - 1
    tenor_drifts = np.array([vix_drift_coeff(tenor_dates[i + 1], model_params) for i in range(m - 1)])

    VT_i = Vi_0[0] * np.ones((len(diff_V), len(Vi_0)))
    VT_i[:, 1:] = Vi_0[1:] * np.exp(tenor_drifts + model_params.omega * diff_V + jump_V)

    return VT_i


def get_sigmas(
    Vi0: np.ndarray,
    VTi: np.ndarray,
    constraint_cf: float,
    b_i: np.ndarray,
) -> np.ndarray:
    """
    Compute squared tenor volatilities σ²_i.

    Args:
        Vi0: Initial variance swap rates, shape (k,)
        VTi: Simulated variance swap rates, shape (num_paths, k)
        constraint_cf: Scalar coefficient from jump distribution moments
        b_i: Index jump exposure coefficients, shape (k,)

    Returns:
        sigmas_sq: shape (num_paths, k)
    """
    return VTi - constraint_cf * (VTi / Vi0) * (b_i ** 2)


def get_UTm(
    Vi_0: np.ndarray,
    VT_i: np.ndarray,
    diff_Z: np.ndarray,
    jumps_S: np.ndarray,
    sigmas_sq: np.ndarray,
    b_i: np.ndarray,
    tenor_dates: np.ndarray,
    model_params: VIXMertonModelParameters | VIXKouModelParameters,
) -> np.ndarray:
    """
    Compute the auxiliary index price ratio U_{T_k} = S_{T_k} / S_0.

    See Cont & Kokholm (2013), equation on p. 259.

    Args:
        Vi_0: Initial variance swap rates, shape (k,)
        VT_i: Simulated variance swap rates, shape (num_paths, k)
        diff_Z: Index diffusion increments, shape (num_paths, k)
        jumps_S: Index jump totals per tenor, shape (num_paths, k)
        sigmas_sq: Squared volatilities, shape (num_paths, k)
        b_i: Index jump exposure coefficients, shape (k,)
        tenor_dates: Tenor dates, shape (k+1,)
        model_params: Model parameters

    Returns:
        U_{T_k}: shape (num_paths,)
    """
    ui_factors = np.sqrt(VT_i / Vi_0) * b_i
    lmbd = model_params.lmbd

    match model_params:
        case VIXMertonModelParameters():
            integral = lmbd * (np.exp(model_params.m * ui_factors + 0.5 * (model_params.delta ** 2) * (ui_factors ** 2)) - 1)
        case VIXKouModelParameters():
            a_plus, a_minus = model_params.alpha_plus, model_params.alpha_minus
            integral = lmbd * (
                model_params.p * a_plus / (a_plus - ui_factors)
                + (1 - model_params.p) * a_minus / (a_minus + ui_factors)
                - 1
            )

    drift_term = np.sum(-(0.5 * (model_params.rho ** 2) * sigmas_sq + integral) * np.diff(tenor_dates), axis=1)
    diffusion_term = model_params.rho * np.sum(diff_Z * np.sqrt(sigmas_sq), axis=1)
    jump_term = np.sum(jumps_S * ui_factors, axis=1)

    return np.exp(drift_term + diffusion_term + jump_term)


class VixModel:
    """
    Simultaneous pricing model for S&P index options and VIX derivatives.

    Implements the Cont & Kokholm (2013) framework with Merton (Gaussian)
    or Kou (double-exponential) jump distributions.

    Args:
        tenor_dates: Sorted tenor dates in years starting from 0.
        model_type: Jump distribution — 'Merton' or 'Kou'.
    """

    def __init__(
        self,
        tenor_dates: np.ndarray = TENOR_DATES,
        model_type: str = 'Merton',
    ):
        self.model_type = model_type
        self.tenor_dates = tenor_dates

        match model_type:
            case 'Merton':
                self.params = MERTON_PARAMS
                self.Vi_0 = VI_0.copy()
                self.b_i = MERTON_B_I.copy()
                self.constraint_cf = self.params.lmbd * (self.params.m ** 2 + self.params.delta ** 2)
            case 'Kou':
                self.params = KOU_PARAMS
                self.Vi_0 = VI_0.copy()
                self.b_i = KOU_B_I.copy()
                self.constraint_cf = 2 * self.params.lmbd * (
                    self.params.p / self.params.alpha_plus ** 2
                    + (1 - self.params.p) / self.params.alpha_minus ** 2
                )
            case _:
                raise ValueError(f"Unknown model type: {model_type!r}. Must be 'Merton' or 'Kou'.")

        self._stored_data: list = []

    def store_tenor_data(self, num_paths: int = 10 ** 5) -> None:
        """
        Simulate and store diffusion and jump terms for all tenor periods.

        Args:
            num_paths: Number of Monte Carlo paths.
        """
        diff_V, diff_Z = simulate_tenor_diffusions(num_paths, self.tenor_dates, k1=self.params.k1)
        jumps_S, jumps_V = simul_tenor_jumps(self.params, num_paths, self.tenor_dates)
        VTi = get_vix_prices(self.Vi_0, diff_V[:, :-1], jumps_V[:, :-1], self.tenor_dates, self.params)
        self._stored_data = [diff_V, diff_Z, jumps_S, jumps_V, VTi]

    def get_index_prices(self, S0: float) -> np.ndarray:
        """
        Return simulated index prices S_{T_k} = S_0 * U_{T_k} at each tenor date.

        Args:
            S0: Current spot index price.

        Returns:
            Array of shape (num_paths, m) where m = len(tenor_dates) - 1.
        """
        if not self._stored_data:
            self.store_tenor_data()

        diff_V, diff_Z, jumps_S, jumps_V, VTi = self._stored_data
        m = len(self.tenor_dates) - 1
        num_paths = len(diff_Z)
        S_paths = np.zeros((num_paths, m))

        for k in range(1, m + 1):
            sigmas_sq = get_sigmas(self.Vi_0[:k], VTi[:, :k], self.constraint_cf, self.b_i[:k])
            U_Tk = get_UTm(
                self.Vi_0[:k], VTi[:, :k], diff_Z[:, :k], jumps_S[:, :k],
                sigmas_sq, self.b_i[:k], self.tenor_dates[:k + 1], self.params,
            )
            S_paths[:, k - 1] = S0 * U_Tk

        return S_paths

    def vix_option_pricer(
        self,
        S0: float,
        K: float,
        T: float,
        r: float,
        option_type: str = 'call',
        pricing_method: str = 'Kar&Madan',
    ) -> float:
        """
        Price a VIX option using Fourier transform methods.

        Args:
            S0: Current VIX level.
            K: Strike price.
            T: Time to maturity (years).
            r: Risk-free rate.
            option_type: 'call' or 'put'.
            pricing_method: 'Cont&Tankov' or 'Kar&Madan'.

        Returns:
            Option price.
        """
        from finpricing.models.vix.pricing import price_vix_options
        return price_vix_options(S0, K, T, r, self.params, pricing_method, option_type)

    def index_option_pricer(
        self,
        S0: float,
        strikes: np.ndarray,
        tenor_index: int,
        r: float = 0.0,
        option_type: str = 'call',
    ) -> np.ndarray:
        """
        Price index options via Monte Carlo simulation.

        Args:
            S0: Spot price.
            strikes: Array of strike prices.
            tenor_index: Index into tenor_dates for the expiry date.
            r: Risk-free rate.
            option_type: 'call' or 'put'.

        Returns:
            Array of option prices for each strike.
        """
        if not self._stored_data:
            self.store_tenor_data()

        from finpricing.models.vix.pricing import price_index_options
        return price_index_options(
            S0, self.Vi_0, strikes, r,
            self.tenor_dates[:tenor_index + 1],
            self._stored_data, self.b_i, self.params, option_type,
        )

    def fit_data(self, option_surface, model_type: str | None = None) -> None:
        """
        Calibrate model parameters to an observed option surface.

        Args:
            option_surface: VIX option quotes (``finpricing.calibration.OptionSurface``);
                            its ``spot`` is the current VIX level.
            model_type: Override model type for calibration. Defaults to self.model_type.
        """
        from finpricing.calibration.cont_kokholm import ContKokholmCalibrator
        target_type = model_type or self.model_type
        calibrator = ContKokholmCalibrator(model_type=target_type)
        new_params, new_b_i = calibrator.calibrate(option_surface, Vi_0=self.Vi_0)

        self.params = new_params
        if new_b_i is not None:
            self.b_i = new_b_i

        match new_params:
            case VIXMertonModelParameters():
                self.constraint_cf = new_params.lmbd * (new_params.m ** 2 + new_params.delta ** 2)
            case VIXKouModelParameters():
                self.constraint_cf = 2 * new_params.lmbd * (
                    new_params.p / new_params.alpha_plus ** 2
                    + (1 - new_params.p) / new_params.alpha_minus ** 2
                )

        self._stored_data = []  # invalidate cached MC paths
