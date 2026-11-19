"""
VIX Model implementation for simultaneous pricing of S&P and VIX derivatives
"""
import numpy as np

from finpricing.data.fixtures import (
    KOU_B_I,
    KOU_PARAMS,
    MERTON_B_I,
    MERTON_PARAMS,
    TENOR_DATES,
    VI_0,
)
from finpricing.models.jump_diffusion.jump_diffusion import (
    simul_tenor_jumps,
    simulate_tenor_diffusions,
)
from finpricing.models.vix.utils import jump_second_moment
from finpricing.parameters import VIXKouModelParameters, VIXMertonModelParameters
from finpricing.utils.math_utils import RngLike, as_generator


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


def check_index_variance(Vi0: np.ndarray, constraint_cf: float, b_i: np.ndarray) -> None:
    """
    Raise if some σ²_i would be negative.

    σ²_i = (V^i_{T_i} / V^i_0) · (V^i_0 − cf · b_i²) and V^i_{T_i} > 0, so σ²_i has the
    sign of V^i_0 − cf · b_i² on every path: one negative gap makes the model ill-defined.

    Args:
        Vi0: Initial variance swap rates, shape (k,)
        constraint_cf: Scalar coefficient from jump distribution moments, ∫ x² ν(dx)
        b_i: Index jump exposure coefficients, shape (k,)

    Raises:
        ValueError: if V^i_0 − cf · b_i² < 0 for some i.
    """
    gap = np.asarray(Vi0, dtype=float) - constraint_cf * np.asarray(b_i, dtype=float) ** 2
    bad = np.flatnonzero(gap < 0)
    if bad.size:
        raise ValueError(
            f"sigma_i^2 < 0 for tenor periods {bad.tolist()}: V^i_0 - cf * b_i^2 = "
            f"{np.round(gap[bad], 6).tolist()} with cf = {constraint_cf:.6g}. The index jumps "
            "carry more variance than the variance swap rate allows; reduce the jump second "
            "moment or b_i."
        )


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

    Raises:
        ValueError: if the parameters make some σ²_i negative (see ``check_index_variance``).
    """
    check_index_variance(Vi0, constraint_cf, b_i)
    return VTi - constraint_cf * (VTi / Vi0) * (b_i ** 2)


def log_U_increments(
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
    Per-period contributions to ln U_{T_k}; their cumulative sum over periods is ln U.

    Period i contributes the compensated jump drift, the ρ dZ diffusion and the index jumps
    (Cont & Kokholm 2013, p. 259). Arguments are as in ``get_UTm``.

    Returns:
        Array of shape (num_paths, k).
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

    drift_term = -(0.5 * (model_params.rho ** 2) * sigmas_sq + integral) * np.diff(tenor_dates)
    diffusion_term = model_params.rho * diff_Z * np.sqrt(sigmas_sq)
    jump_term = jumps_S * ui_factors

    return drift_term + diffusion_term + jump_term


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
    Compute the auxiliary ratio U_{T_k}: the part of S_{T_k} / S_0 driven by Z and the jumps.

    S_{T_k} = S_0 e^{r T_k} U_{T_k} exp(sqrt(1−ρ²) Σ σ_i ΔW_i − ½ (1−ρ²) Σ σ²_i Δ_i), with W
    independent of Z and the jumps, so E[U_{T_k}] = 1. See Cont & Kokholm (2013), p. 259.

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
    increments = log_U_increments(Vi_0, VT_i, diff_Z, jumps_S, sigmas_sq, b_i, tenor_dates,
                                  model_params)
    return np.exp(np.sum(increments, axis=1))


_MODEL_DEFAULTS = {
    'Merton': (VIXMertonModelParameters, MERTON_PARAMS, MERTON_B_I),
    'Kou': (VIXKouModelParameters, KOU_PARAMS, KOU_B_I),
}


def _frozen(values, name: str, length: int | None = None) -> np.ndarray:
    """Copy ``values`` into a read-only 1-D float array, checking it is finite (and its length)."""
    arr = np.array(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array, got shape {arr.shape}.")
    if length is not None and len(arr) != length:
        raise ValueError(f"{name} must have {length} values (one per tenor period), got {len(arr)}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite.")
    arr.flags.writeable = False
    return arr


class VixModel:
    """
    Simultaneous pricing model for S&P index options and VIX derivatives.

    Implements the Cont & Kokholm (2013) framework with Merton (Gaussian)
    or Kou (double-exponential) jump distributions.

    The inputs are read-only: ``tenor_dates``, ``params``, ``Vi_0`` and ``b_i`` (and
    ``model_type``) can't be reassigned, and the arrays are not writeable, so the cached
    Monte Carlo paths always match them. Build a new ``VixModel`` to change an input;
    ``fit_data`` is the one method that replaces ``params``, and it clears the cache.

    Args:
        tenor_dates: Sorted tenor dates T_0 = 0 < T_1 < ... < T_m in years, shape (m+1,).
        model_type: Jump distribution — 'Merton' or 'Kou'.
        params: Model parameters of the matching family (``VIXMertonModelParameters`` or
            ``VIXKouModelParameters``). Defaults to Table 2 of the paper.
        Vi_0: Initial forward variance swap rates V^i_0, one per tenor period, shape (m,).
            Defaults to the paper's values.
        b_i: Index jump loadings, one per tenor period, shape (m,). Defaults to the paper's
            values for ``model_type``.

    Raises:
        ValueError: for an unknown ``model_type``, ``params`` of the other family, tenor dates
            that do not start at 0 or are not strictly increasing, or ``Vi_0`` / ``b_i`` whose
            length is not ``len(tenor_dates) - 1`` (pass all three when changing the tenors).
    """

    def __init__(
        self,
        tenor_dates: np.ndarray = TENOR_DATES,
        model_type: str = 'Merton',
        params: VIXMertonModelParameters | VIXKouModelParameters | None = None,
        Vi_0: np.ndarray | None = None,
        b_i: np.ndarray | None = None,
    ):
        if model_type not in _MODEL_DEFAULTS:
            raise ValueError(f"Unknown model type: {model_type!r}. Must be 'Merton' or 'Kou'.")
        params_cls, default_params, default_b_i = _MODEL_DEFAULTS[model_type]

        if params is None:
            params = default_params
        elif not isinstance(params, params_cls):
            raise ValueError(f"model_type={model_type!r} needs {params_cls.__name__}, "
                             f"got {type(params).__name__}.")

        tenor_dates = _frozen(tenor_dates, "tenor_dates")
        if len(tenor_dates) < 2 or tenor_dates[0] != 0 or np.any(np.diff(tenor_dates) <= 0):
            raise ValueError("tenor_dates must start at 0 and be strictly increasing, "
                             f"with at least one period; got {tenor_dates.tolist()}.")
        m = len(tenor_dates) - 1

        self._model_type = model_type
        self._tenor_dates = tenor_dates
        self._params = params
        self._Vi_0 = _frozen(VI_0 if Vi_0 is None else Vi_0, "Vi_0", m)
        if np.any(self._Vi_0 <= 0):
            raise ValueError("Vi_0 must be positive.")
        self._b_i = _frozen(default_b_i if b_i is None else b_i, "b_i", m)

        self._stored_data: list = []
        self._index_dW: np.ndarray | None = None

    # -- read-only inputs ---------------------------------------------------

    @property
    def model_type(self) -> str:
        """'Merton' or 'Kou'."""
        return self._model_type

    @property
    def tenor_dates(self) -> np.ndarray:
        """Tenor dates T_0 = 0 < ... < T_m (read-only array)."""
        return self._tenor_dates

    @property
    def params(self) -> VIXMertonModelParameters | VIXKouModelParameters:
        """Model parameters (a frozen pydantic model)."""
        return self._params

    @property
    def Vi_0(self) -> np.ndarray:
        """Initial forward variance swap rates, one per tenor period (read-only array)."""
        return self._Vi_0

    @property
    def b_i(self) -> np.ndarray:
        """Index jump loadings, one per tenor period (read-only array)."""
        return self._b_i

    @property
    def num_tenors(self) -> int:
        """Number of tenor periods m = len(tenor_dates) - 1."""
        return len(self._tenor_dates) - 1

    @property
    def constraint_cf(self) -> float:
        """Jump second moment ∫ x² ν(dx) of ``params`` (see ``utils.jump_second_moment``)."""
        return jump_second_moment(self._params)

    # -- simulation ---------------------------------------------------------

    def store_tenor_data(self, num_paths: int = 10 ** 5, rng: RngLike = None) -> None:
        """
        Simulate and store diffusion and jump terms for all tenor periods.

        Also draws the index Brownian increments ΔW (independent of Z and the jumps) that
        ``get_index_prices`` needs; ``index_option_pricer`` integrates W out exactly.

        Args:
            num_paths: Number of Monte Carlo paths.
            rng: Generator or int seed; one Generator drives every draw. None draws fresh
                entropy. NumPy's global random state is never used.
        """
        if int(num_paths) != num_paths or num_paths < 1:
            raise ValueError(f"num_paths must be a positive integer, got {num_paths}.")
        num_paths = int(num_paths)
        rng = as_generator(rng)
        diff_V, diff_Z = simulate_tenor_diffusions(num_paths, self.tenor_dates, k1=self.params.k1,
                                                   rng=rng)
        jumps_S, jumps_V = simul_tenor_jumps(self.params, num_paths, self.tenor_dates, rng=rng)
        VTi = get_vix_prices(self.Vi_0, diff_V[:, :-1], jumps_V[:, :-1], self.tenor_dates, self.params)
        self._index_dW = rng.standard_normal(diff_Z.shape) * np.sqrt(np.diff(self.tenor_dates))
        self._stored_data = [diff_V, diff_Z, jumps_S, jumps_V, VTi]

    def _ensure_paths(self) -> None:
        if not self._stored_data:
            self.store_tenor_data()

    def _check_tenor_index(self, tenor_index: int) -> int:
        if isinstance(tenor_index, bool) or not isinstance(tenor_index, int | np.integer):
            raise ValueError(f"tenor_index must be an integer, got {tenor_index!r}.")
        if not 1 <= tenor_index <= self.num_tenors:
            raise ValueError(f"tenor_index must be between 1 and {self.num_tenors} (the number of "
                             f"tenor periods), got {tenor_index}.")
        return int(tenor_index)

    def get_index_prices(self, S0: float, r: float = 0.0) -> np.ndarray:
        """
        Return simulated index prices S_{T_k} at each tenor date T_1, ..., T_m.

        S_{T_k} = S_0 e^{r T_k} U_{T_k} exp(sqrt(1−ρ²) Σ_{i≤k} σ_i ΔW_i − ½ (1−ρ²) Σ_{i≤k} σ²_i Δ_i),
        on the stored paths (simulated with ``store_tenor_data``, or with a fresh generator if
        none are stored). The paths are consistent across tenors, and discounted payoffs
        averaged over them match ``index_option_pricer`` up to Monte Carlo error.

        Args:
            S0: Current spot index price.
            r: Risk-free rate (continuous).

        Returns:
            Array of shape (num_paths, m) where m = len(tenor_dates) - 1.

        Raises:
            ValueError: if the parameters make some σ²_i negative (see ``check_index_variance``).
        """
        self._ensure_paths()
        _, diff_Z, jumps_S, _, VTi = self._stored_data
        rho, dt = self.params.rho, np.diff(self.tenor_dates)

        sigmas_sq = get_sigmas(self.Vi_0, VTi, self.constraint_cf, self.b_i)
        log_U = log_U_increments(self.Vi_0, VTi, diff_Z, jumps_S, sigmas_sq, self.b_i,
                                 self.tenor_dates, self.params)
        log_W = np.sqrt(1 - rho ** 2) * np.sqrt(sigmas_sq) * self._index_dW \
            - 0.5 * (1 - rho ** 2) * sigmas_sq * dt
        return S0 * np.exp(r * self.tenor_dates[1:] + np.cumsum(log_U + log_W, axis=1))

    # -- pricing ------------------------------------------------------------

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
            S0: Current VIX future level.
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

        Uses the stored paths (``store_tenor_data``); if none are stored, simulates 10^5
        paths with a fresh generator.

        Args:
            S0: Spot price.
            strikes: Array of strike prices.
            tenor_index: Expiry T_{tenor_index}, an integer from 1 to m = len(tenor_dates) - 1.
            r: Risk-free rate.
            option_type: 'call' or 'put'.

        Returns:
            Array of option prices for each strike.

        Raises:
            ValueError: if ``tenor_index`` is outside 1..m.
        """
        tenor_index = self._check_tenor_index(tenor_index)
        self._ensure_paths()

        from finpricing.models.vix.pricing import price_index_options
        return price_index_options(
            S0, self.Vi_0, strikes, r,
            self.tenor_dates[:tenor_index + 1],
            self._stored_data, self.b_i, self.params, option_type,
        )

    def fit_data(self, option_surface, **calibrator_options) -> dict:
        """
        Calibrate the jump and variance parameters to VIX option quotes.

        VIX options do not depend on ρ, so ρ stays at ``params.rho``; ``Vi_0`` and ``b_i`` are
        not fitted. Replaces ``params`` and clears the stored paths.

        Args:
            option_surface: VIX option quotes (``finpricing.calibration.OptionSurface``);
                            its ``spot`` is the VIX future.
            **calibrator_options: Passed to ``ContKokholmCalibrator`` (``pricing_method``,
                ``seed``, ``de_maxiter``, ``bounds``, ...).

        Returns:
            The calibration report (see ``ContKokholmCalibrator.calibrate``).
        """
        from finpricing.calibration.cont_kokholm import ContKokholmCalibrator
        calibrator = ContKokholmCalibrator(self.model_type, rho=self.params.rho,
                                           **calibrator_options)
        new_params, info = calibrator.calibrate(option_surface)
        self._params = new_params
        self._stored_data = []  # invalidate cached MC paths
        self._index_dW = None
        return info
