"""
Jump-Diffusion Models: Merton and Kou
"""
import numpy as np
from scipy.integrate import quad_vec

from finpricing.parameters import VIXKouModelParameters, VIXMertonModelParameters
from finpricing.utils.math_utils import RngLike, as_generator

try:
    from finpricing._sim import simul_total_jumps_kou as _rust_kou
    from finpricing._sim import simul_total_jumps_merton as _rust_merton
    _RUST_AVAILABLE = True
except ImportError:
    _RUST_AVAILABLE = False


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                               DRIFT TERMS                                       #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def shared_drift_base(T: float, omega: float, k1: float) -> float:
    """
    Computes the shared drift base term for jump-diffusion models.
    """
    return -0.5 * (omega**2) * (1 - np.exp(-2 * k1 * T)) / (2 * k1)


def merton_integrand(s: float, T: float, k2: float, m: float, delta: float) -> float:
    """
    Drift integrand for the Merton model.
    """
    return np.exp(np.exp(-k2 * (T - s)) * m + 0.5 * np.exp(-2 * k2 * (T - s)) * delta**2) - 1


def kou_integrand(
        s: float,
        T: float,
        k2: float,
        p: float,
        alpha_plus: float,
        alpha_minus: float
) -> float:
    """
    Drift integrand for the Kou model.

    Args:
        s: Time s.
        T: Time T.
        k2: Jump intensity.
        p: Jump probability.
        alpha_plus: Positive jump rate.
        alpha_minus: Negative jump rate.

    Returns:
        Value of the integrand at time s.
    """
    return (p * alpha_plus / (alpha_plus - np.exp(-k2 * (T - s)))
            + (1 - p) * alpha_minus / (alpha_minus + np.exp(-k2 * (T - s))) - 1)


def merton_drift(T: float, omega: float, lmbd: float, k1: float, k2: float, m: float, delta: float) -> float:
    """
    Computes the drift term for the Merton jump-diffusion model.
    """
    base_drift = shared_drift_base(T, omega, k1)
    jump_term = lmbd * quad_vec(lambda s: merton_integrand(s, T, k2, m, delta), 0, T)[0]
    return base_drift - jump_term


def kou_drift(T: float, omega: float, lmbd: float, k1: float, k2: float, p: float, alpha_plus: float, alpha_minus: float) -> float:
    """
    Computes the drift term for the Kou jump-diffusion model.
    """
    base_drift = shared_drift_base(T, omega, k1)
    jump_term = lmbd * quad_vec(lambda s: kou_integrand(s, T, k2, p, alpha_plus, alpha_minus), 0, T)[0]
    return base_drift - jump_term


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                               JUMP TERMS                                       #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def simul_total_jumps(
        model_params: VIXMertonModelParameters | VIXKouModelParameters,
        num_paths: int,
        T1: float,
        T2: float,
        rng: RngLike = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate jump terms according to model parameters between times T1 and T2.

    Args:
        model_params: Model parameters (VIXMertonModelParameters or VIXKouModelParameters).
        num_paths: Number of simulation paths.
        T1: Start time (non-negative).
        T2: End time (positive, greater than T1).
        rng: Generator or int seed (see ``utils.math_utils.as_generator``). None draws
            fresh entropy. NumPy's global random state is never used. With the Rust
            extension built, one 64-bit seed is drawn from ``rng`` and the extension runs
            a seeded ChaCha8 stream per path, so results are reproducible for a given
            build but differ from the pure-Python ones.

    Returns:
        Two 1D arrays:
            - Total jumps for each path: sum Y_j
            - Exponential-weighted jumps for each path: sum exp(k2 * tau_j) * Y_j
    """
    if T1 < 0 or T2 <= T1:
        raise ValueError("T1 must be non-negative and T2 must be greater than T1.")
    rng = as_generator(rng)

    if _RUST_AVAILABLE:
        seed = int(rng.integers(0, 2**64, dtype=np.uint64))
        match model_params:
            case VIXMertonModelParameters():
                return _rust_merton(num_paths, T1, T2, model_params.lmbd, model_params.k2, model_params.m, model_params.delta, seed)
            case VIXKouModelParameters():
                return _rust_kou(num_paths, T1, T2, model_params.lmbd, model_params.k2, model_params.p, model_params.alpha_plus, model_params.alpha_minus, seed)
            case _:
                raise ValueError("Unsupported model parameters.")

    # Pure-Python fallback
    match model_params:
        case VIXMertonModelParameters():
            lmbd = model_params.lmbd
            k2 = model_params.k2
        case VIXKouModelParameters():
            lmbd = model_params.lmbd
            k2 = model_params.k2
        case _:
            raise ValueError("Unsupported model parameters. Must be VIXMertonModelParameters or VIXKouModelParameters.")

    jump_counts = rng.poisson(lmbd * (T2 - T1), size=num_paths)
    max_jumps = int(np.max(jump_counts))

    if max_jumps == 0:
        return np.zeros(num_paths), np.zeros(num_paths)

    jump_times = rng.uniform(low=T1, high=T2, size=(num_paths, max_jumps))

    match model_params:
        case VIXMertonModelParameters():
            jump_sizes = rng.normal(loc=model_params.m, scale=model_params.delta, size=(num_paths, max_jumps))
        case VIXKouModelParameters():
            p = model_params.p
            jump_directions = np.where(rng.random((num_paths, max_jumps)) < p, 1, -1)
            jump_sizes = jump_directions * rng.exponential(
                scale=(jump_directions > 0) / model_params.alpha_plus + (jump_directions < 0) / model_params.alpha_minus,
                size=(num_paths, max_jumps),
            )

    # mask[i, j] = True only when j < jump_counts[i] — excludes phantom samples
    mask = np.arange(max_jumps)[None, :] < jump_counts[:, None]
    total_jumps = np.sum(jump_sizes * mask, axis=1)
    exp_weighted_jumps = np.sum(jump_sizes * np.exp(k2 * jump_times) * mask, axis=1)

    return total_jumps, exp_weighted_jumps


def simul_tenor_jumps(
        model_params: VIXMertonModelParameters | VIXKouModelParameters,
        num_paths: int = 10**5,
        tenor_dates: np.ndarray | None = None,
        rng: RngLike = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulates the sum of jumps for each tenor period for the index and
    the total jump term for variance swaps up to each tenor date.

    Args:
        model_params: Model parameters containing jump process parameters.
        num_paths: Number of Monte Carlo simulation paths.
        tenor_dates: Sorted numpy array of tenor dates (default: monthly up to 5 months).
        rng: Generator or int seed shared by every tenor period (see
            ``utils.math_utils.as_generator``). None draws fresh entropy.

    Returns:
        Two 2D arrays of shape (num_paths, len(tenor_dates) - 1):
            - sum_{T_{i-1} <= tau_j < T_i} Y_j
            - sum_{tau_j < T_i} exp(-k2 * (T_i - tau_j)) Y_j
    """
    if tenor_dates is None:
        tenor_dates = np.arange(0, 1/2, 1/12)
    k2 = model_params.k2
    rng = as_generator(rng)

    total_jumps = np.zeros((num_paths, len(tenor_dates) - 1))
    vix_jumps = np.zeros((num_paths, len(tenor_dates) - 1))

    for i in range(len(tenor_dates) - 1):
        total_jumps[:, i], vix_jumps[:, i] = simul_total_jumps(
            model_params, num_paths, tenor_dates[i], tenor_dates[i + 1], rng=rng
        )

    return total_jumps, np.exp(-k2 * tenor_dates[1:]) * np.cumsum(vix_jumps, axis=1)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                               DIFFUSION TERMS                                  #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def simulate_tenor_diffusions(
        num_paths: int,
        tenor_dates: np.ndarray,
        k1: float,
        rng: RngLike = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate the diffusion parts of VIX and the index on tenor dates.

    Args:
        num_paths: The number of Monte Carlo simulation paths.
        tenor_dates: Sorted numpy array of tenor dates.
        k1: Volatility scaling parameter.
        rng: Generator or int seed (see ``utils.math_utils.as_generator``). None draws
            fresh entropy.

    Returns:
        Two 2D arrays of shape (num_paths, len(tenor_dates) - 1):
            - ZV: Diffusion term in volatility swap dynamics from Cont&Kokholm equation (3.1)
                  ZV_i = integral_0^T_i exp(k1*(T_i-s)) dZ_s
            - dZ: Increments of the diffusion controlling the index prices,
                  each column representing Z(T_i) - Z(T_{i-1})
    """
    # Moments relative to each period's end, so only exp(-k1 * dT) appears and
    # nothing overflows at long tenors. Over [T_{i-1}, T_i] with I_i = int exp(-k1 (T_i - s)) dZ_s:
    #   Var(I_i) = (1 - exp(-2 k1 dT)) / (2 k1),  Cov(I_i, dZ_i) = (1 - exp(-k1 dT)) / k1.
    rng = as_generator(rng)
    dT = np.diff(tenor_dates)
    variations = -np.expm1(-2 * k1 * dT) / (2 * k1)
    correlations = -np.expm1(-k1 * dT) / k1

    dIZ = rng.standard_normal((num_paths, len(tenor_dates) - 1)) * np.sqrt(variations)

    dZ = rng.standard_normal((num_paths, len(tenor_dates) - 1))
    dZ = (correlations / variations) * dIZ + \
         np.sqrt(np.maximum(dT - correlations**2 / variations, 0.0)) * dZ

    # ZV_i = exp(-k1 dT_i) ZV_{i-1} + I_i
    decay = np.exp(-k1 * dT)
    ZV = np.empty_like(dIZ)
    acc = np.zeros(num_paths)
    for i in range(len(dT)):
        acc = decay[i] * acc + dIZ[:, i]
        ZV[:, i] = acc

    return ZV, dZ
