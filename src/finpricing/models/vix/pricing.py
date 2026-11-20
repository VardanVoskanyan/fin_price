"""
VIX pricing functions using various methods
"""
import numpy as np
from scipy import interpolate

from finpricing.models.vix.model import get_sigmas, get_UTm
from finpricing.models.vix.utils import jump_second_moment
from finpricing.utils.bs_utils import bs_call, bs_put
from finpricing.utils.characteristic_functions import characteristic_function
from finpricing.utils.fourier import FourierMethod, fourier_call_pricer

# Kar-Madan damping and default grid (see utils/fourier.py).
_KM_ALPHA, _KM_N, _KM_DELTA = 0.75, 1024, 0.17


def _kar_madan_grid(model_params) -> dict:
    """
    Grid for Kar-Madan VIX pricing.

    Kar-Madan evaluates the VIX characteristic function at u - (1 + alpha) i. With Kou
    jumps (scaled by 1/2 in the VIX transform) that point lies a distance
    alpha_plus - (1 + alpha)/2 from a pole. As alpha_plus -> 1 the integrand peaks sharply,
    so the integration step shrinks with that distance and N grows to keep the same
    frequency range. Merton jumps have no pole and use the default grid.
    """
    from finpricing.parameters import VIXKouModelParameters

    if not isinstance(model_params, VIXKouModelParameters):
        return {}
    distance = model_params.alpha_plus - 0.5 * (1 + _KM_ALPHA)
    delta = min(_KM_DELTA, 0.3 * distance)
    n = int(2 ** np.ceil(np.log2(_KM_N * _KM_DELTA / delta)))
    return {"N": n, "Delta": delta, "alpha": _KM_ALPHA}


def _fourier_method(pricing_method: str) -> FourierMethod:
    try:
        return FourierMethod(pricing_method)
    except ValueError:
        raise ValueError(f"Unknown pricing_method {pricing_method!r}; use one of "
                         f"{[m.value for m in FourierMethod]}.") from None


def price_vix_option_slice(V0: float, strikes, T: float, r: float, model_params,
                           pricing_method: str = 'Kar&Madan', option_type='call') -> np.ndarray:
    """
    Price VIX options with one maturity and many strikes from a single Fourier transform.

    The FFT gives call prices on a log-moneyness grid; each strike is interpolated from it
    and puts follow from parity on the VIX future (a martingale, discounted at r):
    C - P = exp(-rT) (V0 - K).

    Args:
        V0: Current VIX future level.
        strikes: Strikes, shape (n,).
        T: Common time to maturity.
        r: Risk-free rate.
        model_params: VIXMertonModelParameters or VIXKouModelParameters.
        pricing_method: 'Kar&Madan' or 'Cont&Tankov'.
        option_type: 'call' / 'put', one value or one per strike.

    Returns:
        Prices, shape (n,).

    Raises:
        ValueError: for an unknown ``pricing_method`` or ``option_type``, or a strike outside
            the FFT grid.
    """
    method = _fourier_method(pricing_method)
    strikes = np.atleast_1d(np.asarray(strikes, dtype=float))
    option_type = np.broadcast_to(np.asarray(option_type), strikes.shape)
    bad = ~np.isin(option_type, ['call', 'put'])
    if np.any(bad):
        raise ValueError(f"Unknown option type: {option_type[bad][0]}")

    def chr_func(x):
        return characteristic_function(x, T, model_params, asset_type='VIX')

    grid = _kar_madan_grid(model_params) if method == FourierMethod.KAR_MADAN else {}
    us, ops = fourier_call_pricer(chr_func, T, r, method=method, **grid)

    # Interpolate to get prices for moneyness values between the points of np.exp(us)
    calls = V0 * interpolate.interp1d(np.exp(us), ops)(strikes / V0)
    return np.where(option_type == 'call', calls, calls - np.exp(-r * T) * (V0 - strikes))


def price_vix_options(V0: float, K: float, T: float, r: float, model_params,
                     pricing_method: str = 'Kar&Madan', option_type: str = 'call') -> float:
    """
    Returns the price of an option on VIX future using Fourier transform method.

    To price several strikes of one maturity, ``price_vix_option_slice`` runs the
    transform once instead of once per strike.

    Args:
        V0: Current VIX future level
        K: Strike price
        T: Time to maturity
        r: Risk-free rate
        model_params: Model parameters (VIXMertonModelParameters or VIXKouModelParameters)
        pricing_method: Fourier pricing method ('Cont&Tankov' or 'Kar&Madan')
        option_type: Option type ('call' or 'put')

    Returns:
        Option price

    Raises:
        ValueError: for an unknown ``pricing_method`` or ``option_type``.
    """
    return float(price_vix_option_slice(V0, [K], T, r, model_params, pricing_method,
                                        option_type)[0])


def price_index_options(S0, Vi_0, strikes, r, tenor_dates, stored_data, b_i, model_params, option_type='call'):
    """
    Returns the prices of an Index option with expiry at the model's tenor days
    using Monte Carlo simulation.
    
    Args:
        S0: Initial stock price
        Vi_0: Initial variance swap rates
        strikes: Array of strike prices
        r: Risk-free rate
        tenor_dates: Array of tenor dates
        stored_data: Pre-simulated Monte Carlo data
        b_i: Model coefficients for index jumps
        model_params: Model parameters object
        option_type: Option type ('call' or 'put')
        
    Returns:
        Array of option prices for given strikes
    """
    if option_type not in ('call', 'put'):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    k = len(tenor_dates) -1
    # Get the stored data
    diff_V, diff_Z, jumps_S, jumps_V, VTi = stored_data
    
    # Compute the sigma_i
    rho = model_params.rho
    
    constraint_cf = jump_second_moment(model_params)
    sigmas_sq = get_sigmas(Vi_0[:k], VTi[:,:k], constraint_cf, b_i[:k])
    
    # Compute U_T_k and sigma*
    U_Tm = get_UTm(Vi_0[:k], VTi[:,:k], diff_Z[:,:k], jumps_S[:,:k], sigmas_sq[:,:k],
                   b_i[:k], tenor_dates[:k+1], model_params)

    sigma_star = np.sqrt((1 - rho**2)*np.sum(sigmas_sq[:,:k]*np.diff(tenor_dates[:k+1]), axis=1)/tenor_dates[k])

    # Conditional on each path the index is lognormal: average the Black-Scholes
    # price with spot S0 U and volatility sigma*.
    bs = bs_call if option_type == 'call' else bs_put
    return np.array([np.mean(bs(S0*U_Tm, K, tenor_dates[k], r, sigma_star)) for K in strikes])