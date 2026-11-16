"""
VIX pricing functions using various methods
"""
import numpy as np
from scipy import interpolate

from finpricing.models.vix.model import get_sigmas, get_UTm
from finpricing.utils.bs_utils import bs_call, bs_put
from finpricing.utils.characteristic_functions import characteristic_function
from finpricing.utils.fourier import FourierMethod, fourier_call_pricer


def price_vix_options(V0: float, K: float, T: float, r: float, model_params, 
                     pricing_method: str = 'Kar&Madan', option_type: str = 'call') -> float:
    """
    Returns the price of an option on VIX future using Fourier transform method.
    
    Args:
        V0: Current VIX level
        K: Strike price
        T: Time to maturity
        r: Risk-free rate
        model_params: Model parameters (VIXMertonModelParameters or VIXKouModelParameters)
        pricing_method: Fourier pricing method ('Cont&Tankov' or 'Kar&Madan')
        option_type: Option type ('call' or 'put')
        
    Returns:
        Option price
    """
    moneyness = K/V0

    if option_type == 'call':
        def chr_func(x):
            return characteristic_function(x, T, model_params, asset_type='VIX')
        
        # Convert string method to enum
        method = FourierMethod.CONT_TANKOV if pricing_method == 'Cont&Tankov' else FourierMethod.KAR_MADAN
        us, ops = fourier_call_pricer(chr_func, T, r, method=method)

        # Interpolate to get prices for moneyness values between the points of np.exp(us)
        f = interpolate.interp1d(np.exp(us), ops)

        return V0*f(moneyness)

    elif option_type == 'put':
        # V0 is the VIX future (a martingale) and prices are discounted at r:
        # C - P = exp(-rT) (V0 - K).
        call_price = price_vix_options(V0, K, T, r, model_params, pricing_method, 'call')
        return call_price - np.exp(-r*T) * (V0 - K)
    
    else:
        raise ValueError(f"Unknown option type: {option_type}")


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
    
    from finpricing.parameters import VIXKouModelParameters, VIXMertonModelParameters
    match model_params:
        case VIXKouModelParameters():
            constraint_cf = 2 * model_params.lmbd * (model_params.p / model_params.alpha_plus**2 +
                               (1 - model_params.p) / model_params.alpha_minus**2)
        case VIXMertonModelParameters():
            constraint_cf = model_params.lmbd * (model_params.m**2 + model_params.delta**2)
    
    sigmas_sq = get_sigmas(Vi_0[:k], VTi[:,:k], constraint_cf, b_i[:k])
    
    # Compute U_T_k and sigma*
    U_Tm = get_UTm(Vi_0[:k], VTi[:,:k], diff_Z[:,:k], jumps_S[:,:k], sigmas_sq[:,:k],
                   b_i[:k], tenor_dates[:k+1], model_params)

    sigma_star = np.sqrt((1 - rho**2)*np.sum(sigmas_sq[:,:k]*np.diff(tenor_dates[:k+1]), axis=1)/tenor_dates[k])

    # Conditional on each path the index is lognormal: average the Black-Scholes
    # price with spot S0 U and volatility sigma*.
    bs = bs_call if option_type == 'call' else bs_put
    return np.array([np.mean(bs(S0*U_Tm, K, tenor_dates[k], r, sigma_star)) for K in strikes])