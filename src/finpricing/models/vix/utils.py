"""
VIX utility functions
"""
from finpricing.parameters import VIXKouModelParameters, VIXMertonModelParameters


def vix_drift_coeff(T: float, model_params) -> float:
    """
    Computes the drift term of the variance swap rate given by the Cont&Kokholm model.
    
    Args:
        T: Time to maturity
        model_params: Model parameters (VIXMertonModelParameters or VIXKouModelParameters)
        
    Returns:
        Drift coefficient value
    """
    from finpricing.models.jump_diffusion.jump_diffusion import kou_drift, merton_drift
    
    if isinstance(model_params, VIXMertonModelParameters):
        return merton_drift(T, model_params.omega, model_params.lmbd, 
                           model_params.k1, model_params.k2, 
                           model_params.m, model_params.delta)
    elif isinstance(model_params, VIXKouModelParameters):
        return kou_drift(T, model_params.omega, model_params.lmbd,
                        model_params.k1, model_params.k2, model_params.p,
                        model_params.alpha_plus, model_params.alpha_minus)
    else:
        raise ValueError(f"Unsupported model type: {type(model_params)}")

