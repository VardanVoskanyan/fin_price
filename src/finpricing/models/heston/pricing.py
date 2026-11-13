"""
European option pricing under Heston.

Two methods:
- 'lewis': Lewis (2001) single-integral formula with fixed Gauss-Legendre quadrature.
  Accurate at arbitrary strikes and vectorised over strikes; the default.
- 'fft':   the library's Kar-Madan / Cont-Tankov FFT pricer, interpolated to the strikes.
  Prices a whole strike grid at once, but is less accurate between grid points.
"""
import numpy as np
from scipy.interpolate import CubicSpline

from finpricing.models.heston.characteristic import heston_cf
from finpricing.parameters import HestonParameters
from finpricing.utils.fourier import FourierMethod, fourier_call_pricer

_PROBE_U = np.geomspace(0.1, 1e4, 200)


_PANEL_X, _PANEL_W = np.polynomial.legendre.leggauss(16)


def _composite_nodes(upper: float, max_abs_log_fk: float) -> tuple[np.ndarray, np.ndarray]:
    """16-point Gauss-Legendre on panels of width <= 2 that also resolve exp(i u ln(F/K))."""
    width = min(2.0, np.pi / max(max_abs_log_fk, 1e-12))
    n_panels = int(np.ceil(upper / width))
    edges = np.linspace(0.0, upper, n_panels + 1)
    half = 0.5 * np.diff(edges)[:, None]
    mid = 0.5 * (edges[:-1] + edges[1:])[:, None]
    return (mid + half * _PANEL_X).ravel(), (half * _PANEL_W).ravel()


def _integration_cutoff(T: float, params: HestonParameters, tol: float = 1e-14) -> float:
    """Smallest u beyond which |phi(u - i/2)| / (u^2 + 1/4) stays below ``tol``.

    The Heston CF decays like a Gaussian for small u but only exponentially for
    large u (rate ~ 1/xi), so the cutoff is found numerically, not from a formula.
    """
    envelope = np.abs(heston_cf(_PROBE_U - 0.5j, T, params)) / (_PROBE_U ** 2 + 0.25)
    above = np.nonzero(envelope > tol)[0]
    return float(_PROBE_U[min(above[-1] + 1, len(_PROBE_U) - 1)]) if len(above) else 1.0


def _lewis_calls(S0: float, K: np.ndarray, T: float, r: float, q: float,
                 params: HestonParameters) -> np.ndarray:
    log_fk = np.log(S0 / K) + (r - q) * T                      # ln(F / K)

    upper = _integration_cutoff(T, params)
    u, weights = _composite_nodes(upper, float(np.max(np.abs(log_fk))))

    phi = heston_cf(u - 0.5j, T, params)                         # (N,)
    integrand = np.real(np.exp(1j * np.outer(log_fk, u)) * phi) / (u ** 2 + 0.25)
    integral = integrand @ weights                               # (M,)

    return S0 * np.exp(-q * T) - np.sqrt(S0 * K) * np.exp(-0.5 * (r + q) * T) * integral / np.pi


def _fft_calls(S0: float, K: np.ndarray, T: float, r: float, q: float,
               params: HestonParameters, method: FourierMethod, **fft_kwargs) -> np.ndarray:
    # The FFT pricers expect the CF of ln(S_T / S_0) including the drift, and return
    # call prices for S_0 = 1 on a log-strike grid.
    def cf(u):
        return np.exp(1j * u * (r - q) * T) * heston_cf(u, T, params)

    log_strikes, prices = fourier_call_pricer(cf, T, r, method=method, **fft_kwargs)
    return S0 * CubicSpline(log_strikes, prices)(np.log(K / S0))


def heston_price(
    S0: float,
    K: np.ndarray | float,
    T: np.ndarray | float,
    r: float,
    params: HestonParameters,
    q: float = 0.0,
    option_type: np.ndarray | str = 'call',
    method: str = 'lewis',
    **fft_kwargs,
) -> np.ndarray | float:
    """
    Price European options under Heston.

    Args:
        S0: Spot price.
        K: Strike(s).
        T: Maturity/maturities in years; broadcast against K.
        r: Risk-free rate (continuous).
        params: Heston parameters.
        q: Dividend yield (continuous).
        option_type: 'call' / 'put', or an array of them broadcast against K.
        method: 'lewis' (default), 'fft' (Kar-Madan) or 'fft-ct' (Cont-Tankov).

    Returns:
        Price(s), with the broadcast shape of K and T (a float for scalar inputs).
    """
    K_arr, T_arr, types = np.broadcast_arrays(
        np.asarray(K, dtype=float), np.asarray(T, dtype=float), np.asarray(option_type)
    )
    shape = K_arr.shape
    K_arr, T_arr, types = K_arr.ravel(), T_arr.ravel(), types.ravel()
    if np.any(K_arr <= 0) or np.any(T_arr <= 0):
        raise ValueError("Strikes and maturities must be positive.")
    if not np.all(np.isin(types, ['call', 'put'])):
        raise ValueError("option_type must be 'call' or 'put'.")

    calls = np.empty_like(K_arr)
    for T_i in np.unique(T_arr):
        idx = T_arr == T_i
        match method:
            case 'lewis':
                calls[idx] = _lewis_calls(S0, K_arr[idx], T_i, r, q, params)
            case 'fft':
                calls[idx] = _fft_calls(S0, K_arr[idx], T_i, r, q, params,
                                        FourierMethod.KAR_MADAN, **fft_kwargs)
            case 'fft-ct':
                calls[idx] = _fft_calls(S0, K_arr[idx], T_i, r, q, params,
                                        FourierMethod.CONT_TANKOV, **fft_kwargs)
            case _:
                raise ValueError(f"Unknown method: {method!r}")

    # Put-call parity: P = C - S e^{-qT} + K e^{-rT}
    puts = calls - S0 * np.exp(-q * T_arr) + K_arr * np.exp(-r * T_arr)
    prices = np.where(types == 'put', puts, calls)
    return float(prices[0]) if shape == () else prices.reshape(shape)
