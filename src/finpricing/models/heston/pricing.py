"""
European option pricing under Heston.

Two methods:
- 'lewis': Lewis (2001) single-integral formula with Gauss-Legendre quadrature up to a
  cutoff found by probing the CF, and a closed-form tail when the CF decays slowly.
  Accurate at arbitrary strikes and vectorised over strikes; the default.
- 'fft':   the library's Kar-Madan / Cont-Tankov FFT pricer, interpolated to the strikes.
  Prices a whole strike grid at once, but is less accurate between grid points. The grid
  is sized from the decay of the CF, so it is finer for short maturities.
  Kar-Madan needs E[S_T^(1+alpha)] < inf; alpha is chosen from the moment explosion
  time unless given, and an alpha past the explosion raises.
"""
import warnings

import numpy as np
from scipy import special
from scipy.interpolate import CubicSpline

from finpricing.models.heston.characteristic import heston_cf
from finpricing.parameters import HestonParameters
from finpricing.utils.fourier import FourierMethod, fourier_call_pricer

# Probe points for the integration cutoffs, 40 per decade. |phi(u - i/2)| <= 1, so the
# Lewis envelope is below 1 / u^2 and has always fallen below 1e-14 by the end.
_PROBE_U = np.geomspace(0.1, 1e8, 361)
_LEWIS_TOL = 1e-14
# A cutoff beyond _TAIL_FROM is not integrated numerically all the way: from where phi
# has settled into its exponential large-u asymptote, the rest is done in closed form.
_TAIL_FROM = 1e4
_N_NEAR = int(np.searchsorted(_PROBE_U, _TAIL_FROM * (1 + 1e-9)))   # probe points up to it
_TAIL_TOL = 1e-12            # bound on the integral error of that closed-form tail
_MAX_UPPER = 1e5             # largest upper limit integrated numerically
_CHUNK = 1 << 21             # strikes x nodes per block of the quadrature


_PANEL_X, _PANEL_W = np.polynomial.legendre.leggauss(16)


def _composite_nodes(upper: float, max_abs_log_fk: float) -> tuple[np.ndarray, np.ndarray]:
    """16-point Gauss-Legendre on panels of width <= 2 that also resolve exp(i u ln(F/K)).

    The first panel is split geometrically, because 1 / (u^2 + 1/4) has poles at +-i/2.
    """
    width = min(2.0, np.pi / max(max_abs_log_fk, 1e-12))
    n_panels = int(np.ceil(upper / width))
    edges = np.linspace(0.0, upper, n_panels + 1)
    edges = np.concatenate([[0.0], edges[1] * np.array([0.125, 0.25, 0.5]), edges[1:]])
    half = 0.5 * np.diff(edges)[:, None]
    mid = 0.5 * (edges[:-1] + edges[1:])[:, None]
    return (mid + half * _PANEL_X).ravel(), (half * _PANEL_W).ravel()


def _scaled_expint(x: np.ndarray, n_max: int) -> list[np.ndarray]:
    """[e^x E_n(x) for n = 0..n_max] for complex x with Re x >= 0 (E_n: exponential integrals)."""
    x = np.asarray(x, dtype=complex)
    small = np.abs(x) < 60
    xs = np.where(small, x, 1.0)
    with np.errstate(all='ignore'):
        s = [np.where(xs == 0, 0.0, 1 / xs), np.exp(xs) * special.exp1(xs)]
        s[1] = np.where(xs == 0, np.inf, s[1])
        for n in range(1, n_max):                    # E_{n+1} = (e^-x - x E_n) / n
            s.append((1 - np.where(xs == 0, 0.0, xs * s[n])) / n)
    # Large |x|: asymptotic series e^x E_n(x) ~ sum_k (-1)^k (n)_k / x^(k+1).
    xl = np.where(small, 1.0, x)
    for n in range(n_max + 1):
        term = total = 1 / xl
        for k in range(16):
            term = -term * (n + k) / xl
            total = total + term
        s[n] = np.where(small, s[n], total)
    return s


def _lewis_tail(U: float, phi_U: complex, lam: complex, a: complex, log_fk: np.ndarray) -> np.ndarray:
    """Integral over [U, inf) of Re[e^{iul} phi(u - i/2)] / (u^2 + 1/4) for each l in
    ``log_fk``, with phi(u - i/2) = phi_U exp(-lam (u - U) + a (1/u - 1/U)).

    e^{a/u} / (u^2 + 1/4) is expanded in powers of 1/u, and the integral of
    e^{-z (u - U)} u^-n over [U, inf) is U^(1-n) e^{zU} E_n(zU).
    """
    coef = {2: 1, 3: a, 4: a ** 2 / 2 - 0.25, 5: a ** 3 / 6 - a / 4,
            6: a ** 4 / 24 - a ** 2 / 8 + 1 / 16}
    s = _scaled_expint((lam - 1j * log_fk) * U, max(coef))
    weight = sum(c * s[n] / U ** (n - 1) for n, c in coef.items())
    return np.real(np.exp(1j * U * log_fk) * phi_U * np.exp(-a / U) * weight)


def _lewis_upper(T: float, params: HestonParameters) -> tuple[float, tuple | None]:
    """Upper limit of the numerical Lewis integral, and the closed-form tail beyond it.

    The integrand's envelope |phi(u - i/2)| / (u^2 + 1/4) decays like a Gaussian for
    small u but only exponentially for large u, at the rate Re(lam) of
    lam = (v0 + kappa theta T)(sqrt(1 - rho^2) + i rho) / xi, which vanishes as
    rho -> +-1. The cutoff where it falls below 1e-14 is found by probing phi.

    When that cutoff is far out (beyond _TAIL_FROM), phi(u - i/2) is replaced from some
    point U on by its asymptotic form phi(U - i/2) exp(-lam (u - U) + a (1/u - 1/U)),
    whose integral is known in closed form; a is fitted to the slope of ln(phi) at U.
    U is the first probe point from which this form matches phi at all later probe
    points, up to an integral error of _TAIL_TOL. If there is none and the cutoff is
    beyond _MAX_UPPER, the integral is inaccurate and a warning is issued.
    """
    probe = _PROBE_U
    phi = heston_cf(probe[:_N_NEAR] - 0.5j, T, params)
    if np.abs(phi[-1]) / probe[_N_NEAR - 1] ** 2 > _LEWIS_TOL:     # probe further only if needed
        phi = np.concatenate([phi, heston_cf(probe[_N_NEAR:] - 0.5j, T, params)])
    envelope = np.abs(phi) / (probe[:len(phi)] ** 2 + 0.25)
    above = np.nonzero(envelope > _LEWIS_TOL)[0]
    if not len(above):
        return 1.0, None
    cutoff = float(probe[min(above[-1] + 1, len(probe) - 1)])
    if cutoff <= _TAIL_FROM:
        return cutoff, None

    cand = np.nonzero((probe >= 100) & (probe <= min(cutoff, _MAX_UPPER)))[0]
    U = probe[cand]
    lam = (params.v0 + params.kappa * params.theta * T) * (
        np.sqrt(1 - params.rho ** 2) + 1j * params.rho) / params.xi
    h = 1e-4
    with np.errstate(all='ignore'):
        up, down = heston_cf(U * (1 + h) - 0.5j, T, params), heston_cf(U * (1 - h) - 0.5j, T, params)
        slope = np.log(up / down) / (2 * U * h)                   # d ln(phi) / du at U
        a = -(slope + lam) * U ** 2
        # Error of each extrapolation at the later probe points, integrated against du / u^2.
        du = np.diff(probe, prepend=0.0)
        later = probe[None, :] > U[:, None]
        gap = np.where(later, probe - U[:, None], 0)
        pred = phi[cand, None] * np.exp(-lam * gap + a[:, None] * (1 / probe - 1 / U[:, None]))
        err = np.sum(np.where(later, np.abs(phi - pred), 0) * du / probe ** 2, axis=1)
    err = np.where(np.isfinite(err) & (np.abs(a) < 0.01 * U), err, np.inf)
    good = np.nonzero(err <= _TAIL_TOL)[0]
    if len(good):
        j = good[0]
        return float(U[j]), (float(U[j]), complex(phi[cand[j]]), complex(lam), complex(a[j]))
    if cutoff > _MAX_UPPER:
        warnings.warn(
            f"The Lewis integrand decays too slowly: it is still above {_LEWIS_TOL:g} at "
            f"u = {_MAX_UPPER:g}, and its tail is not yet exponential there. The price may "
            "be inaccurate.", RuntimeWarning, stacklevel=4)
    return min(cutoff, _MAX_UPPER), None


def _lewis_calls(S0: float, K: np.ndarray, T: float, r: float, q: float,
                 params: HestonParameters) -> np.ndarray:
    log_fk = np.log(S0 / K) + (r - q) * T                      # ln(F / K)

    upper, tail = _lewis_upper(T, params)
    u, weights = _composite_nodes(upper, float(np.max(np.abs(log_fk))))

    phi = heston_cf(u - 0.5j, T, params) * weights / (u ** 2 + 0.25)
    integral = np.empty_like(log_fk)
    step = max(1, _CHUNK // len(u))                              # bounds the memory used
    for i in range(0, len(log_fk), step):
        block = log_fk[i:i + step]
        integral[i:i + step] = np.real(np.exp(1j * np.outer(block, u)) @ phi)
    if tail is not None:
        integral += _lewis_tail(*tail, log_fk)

    return S0 * np.exp(-q * T) - np.sqrt(S0 * K) * np.exp(-0.5 * (r + q) * T) * integral / np.pi


def moment_explosion_time(omega: float, params: HestonParameters) -> float:
    """
    Time T* after which E[S_T^omega] is infinite (Andersen & Piterbarg, 2007).

    The moment is exp(A + B v0) where B solves the Riccati equation
    B' = xi^2 B^2 / 2 + (rho xi omega - kappa) B + omega (omega - 1) / 2, B(0) = 0.
    It blows up in finite time only for omega outside [0, 1].
    """
    if 0.0 <= omega <= 1.0:
        return np.inf
    b = params.rho * params.xi * omega - params.kappa
    disc = b ** 2 - params.xi ** 2 * omega * (omega - 1)
    if disc >= 0:
        if b < 0:
            return np.inf
        root = np.sqrt(disc)
        return 2 / b if root < 1e-12 * b else float(np.log((b + root) / (b - root)) / root)
    gamma = np.sqrt(-disc)
    if b == 0:
        return float(np.pi / gamma)
    return float(2 / gamma * (np.pi * (b < 0) + np.arctan(gamma / b)))


# Kar-Madan damping: the largest alpha whose (1 + alpha) moment stays finite until well
# past the maturity.
_KAR_MADAN_ALPHAS = (0.75, 0.5, 0.4, 0.3, 0.2, 0.1)
_EXPLOSION_MARGIN = 2.0

# FFT grids. The integration range N * Delta is _FFT_OVERSAMPLE times the point where the
# transformed CF falls below _FFT_TOL, and at least _FFT_MIN_RANGE. It also sets the
# log-strike spacing 2 pi / (N * Delta), so short maturities, whose CF decays slowly but
# whose smile is narrow, get a finer strike grid for the cubic spline. Delta sets the
# log-strike range 2 pi / Delta: Kar-Madan's damped price e^{alpha k} C(k) must vanish
# at its ends.
_FFT_TOL = 1e-12
_FFT_OVERSAMPLE = 4
_FFT_MIN_RANGE = 2 * np.pi / 0.02           # log-strike spacing at most 0.02
_FFT_MAX_N = 1 << 20
_CONT_TANKOV_DELTA = 0.17
_CONT_TANKOV_SIGMA = 0.3575
_FFT_KWARGS = {FourierMethod.KAR_MADAN: {"N", "Delta", "alpha"},
               FourierMethod.CONT_TANKOV: {"N", "Delta", "sigma"}}


def _kar_madan_alpha(T: float, params: HestonParameters, alpha: float | None) -> float:
    if alpha is not None:
        t_star = moment_explosion_time(1 + alpha, params)
        if t_star <= T:
            raise ValueError(
                f"Kar-Madan with alpha={alpha} needs E[S_T^{1 + alpha:g}], but that moment "
                f"is infinite from T = {t_star:.3g}; use a smaller alpha or method='lewis'.")
        return alpha
    for alpha in _KAR_MADAN_ALPHAS:
        if moment_explosion_time(1 + alpha, params) > _EXPLOSION_MARGIN * T:
            return alpha
    raise ValueError(
        f"Every Kar-Madan damping needs a moment of S_T that is infinite or close to it at "
        f"T = {T:g}; use method='lewis' or 'fft-ct'.")


def _fft_kwargs(T: float, params: HestonParameters, method: FourierMethod, fft_kwargs: dict) -> dict:
    """Grid (N, Delta) and damping alpha or reference volatility sigma for the FFT
    pricer. Values the caller gave are kept."""
    kwargs = dict(fft_kwargs)
    if method == FourierMethod.KAR_MADAN:
        kwargs["alpha"] = _kar_madan_alpha(T, params, kwargs.get("alpha"))
        shift = 1 + kwargs["alpha"]                   # the pricer evaluates phi(v - i shift)
        kwargs.setdefault("Delta", 0.1 * kwargs["alpha"])     # e^{-alpha pi / Delta} ~ 1e-14
        reference_vol = None
    else:
        shift = 1.0
        kwargs.setdefault("Delta", _CONT_TANKOV_DELTA)
        reference_vol = kwargs.setdefault("sigma", _CONT_TANKOV_SIGMA)
    if "N" not in kwargs:
        v = _PROBE_U
        envelope = (np.abs(heston_cf(v - 1j * shift, T, params))
                    / np.abs((shift - 1 + 1j * v) * (shift + 1j * v)))
        above = np.nonzero(envelope > _FFT_TOL)[0]
        cutoff = float(v[min(above[-1] + 1, len(v) - 1)]) if len(above) else 1.0
        if reference_vol is not None:                 # Cont-Tankov's Black-Scholes reference CF
            cutoff = max(cutoff, np.sqrt(2 * np.log(1 / _FFT_TOL) / (reference_vol ** 2 * T)))
        n = max(_FFT_OVERSAMPLE * cutoff, _FFT_MIN_RANGE) / kwargs["Delta"]
        if n > _FFT_MAX_N:
            warnings.warn(
                f"The FFT grid would need N = {n:.3g} points at T = {T:g}; using "
                f"N = {_FFT_MAX_N}, so prices may be inaccurate. Use method='lewis'.",
                RuntimeWarning, stacklevel=3)
        kwargs["N"] = int(2 ** np.ceil(np.log2(min(n, _FFT_MAX_N))))
    return kwargs


def _check_fft_kwargs(method: FourierMethod, fft_kwargs: dict) -> None:
    unknown = set(fft_kwargs) - _FFT_KWARGS[method]
    if unknown:
        raise TypeError(f"Unexpected keyword argument(s) {sorted(unknown)}; this method "
                        f"accepts {sorted(_FFT_KWARGS[method])}.")
    if "N" in fft_kwargs:
        N = fft_kwargs["N"]
        if not isinstance(N, int | np.integer) or N < 4:
            raise ValueError(f"N must be an integer >= 4, got {N!r}.")
        if method == FourierMethod.CONT_TANKOV and N % 2:
            raise ValueError(f"Cont-Tankov needs an even N (an odd one puts a node on the "
                             f"removable singularity at u = 0), got {N}.")
    for name in ("Delta", "alpha", "sigma"):
        if name in fft_kwargs and not fft_kwargs[name] > 0:
            raise ValueError(f"{name} must be positive, got {fft_kwargs[name]!r}.")


def _fft_calls(S0: float, K: np.ndarray, T: float, r: float, q: float,
               params: HestonParameters, method: FourierMethod, **fft_kwargs) -> np.ndarray:
    # The FFT pricers expect the CF of ln(S_T / S_0) including the drift, and return
    # call prices for S_0 = 1 on a log-strike grid.
    def cf(u):
        return np.exp(1j * u * (r - q) * T) * heston_cf(u, T, params)

    log_strikes, prices = fourier_call_pricer(cf, T, r, method=method, **fft_kwargs)
    return S0 * CubicSpline(log_strikes, prices)(np.log(K / S0))


_FOURIER_METHODS = {'fft': FourierMethod.KAR_MADAN, 'fft-ct': FourierMethod.CONT_TANKOV}


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
        **fft_kwargs: FFT settings: N, Delta and alpha for 'fft'; N, Delta and sigma
            for 'fft-ct'; none for 'lewis'. Settings not given are chosen from the
            maturity and the parameters.

    Returns:
        Price(s), with the broadcast shape of K and T (a float for scalar inputs),
        floored at the no-arbitrage lower bounds max(S e^{-qT} - K e^{-rT}, 0) for calls
        and max(K e^{-rT} - S e^{-qT}, 0) for puts.

    Raises:
        ValueError: for a spot, strike or maturity that is not positive and finite, an
            unknown option type or method, or invalid FFT settings.
        TypeError: for a keyword argument the method does not take.
    """
    if method == 'lewis':
        if fft_kwargs:
            raise TypeError(f"method='lewis' takes no extra keyword arguments, got "
                            f"{sorted(fft_kwargs)}.")
    elif method in _FOURIER_METHODS:
        _check_fft_kwargs(_FOURIER_METHODS[method], fft_kwargs)
    else:
        raise ValueError(f"Unknown method: {method!r}")
    if not (np.isfinite(S0) and S0 > 0):
        raise ValueError(f"Spot must be positive and finite, got {S0!r}.")

    K_arr, T_arr, types = np.broadcast_arrays(
        np.asarray(K, dtype=float), np.asarray(T, dtype=float), np.asarray(option_type)
    )
    shape = K_arr.shape
    K_arr, T_arr, types = K_arr.ravel(), T_arr.ravel(), types.ravel()
    if not (np.all(np.isfinite(K_arr)) and np.all(np.isfinite(T_arr))):
        raise ValueError("Strikes and maturities must be finite.")
    if np.any(K_arr <= 0) or np.any(T_arr <= 0):
        raise ValueError("Strikes and maturities must be positive.")
    if not np.all(np.isin(types, ['call', 'put'])):
        raise ValueError("option_type must be 'call' or 'put'.")

    calls = np.empty_like(K_arr)
    for T_i in np.unique(T_arr):
        idx = T_arr == T_i
        if method == 'lewis':
            calls[idx] = _lewis_calls(S0, K_arr[idx], T_i, r, q, params)
        else:
            fourier = _FOURIER_METHODS[method]
            calls[idx] = _fft_calls(S0, K_arr[idx], T_i, r, q, params, fourier,
                                    **_fft_kwargs(T_i, params, fourier, fft_kwargs))

    # Floor at the no-arbitrage bounds (quadrature noise can leave far out-of-the-money
    # prices slightly negative), then put-call parity: P = C - S e^{-qT} + K e^{-rT}.
    spot_pv, strike_pv = S0 * np.exp(-q * T_arr), K_arr * np.exp(-r * T_arr)
    calls = np.maximum(calls, np.maximum(spot_pv - strike_pv, 0.0))
    puts = np.maximum(calls - spot_pv + strike_pv, np.maximum(strike_pv - spot_pv, 0.0))
    prices = np.where(types == 'put', puts, calls)
    return float(prices[0]) if shape == () else prices.reshape(shape)
