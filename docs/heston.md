# Heston stochastic volatility

Heston (1993). Variance is a mean-reverting square-root process correlated with the index.
It produces a skew through $\rho$ and smile curvature through $\xi$, and has a closed-form
characteristic function, so vanilla options are priced by Fourier methods.

Code: `finpricing.models.heston` (`HestonModel`, `heston_price`, `simulate_heston`,
`vix_future`, `vix_option_price`), `finpricing.calibration.HestonCalibrator`.

## Dynamics

$$
\begin{aligned}
dS_t &= (r - q) S_t\,dt + \sqrt{v_t}\, S_t\, dW^S_t, \\
dv_t &= \kappa(\theta - v_t)\,dt + \xi \sqrt{v_t}\, dW^v_t, \qquad d\langle W^S, W^v \rangle_t = \rho\,dt.
\end{aligned}
$$

| Symbol | Code | Meaning | Typical index value |
|---|---|---|---|
| $v_0$ | `v0` | initial variance | 0.01–0.09 |
| $\kappa$ | `kappa` | speed of mean reversion | 0.5–5 |
| $\theta$ | `theta` | long-run variance | 0.02–0.09 |
| $\xi$ | `xi` | volatility of variance | 0.2–1.5 |
| $\rho$ | `rho` | spot–variance correlation | −0.9 to −0.5 |

The **Feller condition** $2\kappa\theta \ge \xi^2$ keeps $v$ strictly positive
(`HestonParameters.feller_satisfied`). Calibrated index parameters often violate it; the
pricing methods below do not need it.

## Characteristic function

For $X_T = \ln(S_T/S_0) - (r-q)T$, in the "little Heston trap" form of Albrecher et al.
(2007), which stays continuous for long maturities:

$$
\begin{aligned}
\beta &= \kappa - \rho\xi\, iu, \qquad d = \sqrt{\beta^2 + \xi^2 (iu + u^2)}, \qquad
g = \frac{\beta - d}{\beta + d}, \\
C &= \frac{\kappa\theta}{\xi^2}\left[ (\beta - d) T - 2 \ln\frac{1 - g e^{-dT}}{1 - g} \right], \qquad
D = \frac{\beta - d}{\xi^2}\cdot\frac{1 - e^{-dT}}{1 - g e^{-dT}}, \\
\varphi(u) &= \mathbb{E}\left[e^{iuX_T}\right] = \exp(C + D v_0).
\end{aligned}
$$

$X_T$ is the martingale part of the log price, so $\varphi(-i) = 1$ (`heston_cf`).

## Vanilla options

`heston_price(..., method=...)`:

**`'lewis'` (default).** Lewis (2001) single-integral formula, with $F = S_0 e^{(r-q)T}$:

$$
C = S_0 e^{-qT} - \frac{\sqrt{S_0 K}\, e^{-(r+q)T/2}}{\pi}
\int_0^\infty \mathrm{Re}\left[ e^{iu \ln(F/K)}\, \varphi\left(u - \tfrac{i}{2}\right) \right]
\frac{du}{u^2 + \tfrac14}.
$$

The integral is cut off where the integrand's envelope falls below $10^{-14}$, found by
probing $\varphi$ numerically, because the decay rate changes from Gaussian to exponential
(rate about $1/\xi$) as $u$ grows. It is computed with 16-point Gauss–Legendre panels
narrow enough to resolve the oscillation $e^{iu\ln(F/K)}$. All strikes of one maturity
share the same nodes. The result agrees with an independent implementation to about
$10^{-8}$.

**`'fft'`, `'fft-ct'`.** The library's Kar–Madan and Cont–Tankov FFT pricers (shared with
Cont–Kokholm), applied to the CF of $\ln(S_T/S_0)$ and interpolated to the strikes with a
cubic spline. They price a whole strike grid at once but are less accurate between grid
points.

Puts use put–call parity, $P = C - S_0 e^{-qT} + K e^{-rT}$.
`HestonModel.implied_vol` returns the Black–Scholes implied volatility of out-of-the-money
options, which is the model smile.

## Monte Carlo

`simulate_heston` uses Andersen's (2008) Quadratic-Exponential scheme. Given $v_t$, the
next variance has conditional mean $m$ and variance $s^2$ known in closed form. With
$\psi = s^2/m^2$:

- $\psi \le 1.5$: $v_{t+\Delta} = a(b + Z)^2$, a moment-matched squared Gaussian;
- $\psi > 1.5$: a point mass at zero plus an exponential tail, sampled by inverting its CDF.

The log price uses the scheme's central discretisation ($\gamma_1 = \gamma_2 = \tfrac12$),
which correlates its Gaussian with the variance increment. QE stays accurate with a few
dozen steps per year even when Feller fails, where Euler schemes are biased.
`simulate_heston` returns `(times, S, v)` with paths in rows.

## VIX

The VIX is the 30-day expected realised variance. Under Heston it is affine in $v_t$:

$$
\mathrm{VIX}_t^2 = a\, v_t + b, \qquad
a = \frac{1 - e^{-\kappa\tau}}{\kappa\tau}, \quad b = \theta(1 - a), \quad \tau = \tfrac{30}{365}.
$$

$v_T$ given $v_0$ is a scaled non-central chi-squared variable, $v_T = c\,X$ with
$X \sim \chi'^2(\delta, \lambda)$ and

$$
c = \frac{\xi^2 (1 - e^{-\kappa T})}{4\kappa}, \qquad \delta = \frac{4\kappa\theta}{\xi^2}, \qquad
\lambda = \frac{v_0 e^{-\kappa T}}{c}.
$$

So VIX futures $\mathbb{E}[\mathrm{VIX}_T]$ and calls $e^{-rT}\mathbb{E}[(\mathrm{VIX}_T - K)^+]$
are one-dimensional integrals against the $\chi'^2$ density (`vix_future`,
`vix_option_price`). VIX puts use parity with the future. VIX levels are in decimal units
(0.20, not 20).

Heston cannot match the VIX smile and the S&P smile at the same time; the VIX smile it
produces slopes the wrong way. That gap is what the Cont–Kokholm model addresses, and the
Heston VIX functions are mainly useful for showing it.

## Calibration

`HestonCalibrator().calibrate(surface)` fits $(v_0, \kappa, \theta, \xi, \rho)$ to an
`OptionSurface` of index options. The loss is

$$
L = \sum_j w_j \frac{\left(C^{\text{model}}_j - C^{\text{mkt}}_j\right)^2}{\mathcal{V}_j^2},
$$

where $\mathcal{V}_j$ is the Black–Scholes vega at the market implied volatility, floored
at `min_vega * spot`. To first order this is the squared implied-volatility error, without
inverting Black–Scholes at every iteration. Differential evolution over the bounds below
is followed by L-BFGS-B.

| Parameter | Bounds |
|---|---|
| `v0`, `theta` | [1e-4, 1] |
| `kappa` | [0.01, 10] |
| `xi` | [0.01, 3] |
| `rho` | [−0.99, 0.99] |

It returns the fitted `HestonParameters` and `{'loss', 'iv_rmse', 'feller'}`, where
`iv_rmse` is the exact implied-volatility RMSE.

## References

- S. Heston. *A closed-form solution for options with stochastic volatility with
  applications to bond and currency options.* Review of Financial Studies 6(2), 1993.
- H. Albrecher, P. Mayer, W. Schoutens, J. Tistaert. *The little Heston trap.*
  Wilmott Magazine, 2007.
- A. Lewis. *A simple option formula for general jump-diffusion and other exponential
  Lévy processes.* 2001.
- L. Andersen. *Simple and efficient simulation of the Heston stochastic volatility
  model.* Journal of Computational Finance 11(3), 2008.
