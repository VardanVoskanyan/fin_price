# 𓈗 Heston stochastic volatility

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

As $\xi \to 0$, $\beta - d = O(\xi^2)$ and is then divided by $\xi^2$, so subtracting
$d$ from $\beta$ loses all precision. `heston_cf` uses
$\beta - d = -\xi^2(iu + u^2)/(\beta + d)$ instead (and the same identity the other way
round where $\beta + d$ is the small one), writes $\ln\frac{1 - g e^{-dT}}{1 - g}$ as
$\ln(1 + x)$ with $x = (\beta - d)(1 - e^{-dT})/(2d)$, and evaluates $\ln(1 + x)$ and
$1 - e^{-dT}$ without cancellation. Prices then reach the Black–Scholes limit to
$10^{-8}$ even at $\xi = 10^{-8}$.

## Vanilla options

`heston_price(..., method=...)`:

**`'lewis'` (default).** Lewis (2001) single-integral formula, with $F = S_0 e^{(r-q)T}$:

$$
C = S_0 e^{-qT} - \frac{\sqrt{S_0 K}\, e^{-(r+q)T/2}}{\pi}
\int_0^\infty \mathrm{Re}\left[ e^{iu \ln(F/K)}\, \varphi\left(u - \tfrac{i}{2}\right) \right]
\frac{du}{u^2 + \tfrac14}.
$$

The integral is cut off where the integrand's envelope falls below $10^{-14}$, found by
probing $\varphi$ numerically up to $u = 10^8$, because the decay rate changes from
Gaussian to exponential as $u$ grows. For large $u$,
$\ln\varphi(u) \approx c - \lambda u$ with
$\lambda = (v_0 + \kappa\theta T)(\sqrt{1-\rho^2} + i\rho)/\xi$, which decays very slowly
as $\rho \to \pm 1$: at the corner of the calibrator's bounds the cutoff is near
$u = 10^6$. It is computed with 16-point Gauss–Legendre panels narrow enough to resolve
the oscillation $e^{iu\ln(F/K)}$, the first one split towards the poles of
$1/(u^2 + \frac14)$ at $\pm i/2$. All strikes of one maturity share the same nodes, and
strikes are processed in blocks to bound memory.

When the cutoff is beyond $u = 10^4$, the panels stop at the first probe point $U$ from
which $\varphi(U - \frac i2)\, e^{-\lambda(u - U) + a(1/u - 1/U)}$, with $a$ fitted to
the slope of $\ln\varphi$ at $U$, matches $\varphi$ at every later probe point. The rest
of the integral is done in closed form, with exponential integrals $E_n$. If no such $U$
exists below $u = 10^5$ (only near $\rho = \pm 1$ exactly), the panels stop at
$10^5$ and a `RuntimeWarning` says the price may be inaccurate.

The result agrees with an independent implementation to about $10^{-10}$.

**`'fft'`, `'fft-ct'`.** The library's Kar–Madan and Cont–Tankov FFT pricers (shared with
Cont–Kokholm), applied to the CF of $\ln(S_T/S_0)$ and interpolated to the strikes with a
cubic spline. They price a whole strike grid at once but are less accurate between grid
points. The grid depends on the maturity: the integration range $N\Delta$ is four times
the point where the transformed CF falls below $10^{-12}$ (and at least $2\pi/0.02$). The
log-strike spacing $2\pi/(N\Delta)$ therefore shrinks with the width of the smile, which
a fixed grid cannot resolve at short maturities. Kar–Madan's damping $\alpha$ is the
largest of 0.75, 0.5, …, 0.1 whose moment $\mathbb{E}[S_T^{1+\alpha}]$ stays finite
until twice the maturity, and $\Delta = \alpha/10$ keeps the damped price
$e^{\alpha k}C(k)$ negligible at the ends of the log-strike range $2\pi/\Delta$.
Cont–Tankov uses $\Delta = 0.17$. Both match Lewis to $10^{-3}$ relative at one-day
maturities, and much better at longer ones.

`'fft'` accepts `N`, `Delta` and `alpha`, `'fft-ct'` accepts `N`, `Delta` and `sigma`
(an even `N`), and `'lewis'` takes no extra arguments; anything else raises `TypeError`.

Puts use put–call parity, $P = C - S_0 e^{-qT} + K e^{-rT}$. Prices are floored at the
no-arbitrage lower bounds $\max(S_0 e^{-qT} - K e^{-rT}, 0)$ for calls and
$\max(K e^{-rT} - S_0 e^{-qT}, 0)$ for puts; quadrature noise (about $10^{-12}$ for
Lewis, more for the FFT methods) would otherwise leave some far out-of-the-money prices
slightly negative.

`HestonModel.implied_vol` returns the Black–Scholes implied volatility of out-of-the-money
options, which is the model smile.

## 𓁨 Monte Carlo

`simulate_heston` uses Andersen's (2008) Quadratic-Exponential scheme. Given $v_t$, the
next variance has conditional mean $m$ and variance $s^2$ known in closed form. With
$\psi = s^2/m^2$:

- $\psi \le 1.5$: $v_{t+\Delta} = a(b + Z)^2$, a moment-matched squared Gaussian;
- $\psi > 1.5$: a point mass at zero plus an exponential tail, sampled by inverting its CDF.

The log price uses the scheme's central discretisation ($\gamma_1 = \gamma_2 = \tfrac12$),
which correlates its Gaussian with the variance increment:

$$
\ln S_{t+\Delta} = \ln S_t + (r-q)\Delta + K_0^* + K_1 v_t + K_2 v_{t+\Delta}
+ \sqrt{K_3 v_t + K_4 v_{t+\Delta}}\, Z.
$$

**Martingale correction (QE-M).** With the plain drift constant $K_0$ the discrete
forward is biased when steps are coarse (about 1% at $T = 5$ with 4 steps for
$\xi = 1.5$, $\rho = -0.9$). Following Andersen §4.3, $K_0$ is replaced per path and step
by

$$
K_0^* = -\ln \mathbb{E}\left[e^{A v_{t+\Delta}} \mid v_t\right] - \left(K_1 + \tfrac12 K_3\right) v_t,
\qquad A = K_2 + \tfrac12 K_4,
$$

so that $\mathbb{E}[S_{t+\Delta} \mid S_t, v_t] = S_t e^{(r-q)\Delta}$ exactly. The
expectation is taken under the distribution QE actually samples:

- quadratic branch: $\dfrac{\exp\left(A b^2 a / (1 - 2Aa)\right)}{\sqrt{1 - 2Aa}}$, finite for $A < 1/(2a)$;
- exponential branch: $p + \dfrac{\beta(1-p)}{\beta - A}$, finite for $A < \beta$.

$A$ is negative for the usual $\rho < 0$, so both always hold. Where one fails (large
positive $\rho$ with coarse steps) that path keeps the uncorrected $K_0$ for that step.

QE stays accurate with a few dozen steps per year even when Feller fails, where Euler
schemes are biased. `simulate_heston` returns `(times, S, v)` with paths in rows.

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

The integrals run between the $10^{-12}$ and $1 - 10^{-12}$ quantiles of $X$, with a
breakpoint at its mean. When $\lambda$ is large (small $\xi$ or short $T$) the density is a
narrow spike, which an integral over $[0, \infty)$ misses, and scipy's density is NaN far out
in its left tail. Above $\lambda = 10^8$ the skewness of $X$, about $3/\sqrt{\lambda}$, is
negligible, so $v_T$ is taken as normal with its exact mean and variance. $T = 0$ gives the
spot VIX, and $T < 0$ raises `ValueError`.

Heston cannot match the VIX smile and the S&P smile at the same time; the VIX smile it
produces slopes the wrong way. That gap is what the Cont–Kokholm model addresses, and the
Heston VIX functions are mainly useful for showing it.

## 𓍝 Calibration

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
