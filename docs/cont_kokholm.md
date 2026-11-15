# Cont–Kokholm: joint S&P 500 and VIX model

Cont & Kokholm (2013), *A consistent pricing model for index options and volatility
derivatives*, Mathematical Finance 23(2). The model specifies the dynamics of forward
variance swap rates directly, and builds the index on top of them, so S&P options and
VIX options are priced in one model with one parameter set.

Code: `finpricing.models.vix` (`VixModel`), `finpricing.models.jump_diffusion`,
`finpricing.utils.characteristic_functions`, `finpricing.calibration.ContKokholmCalibrator`.
Paper parameters (Table 2) are in `finpricing.data.fixtures`.

## Forward variance swap rates

The forward variance swap rate quoted at $t$ for the period $[T_1, T_2]$ is the strike
that gives a forward-start variance swap zero value:

$$
V^{T_1,T_2}_t = \frac{\mathbb{E}\left[[\log S]_{T_2} - [\log S]_{T_1} \mid \mathcal{F}_t\right]}{T_2 - T_1}.
$$

It is a martingale under the pricing measure. On a tenor grid
$0 = T_0 < T_1 < \dots < T_m$ write $V^i_t = V^{T_i, T_{i+1}}_t$.

## Variance dynamics

Each $V^i$ is modelled as a positive martingale driven by one Brownian motion $Z$ and one
Poisson random measure $J(dx\,ds)$ with compensator $\nu(dx)\,ds$:

$$
V^i_t = V^i_0 \exp\left\{ \int_0^t \mu^i_s\,ds
  + \int_0^t \omega e^{-k_1 (T_i - s)}\,dZ_s
  + \int_0^t\!\!\int_{\mathbb{R}} e^{-k_2 (T_i - s)}\,x\,J(dx\,ds) \right\},
  \qquad t \le T_i,
$$

and $V^i_t = V^i_{T_i}$ for $t > T_i$. Shocks far from the start of the period are damped:
$k_1$ controls how fast the diffusive effect decays with time to $T_i$, $k_2$ does the
same for jumps. The martingale condition fixes the drift:

$$
\mu^i_t = -\tfrac12 \omega^2 e^{-2 k_1 (T_i - t)}
  - \int_{\mathbb{R}} \left( e^{e^{-k_2 (T_i - t)} x} - 1 \right) \nu(dx).
$$

Integrated over $[0, T]$, the diffusive part is
$-\frac{\omega^2}{2}\cdot\frac{1 - e^{-2 k_1 T}}{2 k_1}$ and the jump part is
$-\lambda \int_0^T \left(\mathbb{E}\left[e^{e^{-k_2 (T-s)} Y}\right] - 1\right) ds$, with
$\mathbb{E}[e^{cY}] = e^{cm + c^2\delta^2/2}$ (Merton) or
$p\frac{\alpha_+}{\alpha_+ - c} + (1-p)\frac{\alpha_-}{\alpha_- + c}$ (Kou).
This is `vix_drift_coeff` in the code.

### Jump distributions

With jump intensity $\lambda$, $\nu(dx) = \lambda f(x)\,dx$ where

| Model | Jump density $f(x)$ | $\int x^2 \nu(dx)$ |
|---|---|---|
| Merton (Gaussian) | $\frac{1}{\delta\sqrt{2\pi}} e^{-(x-m)^2 / 2\delta^2}$ | $\lambda (m^2 + \delta^2)$ |
| Kou (double exponential) | $p\,\alpha_+ e^{-\alpha_+ x}\mathbf 1_{x \ge 0} + (1-p)\,\alpha_- e^{\alpha_- x}\mathbf 1_{x<0}$ | $2\lambda\left(\frac{p}{\alpha_+^2} + \frac{1-p}{\alpha_-^2}\right)$ |

Jumps in variance are mostly upward ($m > 0$, or $p$ close to 1), which produces the
upward-sloping VIX smile.

## VIX options

The VIX at $T$ is approximated by the square root of the variance swap rate starting at
$T$, so $\ln(\mathrm{VIX}_T / \mathrm{VIX}_0) = \tfrac12 X_T$, where $X_T$ is the exponent above.
Its characteristic function is available in closed form up to one-dimensional integrals
over $s$ (`characteristic_function(u, T, params, asset_type='VIX')`, which uses the
coefficients $\tfrac12$ and $\tfrac18$; `asset_type='var_swap'` uses $1$ and $\tfrac12$).

VIX calls are priced by Fourier inversion (`price_vix_options`):

- **Kar–Madan** (Carr–Madan with damping): FFT of the damped call price.
- **Cont–Tankov**: FFT of the difference between the model price and a Black–Scholes
  price with the same forward, which is then added back.

Both return prices for unit spot on a log-strike grid, interpolated to the requested
strikes. Puts follow from put–call parity. The underlying passed as `V0` is treated as a
martingale, so for a given expiry it should be the VIX futures level for that expiry.

## Index options

Over $[T_i, T_{i+1}]$ the index has variance set by $V^i_{T_i}$. Index jumps happen at the
same times as variance jumps, with size proportional to them:

$$
\frac{dS_t}{S_{t-}} = \sigma_i \left( \rho\,dZ_t + \sqrt{1-\rho^2}\,dW_t \right)
  + \int \left(e^{u_i x} - 1\right) \tilde J(dx\,dt),
\qquad u_i = b_i \sqrt{V^i_{T_i} / V^i_0},
$$

where $W \perp Z$ and $\tilde J$ is the compensated measure. $\sigma_i$ is chosen so that
the total quadratic variation over the period matches the variance swap rate:

$$
\sigma_i^2 = V^i_{T_i} - \frac{V^i_{T_i}}{V^i_0}\, b_i^2 \int x^2 \nu(dx).
$$

Conditional on the paths of $Z$ and $J$, $\ln S_{T_k}$ is Gaussian. So an index call is a
Black–Scholes price averaged over simulated variance paths:

$$
C(K, T_k) = \mathbb{E}\left[ C_{BS}\left(S_0 U_{T_k}, K, T_k, r, \sigma^*\right) \right],
\qquad
\sigma^{*2} = \frac{1-\rho^2}{T_k} \sum_{i<k} \sigma_i^2 (T_{i+1} - T_i),
$$

where $U_{T_k}$ collects the drift, the $\rho\,dZ$ part and the index jumps
(`get_UTm`). Only $Z$ and the jumps are simulated; the $W$ part is integrated exactly.
`VixModel.store_tenor_data` simulates and caches these paths once;
`index_option_pricer` reuses them for every strike.

## Parameters

| Symbol | Code | Meaning |
|---|---|---|
| $\omega$ | `omega` | volatility of variance |
| $k_1$ | `k1` | decay rate of diffusive shocks |
| $k_2$ | `k2` | decay rate of jump shocks |
| $\lambda$ | `lmbd` | jump intensity (per year) |
| $\rho$ | `rho` | correlation between index and variance diffusions |
| $m, \delta$ | `m`, `delta` | Merton jump mean and standard deviation |
| $p, \alpha_+, \alpha_-$ | `p`, `alpha_plus`, `alpha_minus` | Kou up-jump probability and rates |
| $V^i_0$ | `VI_0` | initial forward variance swap rates |
| $b_i$ | `MERTON_B_I`, `KOU_B_I` | index jump loading per tenor period |

## Calibration

`ContKokholmCalibrator('Merton' | 'Kou')` fits the parameters to VIX option quotes given
as an `OptionSurface` (its `spot` is the VIX level). The loss is the weighted sum of
squared price errors; differential evolution is followed by L-BFGS-B.

## Reference

R. Cont, T. Kokholm. *A consistent pricing model for index options and volatility
derivatives.* Mathematical Finance 23(2), 248–274, 2013.
