# Black–Scholes and implied volatility

Black–Scholes is the quoting convention for the rest of the library: market and model
prices are compared through their implied volatilities, and calibration losses are
weighted by Black–Scholes vega.

Code: `finpricing.utils.bs_utils` (vectorised functions), `finpricing.models.black_scholes`
(`BlackScholesModel`, `ImpliedVolatilityCalculator`).

## Prices

With spot $S$, strike $K$, maturity $T$, rate $r$, dividend yield $q$ and volatility $\sigma$:

$$
d_{1,2} = \frac{\ln(S/K) + (r - q \pm \tfrac12\sigma^2)T}{\sigma\sqrt T},
$$

$$
C = S e^{-qT} N(d_1) - K e^{-rT} N(d_2), \qquad
P = K e^{-rT} N(-d_2) - S e^{-qT} N(-d_1).
$$

`bs_price(s, k, t, r, sigma, q=0, option_type='call')` broadcasts over all its arguments,
including `option_type`.

## Greeks

| Greek | Call | Put |
|---|---|---|
| Delta $\partial_S$ | $e^{-qT} N(d_1)$ | $-e^{-qT} N(-d_1)$ |
| Gamma $\partial_{SS}$ | $\dfrac{e^{-qT} n(d_1)}{S\sigma\sqrt T}$ | same |
| Vega $\partial_\sigma$ | $S e^{-qT} n(d_1)\sqrt T$ | same |
| Rho $\partial_r$ | $K T e^{-rT} N(d_2)$ | $-K T e^{-rT} N(-d_2)$ |

`bs_vega` is vectorised and used by the calibrators. `BlackScholesModel.greeks` returns
all five Greeks for one option through the `PricingModel` interface.

## Implied volatility

`implied_volatility(price, s, k, t, r, q=0, option_type='call')` inverts Black–Scholes with
Brent's method on $[10^{-4}, 5]$ for each option. Quotes outside the no-arbitrage bounds

$$
\max(S e^{-qT} - K e^{-rT}, 0) \le C \le S e^{-qT}
$$

(and the corresponding put bounds), or that would need a volatility outside the bracket,
return `NaN` rather than raising, so a whole surface can be inverted in one call and the
bad quotes filtered afterwards.

`ImpliedVolatilityCalculator` (calls, no dividends) compares several solvers on one option:

| `Method` | Algorithm |
|---|---|
| `ITERATIVE` | bisection on $[0, 5]$ with a relative price tolerance |
| `BRENTQ` | Brent's method |
| `NEWTON` | Newton–Raphson with vega as the derivative |
| `HYBR`, `LM` | `scipy.optimize.root` (Powell hybrid, Levenberg–Marquardt) |

Newton and the `root` methods start from an initial guess: `'corrado_miller'` (Corrado &
Miller 1996) or `'improved'`, a variant that stays defined away from the money. Newton
converges fastest near the money but can fail far from it, where vega is close to zero;
bracketing methods (bisection, Brent) always converge for quotes inside the bounds.

## Fourier pricing

`finpricing.utils.fourier.fourier_call_pricer` turns a characteristic function into call
prices on a log-strike grid by FFT. It is shared by the Cont–Kokholm VIX pricer and the
Heston FFT methods:

- **Kar–Madan** (Carr & Madan 1999): FFT of the call price damped by $e^{\alpha k}$.
- **Cont–Tankov** (Cont & Tankov 2004): FFT of the difference between the model price and
  a Black–Scholes price with the same forward. The difference decays faster than the price
  itself; the Black–Scholes price is added back in closed form.

The test suite checks that both reproduce Black–Scholes when given its characteristic
function, with rates, dividends and spot different from 1.
