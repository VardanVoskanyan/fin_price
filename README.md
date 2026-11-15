# finpricing

Option pricing models for equity indices and the VIX, in Python. Each model comes with
pricing, simulation and calibration, a page of maths in [`docs/`](docs), and tests that
check it against closed forms and independent methods.

| Model | What it covers | Notes |
|---|---|---|
| **Heston** | Index options (Lewis integral, FFT), QE Monte Carlo, VIX futures and options, calibration | [docs/heston.md](docs/heston.md) |
| **Cont–Kokholm** | Joint model for S&P 500 and VIX options: forward variance with Merton or Kou jumps, Fourier VIX pricing, conditional Monte Carlo index pricing, calibration | [docs/cont_kokholm.md](docs/cont_kokholm.md) |
| **Black–Scholes** | Vectorised prices, Greeks and implied volatility; shared Kar–Madan / Cont–Tankov FFT pricers | [docs/black_scholes.md](docs/black_scholes.md) |

## Installation

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone https://github.com/VardanVoskanyan/fin_price.git
cd fin_price
uv sync --extra dev
```

The core library needs only numpy, scipy and pydantic. Optional extras:

| Extra | Adds | For |
|---|---|---|
| `dev` | pytest, ruff, mypy, maturin | tests and linting |
| `notebook` | ipykernel, matplotlib | the example notebooks |
| `data` | pandas, yfinance | market data and `DataFrame` input |
| `ml` | torch, torchsde, gpytorch | planned machine-learning models |

## Quickstart

### Heston

```python
import numpy as np
from finpricing import HestonModel, HestonParameters

model = HestonModel(HestonParameters(v0=0.04, kappa=1.5, theta=0.04, xi=0.5, rho=-0.7))
strikes = np.array([80, 90, 100, 110, 120])

model.option_price(S0=100, K=strikes, T=0.5, r=0.03)   # call prices
# array([21.890, 13.220, 6.007, 1.548, 0.220])
model.implied_vol(S0=100, K=strikes, T=0.5, r=0.03)    # the model smile
# array([0.255, 0.222, 0.187, 0.155, 0.142])

model.vix_future(T=0.25)                               # 0.177 (VIX in decimal units)
model.vix_option_price(K=0.20, T=0.25, r=0.03)         # 0.0283

times, S, v = model.simulate(S0=100, T=1.0, r=0.03, n_paths=10_000, seed=0)
```

### Calibration

Calibrators take an `OptionSurface`: strikes, maturities and prices as arrays, plus spot,
rate and dividend yield. `OptionSurface.from_columns` also accepts a dict or a `DataFrame`.

```python
from finpricing.calibration import HestonCalibrator, OptionSurface

K, T = np.meshgrid([85, 95, 100, 105, 115], [0.25, 0.5, 1.0])
K, T = K.ravel(), T.ravel()
surface = OptionSurface(strike=K, maturity=T, spot=100, r=0.03,
                        price=model.option_price(100, K, T, 0.03))

params, info = HestonCalibrator().calibrate(surface)
info['iv_rmse']   # below 1e-6: recovers the parameters that generated the prices
```

### Cont–Kokholm

```python
from finpricing import VixModel

ck = VixModel(model_type='Kou')                        # paper parameters (Table 2)
ck.vix_option_pricer(S0=0.20, K=0.22, T=1/12, r=0.0)   # VIX call, Fourier
ck.index_option_pricer(S0=100, strikes=strikes, tenor_index=2)   # index calls, Monte Carlo
```

### Black–Scholes

```python
from finpricing.utils.bs_utils import bs_price, implied_volatility

prices = bs_price(100, strikes, 0.5, 0.03, 0.2, option_type='put')
implied_volatility(prices, 100, strikes, 0.5, 0.03, option_type='put')   # all 0.2
```

## Notebooks

[`notebooks/01_cont_kokholm_replication.ipynb`](notebooks/01_cont_kokholm_replication.ipynb)
reproduces the main figures of Cont & Kokholm (2013): VIX smiles, the ATM term structure,
the index implied-volatility surface, and a martingale check.

```bash
uv sync --extra notebook
uv run jupyter lab
```

## Tests

```bash
uv run pytest            # fast suite
uv run pytest -m slow    # end-to-end calibrations (~20 s)
```

## Optional Rust extension

The Cont–Kokholm jump simulation has a pure-Python fallback that is used automatically.
A faster version is in `rust/` (PyO3 + rayon):

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh   # Rust toolchain, once
uv run maturin develop --release
```

## Project layout

```
src/finpricing/
├── models/
│   ├── black_scholes/   Black–Scholes model and implied-volatility solvers
│   ├── heston/          Heston pricing, simulation, VIX
│   ├── jump_diffusion/  Merton and Kou jump simulation
│   └── vix/             Cont–Kokholm VIX and index option pricing
├── calibration/         OptionSurface and model calibrators
├── data/                Paper fixtures (Cont–Kokholm Table 2)
└── utils/               Characteristic functions, Fourier pricing, Black–Scholes utilities
docs/                    Maths for each model
notebooks/               Worked examples
rust/                    Optional Rust extension for jump simulation
tests/                   pytest suite
```

## License

[MIT](LICENSE)
