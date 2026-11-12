# finpricing

A Python library for pricing S&P index options and VIX derivatives, implementing the
**Cont & Kokholm (2014)** joint model alongside Black-Scholes implied-volatility tools.

## Models

| Model | Description |
|---|---|
| **Cont-Kokholm** | Simultaneous pricing of S&P and VIX derivatives via forward variance swap dynamics with Merton (Gaussian) or Kou (double-exponential) jumps |
| **Heston** | Stochastic volatility: Fourier pricing (Lewis integral or FFT), Andersen QE Monte Carlo, closed-form VIX futures and options, calibration |
| **Black-Scholes** | Analytical call/put pricing, Greeks, and implied-volatility recovery (iterative, Brentq, Newton) |

---

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone <repo-url>
cd fin_price

# create venv and install all dependencies
uv sync --extra dev
```

### Optional: Rust extension (faster MC simulation)

The jump simulation has a pure-Python fallback that works out of the box.
To build the Rust-accelerated version:

```bash
# install Rust toolchain (one-time)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# compile and install into the venv
uv run maturin develop --release
```

---

## Running the tests

```bash
uv run pytest            # fast suite
uv run pytest -m slow    # end-to-end calibrations (~20 s)
```

---

## Paper replication notebook

Reproduces the key figures from Cont & Kokholm (2014): VIX option smile,
index option implied-volatility surface, and martingale validation.

```bash
# install notebook extras (ipykernel, matplotlib) — only needed for notebooks
uv sync --extra notebook

uv run jupyter lab
# open notebooks/01_cont_kokholm_replication.ipynb
```

---

## Dashboard

Four-page Dash app: Black-Scholes pricer, VIX option smile, live SPX vol surface, and index option MC smile.

```bash
uv sync --extra app

cd app && uv run python main.py
# open http://localhost:8050
```

---

## Project structure

```
src/finpricing/
├── models/
│   ├── black_scholes/   — BS pricer, implied vol
│   ├── heston/          — Heston pricing, simulation, VIX
│   ├── jump_diffusion/  — Merton & Kou jump simulation
│   └── vix/             — Cont-Kokholm VIX & index option pricing
├── calibration/         — Parameter calibration (differential evolution)
├── data/                — Paper fixtures (Table 2 parameters)
└── utils/               — Characteristic functions, Fourier pricing, BS utilities

rust/src/lib.rs          — Optional Rust extension for jump simulation (PyO3 + rayon)
notebooks/               — Paper replication notebooks
tests/                   — pytest suite
```
