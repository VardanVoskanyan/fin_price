# 𓏞 Model notes

The maths behind each model, with the formulas the code implements and the names it
uses for them.

- [Black–Scholes and implied volatility](black_scholes.md), including the shared Fourier pricers
- [Heston](heston.md): characteristic function, Lewis and FFT pricing, QE Monte Carlo, VIX, calibration
- [Cont–Kokholm](cont_kokholm.md): joint S&P 500 / VIX model with Merton or Kou jumps

Worked examples, with plots, are in [`notebooks/`](../notebooks).

## The glyphs

The docs and notebooks borrow a few Egyptian hieroglyphs (Unicode block
[U+13000–U+1342F](https://en.wikipedia.org/wiki/Egyptian_Hieroglyphs_(Unicode_block))).
Each one stands for the idea it meant to a scribe. The code is the sign's number in
Gardiner's sign list.

| Sign | Code | Meaning | Marks |
|---|---|---|---|
| 𓊽 | R11 | *djed* pillar: stability | Black–Scholes, where volatility stays constant |
| 𓈗 | N35A | water (*mw*), three ripples | Heston, whose variance ebbs and flows around its mean |
| 𓃹 | E34 | hare | Cont–Kokholm, a model of jumps |
| 𓁨 | C11 | Heh, god of millions | Monte Carlo |
| 𓍝 | U38 | balance | calibration: weighing the model against the market |
| 𓋹 | S34 | *ankh*: life | a notebook check that passed |
| 𓏞 | Y3 | scribe's kit: palette, water pot, reed holder | these notes |
| 𓏺 𓏻 𓏼 𓏽 𓏾 𓏿 | Z15–Z15E | strokes: the numerals 1 to 6 | notebook and section numbers |

They need a font with the block. macOS ships Noto Sans Egyptian Hieroglyphs, Windows has
Segoe UI Historic, and on Linux the Noto package provides it.
