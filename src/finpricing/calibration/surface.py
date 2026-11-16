"""
Option surface: the market quotes a calibrator fits to.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Slack on the no-arbitrage bounds, relative to max(S e^{-qT}, K e^{-rT}), for rounding.
_BOUND_RTOL = 1e-10


def _require(ok: np.ndarray, message: str) -> None:
    """Raise ValueError naming the quotes (by index) where ``ok`` is False."""
    bad = np.flatnonzero(~np.asarray(ok, dtype=bool))
    if bad.size:
        shown = ", ".join(str(i) for i in bad[:10]) + (", ..." if bad.size > 10 else "")
        raise ValueError(f"{message}; failing quote indices: [{shown}].")


@dataclass(frozen=True)
class OptionSurface:
    """
    A set of European option quotes on one underlying, stored as numpy arrays.

    Every value must be finite, weights non-negative with a positive sum, no two
    quotes may share (strike, maturity, option_type), and every price must lie
    within the model-free no-arbitrage bounds

        calls: max(0, S e^{-qT} - K e^{-rT}) <= C <= S e^{-qT}
        puts:  max(0, K e^{-rT} - S e^{-qT}) <= P <= K e^{-rT}

    up to a relative tolerance of 1e-10 for floating-point rounding.

    Options on a future (VIX options, for example, are written on the VIX
    future, not on spot VIX): pass the future price F as ``spot`` and set
    ``q = r``. Then S e^{-qT} = F e^{-rT}, the discounted future, and the bounds
    above are the right ones. This needs one future for all quotes, so build one
    surface per expiry if the futures differ.

    Args:
        strike: Strikes, shape (n,).
        maturity: Times to expiry in years, shape (n,).
        price: Market prices, shape (n,).
        spot: Current level of the underlying (index spot, or VIX level for VIX options).
        option_type: 'call' / 'put' per quote, or one value for all.
        weight: Calibration weight per quote, or one value for all.
        r: Risk-free rate (continuous).
        q: Dividend yield (continuous).

    Raises:
        ValueError: If any check above fails; the message names the offending quotes.
    """
    strike: np.ndarray
    maturity: np.ndarray
    price: np.ndarray
    spot: float
    option_type: np.ndarray = field(default='call')  # type: ignore[assignment]
    weight: np.ndarray = field(default=1.0)  # type: ignore[assignment]
    r: float = 0.0
    q: float = 0.0

    def __post_init__(self) -> None:
        strike = np.atleast_1d(np.asarray(self.strike, dtype=float))
        n = strike.shape[0]
        if strike.ndim != 1 or n == 0:
            raise ValueError("strike must be a non-empty 1-D array.")

        def column(values: Any, dtype: type) -> np.ndarray:
            arr = np.asarray(values, dtype=dtype)
            if arr.ndim == 0:
                return np.full(n, arr.item(), dtype=dtype)
            if arr.shape != (n,):
                raise ValueError(f"Expected {n} values, got shape {arr.shape}.")
            return arr

        maturity = column(self.maturity, float)
        price = column(self.price, float)
        option_type = column(self.option_type, object).astype(str)
        weight = column(self.weight, float)

        spot, r, q = float(self.spot), float(self.r), float(self.q)
        for name, value in [('spot', spot), ('r', r), ('q', q)]:
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value}.")
        if not spot > 0:
            raise ValueError("spot must be positive.")

        for name, values in [('strike', strike), ('maturity', maturity), ('price', price),
                             ('weight', weight)]:
            _require(np.isfinite(values), f"{name} must be finite")
        _require(strike > 0, "Strikes must be positive")
        _require(maturity > 0, "Maturities must be positive")
        _require(price >= 0, "Prices must be non-negative")
        _require(np.isin(option_type, ['call', 'put']), "option_type must be 'call' or 'put'")
        _require(weight >= 0, "Weights must be non-negative")
        if not weight.sum() > 0:
            raise ValueError("Weights must have a positive sum.")

        first: dict[tuple, int] = {}
        duplicates = []
        for i, key in enumerate(zip(strike, maturity, option_type, strict=True)):
            if key in first:
                duplicates.append(f"{first[key]} and {i} ({key[2]}, strike={key[0]}, "
                                  f"maturity={key[1]})")
            else:
                first[key] = i
        if duplicates:
            raise ValueError("Duplicate (strike, maturity, option_type) quotes: "
                             + "; ".join(duplicates) + ".")

        fwd = spot * np.exp(-q * maturity)     # S e^{-qT}
        pv_k = strike * np.exp(-r * maturity)  # K e^{-rT}
        is_call = option_type == 'call'
        lower = np.maximum(np.where(is_call, fwd - pv_k, pv_k - fwd), 0.0)
        upper = np.where(is_call, fwd, pv_k)
        tol = _BOUND_RTOL * np.maximum(fwd, pv_k)
        _require(price <= upper + tol,
                 "Prices above the no-arbitrage upper bound (S e^{-qT} for calls, "
                 "K e^{-rT} for puts)")
        _require(price >= lower - tol,
                 "Prices below the no-arbitrage lower bound (discounted intrinsic value)")

        for name, value in [('strike', strike), ('maturity', maturity), ('price', price),
                            ('option_type', option_type), ('weight', weight)]:
            value.flags.writeable = False
            object.__setattr__(self, name, value)
        object.__setattr__(self, 'spot', spot)
        object.__setattr__(self, 'r', r)
        object.__setattr__(self, 'q', q)

    @classmethod
    def from_columns(cls, table: Mapping[str, Any], spot: float,
                     r: float = 0.0, q: float = 0.0) -> OptionSurface:
        """
        Build from anything indexable by column name: a dict of arrays or a DataFrame.

        Required columns: 'strike', 'maturity', 'price'.
        Optional: 'option_type' (default 'call'), 'weight' (default 1.0).
        """
        columns = set(table.keys())
        missing = {'strike', 'maturity', 'price'} - columns
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        return cls(
            strike=np.asarray(table['strike']),
            maturity=np.asarray(table['maturity']),
            price=np.asarray(table['price']),
            option_type=np.asarray(table['option_type']) if 'option_type' in columns else 'call',
            weight=np.asarray(table['weight']) if 'weight' in columns else 1.0,
            spot=spot, r=r, q=q,
        )

    def __len__(self) -> int:
        return len(self.strike)
