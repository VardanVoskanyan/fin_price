"""
Option surface: the market quotes a calibrator fits to.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class OptionSurface:
    """
    A set of European option quotes on one underlying, stored as numpy arrays.

    Args:
        strike: Strikes, shape (n,).
        maturity: Times to expiry in years, shape (n,).
        price: Market prices, shape (n,).
        spot: Current level of the underlying (index spot, or VIX level for VIX options).
        option_type: 'call' / 'put' per quote, or one value for all.
        weight: Calibration weight per quote, or one value for all.
        r: Risk-free rate (continuous).
        q: Dividend yield (continuous).
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

        if np.any(strike <= 0) or np.any(maturity <= 0):
            raise ValueError("Strikes and maturities must be positive.")
        if np.any(price < 0) or not np.all(np.isfinite(price)):
            raise ValueError("Prices must be finite and non-negative.")
        if not np.all(np.isin(option_type, ['call', 'put'])):
            raise ValueError("option_type must be 'call' or 'put'.")
        if np.any(weight < 0):
            raise ValueError("Weights must be non-negative.")
        if not self.spot > 0:
            raise ValueError("spot must be positive.")

        for name, value in [('strike', strike), ('maturity', maturity), ('price', price),
                            ('option_type', option_type), ('weight', weight)]:
            value.flags.writeable = False
            object.__setattr__(self, name, value)
        object.__setattr__(self, 'spot', float(self.spot))
        object.__setattr__(self, 'r', float(self.r))
        object.__setattr__(self, 'q', float(self.q))

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
