"""
Base Models for Parameters and Configuration
"""
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class MarketData(BaseModel):
    """Market data parameters"""
    model_config = ConfigDict(frozen=True)

    spot_price: float = Field(gt=0, description="Current price of the underlying asset")
    risk_free_rate: float = Field(description="Risk-free interest rate")
    dividend_yield: float = Field(default=0.0, description="Continuous dividend yield")
    timestamp: datetime = Field(default_factory=datetime.now)


class InstrumentParams(BaseModel):
    """Base instrument parameters"""
    model_config = ConfigDict(frozen=True)

    symbol: str = Field(description="Instrument identifier")
    expiry: datetime | None = Field(default=None, description="Expiration date if applicable")


class OptionParams(InstrumentParams):
    """Option-specific parameters"""
    strike_price: float = Field(gt=0, description="Strike price")
    option_type: str = Field(pattern='^(call|put)$', description="Option type")


class ModelConfig(BaseModel):
    """Base configuration for pricing models"""
    model_config = ConfigDict(frozen=True)

    model_type: str
    numerical_params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("numerical_params")
    @classmethod
    def validate_numerical_params(cls, value: dict) -> dict:
        return value


class Greeks(BaseModel):
    """Output structure for Greeks calculation"""
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


# Abstract Base Classes
class PricingModel(ABC):
    """Abstract base class for pricing models"""

    @abstractmethod
    def price(self, market_data: MarketData, instrument: InstrumentParams) -> float:
        """Calculate instrument price"""
        ...

    @abstractmethod
    def greeks(self, market_data: MarketData, instrument: InstrumentParams) -> dict[str, float]:
        """Calculate sensitivity measures (Greeks)"""
        ...


class DataSource(ABC):
    """Abstract base class for market data sources"""

    @abstractmethod
    def get_market_data(self, symbol: str, start_date: datetime, end_date: datetime) -> Any:
        """Fetch market data for given symbol and date range"""
        ...

    @abstractmethod
    def get_current_price(self, symbol: str) -> float:
        """Get current price for a given symbol"""
        ...
