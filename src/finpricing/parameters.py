"""
Parameter definitions for characteristic functions
"""
from pydantic import BaseModel, Field, ConfigDict


class BasicModelParameters(BaseModel):
    """Base model parameters shared across different models"""
    model_config = ConfigDict(frozen=True)

    model_type: str = Field(..., description="Type of the model (e.g., 'Black-Scholes', 'Merton', 'Kou')")


class VIXBlackScholesModelParameters(BasicModelParameters):
    """Parameters for the Black-Scholes model"""
    model_type: str = 'Black-Scholes'
    sigma: float = Field(0.3275, gt=0, description="Volatility of the Black-Scholes model")


class VIXMertonModelParameters(BasicModelParameters):
    """Parameters for the Merton model"""
    model_type: str = 'Merton'
    rho: float = Field(-0.45, ge=-1, le=1, description="Correlation coefficient")
    omega: float = Field(2.04, gt=0, description="Volatility of the jumps")
    lmbd: float = Field(3.52, gt=0, description="Jump intensity")
    k1: float = Field(21.9, gt=0, description="Rate parameter for jump size mean")
    k2: float = Field(2.07, gt=0, description="Rate parameter for jump size variance")
    m: float = Field(0.54, description="Mean of jump size distribution")
    delta: float = Field(0.25, gt=0, description="Standard deviation of jump size distribution")


class VIXKouModelParameters(BasicModelParameters):
    """Parameters for the Kou model"""
    model_type: str = 'Kou'
    rho: float = Field(-0.45, ge=-1, le=1, description="Correlation coefficient")
    omega: float = Field(1.98, gt=0, description="Volatility of the jumps")
    lmbd: float = Field(13.6, gt=0, description="Jump intensity")
    k1: float = Field(22.3, gt=0, description="Rate parameter for jump size mean")
    k2: float = Field(2.20, gt=0, description="Rate parameter for jump size variance")
    p: float = Field(0.86, ge=0, le=1, description="Probability of upward jump")
    alpha_plus: float = Field(4.25, gt=0, description="Rate of upward jumps")
    alpha_minus: float = Field(19.9, gt=0, description="Rate of downward jumps")


class HestonParameters(BasicModelParameters):
    """
    Heston (1993) stochastic volatility parameters.

        dS_t = (r - q) S_t dt + sqrt(v_t) S_t dW_t
        dv_t = kappa (theta - v_t) dt + xi sqrt(v_t) dZ_t,   d<W, Z>_t = rho dt
    """
    model_type: str = 'Heston'
    v0: float = Field(0.04, gt=0, description="Initial variance")
    kappa: float = Field(1.5, gt=0, description="Mean-reversion speed of the variance")
    theta: float = Field(0.04, gt=0, description="Long-run variance")
    xi: float = Field(0.5, gt=0, description="Volatility of variance (vol-of-vol)")
    rho: float = Field(-0.7, ge=-1, le=1, description="Spot/variance correlation")

    @property
    def feller_satisfied(self) -> bool:
        """2 kappa theta >= xi^2: the variance process stays strictly positive."""
        return 2 * self.kappa * self.theta >= self.xi ** 2
