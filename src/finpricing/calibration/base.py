"""
Abstract base class for model calibrators.
"""
from abc import ABC, abstractmethod

from finpricing.calibration.surface import OptionSurface


class Calibrator(ABC):
    """Base class for all calibrators."""

    @abstractmethod
    def calibrate(self, surface: OptionSurface, **kwargs) -> tuple:
        """
        Fit model parameters to observed option quotes.

        Args:
            surface: Market quotes (see ``OptionSurface``).

        Returns:
            Tuple of (fitted_params, auxiliary_info).
        """
        ...
