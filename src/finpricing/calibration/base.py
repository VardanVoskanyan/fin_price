"""
Abstract base class for model calibrators.
"""
from abc import ABC, abstractmethod
from typing import Any

from finpricing.calibration.surface import OptionSurface


class Calibrator(ABC):
    """Base class for all calibrators."""

    @abstractmethod
    def calibrate(self, surface: OptionSurface) -> tuple[Any, dict[str, Any]]:
        """
        Fit model parameters to observed option quotes.

        Args:
            surface: Market quotes (see ``OptionSurface``).

        Returns:
            ``(params, info)``: the fitted parameters and a report on the fit. Every
            calibrator's ``info`` has at least these keys:

            - ``success`` (bool): True only if the fit can be trusted: the optimiser
              converged and no check below failed.
            - ``message`` (str): why ``success`` is False, or a short confirmation.
            - ``loss`` (float): objective value at the fitted parameters (finite).
            - ``n_quotes`` (int): number of quotes in the surface.
            - ``at_bounds`` (list[str]): names of parameters that ended on (within a
              small tolerance of) a bound of the search box. A non-empty list means
              the optimum may lie outside the box, so ``success`` is False.

            Subclasses may add model-specific keys.

        Raises:
            RuntimeError: If no finite loss could be computed at the fitted
                parameters (e.g. the pricer failed everywhere). A calibrator never
                returns parameters it could not evaluate.
        """
        ...
