"""
    Utilities for Fourier Pricing
"""
import numpy as np


def generate_grid(start: float, step: float, num: int) -> np.ndarray:
    """Generate a grid of equally spaced values."""
    return np.linspace(start, start + step * (num - 1), num)


def compute_weights(num: int, method: str = "trapezoidal") -> np.ndarray:
    """Compute quadrature weights."""
    if method == "trapezoidal":
        w = np.ones(num)
        w[0] = w[-1] = 0.5
    elif method == "simpson":
        w = np.array([3 - (-1)**j for j in range(num)])
        w[0] -= 1
        w = w / 3
    else:
        raise ValueError(f"Unsupported weight computation method: {method}")
    return w



def generate_correlated_diffusions(num_paths: int, ticks: np.ndarray, rho: float):
    """
    Simulate increments of correlated Brownian diffusions.

    Args:
        num_paths (int): Number of paths.
        ticks (np.ndarray): Array of floats representing the time steps t_0 < ... < t_{n-1}.
        rho (float): Correlation coefficient.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Two matrices representing correlated Brownian increments
                                       with dimensions (num_paths, len(ticks)-1).
    """

    # Validate inputs
    if not -1 <= rho <= 1:
        raise ValueError(f"Correlation coefficient rho={rho} must be between -1 and 1.")
    if len(ticks) < 2:
        raise ValueError("The ticks array must contain at least two time steps.")
    if np.any(np.diff(ticks) <= 0):
        raise ValueError("Ticks must be strictly increasing.")

    # Simulate independent Brownian increments
    time_diffs = np.sqrt(np.diff(ticks))
    independent_increments_1 = np.random.randn(num_paths, len(ticks) - 1) * time_diffs
    independent_increments_2 = np.random.randn(num_paths, len(ticks) - 1) * time_diffs

    # Create correlated increments
    correlated_increments_2 = rho * independent_increments_1 + np.sqrt(1 - rho ** 2) * independent_increments_2

    return independent_increments_1, correlated_increments_2
