import numpy as np
from numpy.typing import NDArray

def simul_total_jumps_merton(
    num_paths: int,
    t1: float,
    t2: float,
    lmbd: float,
    k2: float,
    m: float,
    delta: float,
    seed: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...

def simul_total_jumps_kou(
    num_paths: int,
    t1: float,
    t2: float,
    lmbd: float,
    k2: float,
    p: float,
    alpha_plus: float,
    alpha_minus: float,
    seed: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
