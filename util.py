import numpy as np
from typing import Tuple


def make_XOR(N_pts: int, noise: float = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset for the XOR problem with a specified number of points and optional noise.

    Make a set of four gaussian distributions centered at points (0,0), (3,3), (0,3) and (3,0). They are labeled to generate the non-linear classification XOR problem 
    """

    X1 = np.random.randn(int(N_pts/4), 2) * noise + [0.0]
    X2 = np.random.randn(int(N_pts/4), 2) * noise + [3, 3]
    X3 = np.random.randn(int(N_pts/4), 2) * noise + [0, 3]
    X4 = np.random.randn(int(N_pts/4), 2) * noise + [3, 0]

    X = np.r_[X1, X2, X3, X4]
    Y = np.r_[np.ones(int(N_pts/2)).T, np.zeros(int(N_pts/2)).T]
    return X, Y
