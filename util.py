import numpy as np
from typing import Tuple


def make_XOR(N_pts: int, noise: float = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset for the XOR problem with a specified number of points and optional gaussian noise.

    Make a set of four gaussian distributions centered at points (0,0), (3,3), (0,3) and (3,0). They are labeled to generate the non-linear classification XOR problem 

    Parameters:
    - N_pts (int): Number of data points to generate. Should be a multiple of 4 for balanced classes.
    - noise (float, optional): Standard deviation of the Gaussian noise added to the data points. Default is 0.

    Returns:
    - Tuple[np.ndarray, np.ndarray]: A tuple containing two arrays:
        - X (np.ndarray): 2D array of shape (N_pts, 2) representing the input features.
        - Y (np.ndarray): 1D array of shape (N_pts,) representing the corresponding binary labels (0 or 1).

    Example:
    >>> X, Y = make_XOR(1000, noise=0.3)
    """

    # Generate the four distributions 
    quadrant1  = np.random.randn(int(N_pts / 4), 2) * noise + [0.0]
    quadrant2  = np.random.randn(int(N_pts / 4), 2) * noise + [3, 3]
    quadrant3  = np.random.randn(int(N_pts / 4), 2) * noise + [0, 3]
    quadrant4  = np.random.randn(int(N_pts / 4), 2) * noise + [3, 0]

    # Concatenate points from all quadrants to form the final dataset
    X = np.concatenate([quadrant1, quadrant2, quadrant3, quadrant4])

    # Generate the binary labels corresponding to the XOR problem
    Y = np.concatenate([np.ones(int(N_pts / 2)).T, np.zeros(int(N_pts / 2)).T])
    
    return X, Y
