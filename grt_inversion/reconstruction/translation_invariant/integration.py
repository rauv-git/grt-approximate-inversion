"""
Integration utilities for contour integrals.
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

def interp_index(points, array):
    """
    (bi-)linearly interpolates float indices 'points' of a 2D array
    """
    assert array.ndim == points.shape[1], "ndim of 'arr' and 2nd shape of 'points' does not match!"
    shape = array.shape

    # calculate lower bound indices
    rounded = points.astype(int)
    x, y = rounded[:, 0], rounded[:, 1]
    # calculate the weights for the four raster values
    factor = points - rounded
    factor_x, factor_y = factor[:, 0], factor[:, 1]
    # calculate neighboring indices and ensure they are not outside of arr.shape
    x_right = np.minimum(x + 1, shape[0] - 1)
    y_right = np.minimum(y + 1, shape[1] - 1)

    # interpolate in x-direction
    arr_interp_x1 = (1 - factor_x) * array[x, y] + factor_x * array[x_right, y]
    arr_interp_x2 = (1 - factor_x) * array[x, y_right] + factor_x * array[x_right, y_right]

    # interpolate the results in y-direction
    return (1 - factor_y) * arr_interp_x1 + factor_y * arr_interp_x2


def quadrature(points, values):
    """
    Trapezoidal rule integration.
    """
    assert len(points) == len(values)

    # distances between each point
    dist = np.linalg.norm(points[1:] - points[:-1], axis=1, ord=2)
    # sum of two adjacent values
    value_sum = values[:-1] + values[1:]
    return np.sum(value_sum * dist) / 2

