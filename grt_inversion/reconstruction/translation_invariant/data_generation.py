"""
Generate data g = Fn here.
"""

from skimage.measure import find_contours
import numpy as np
import warnings

from .integration import quadrature, interp_index

def n_func(points):
    """Test function. Concatenation of characteristic functions."""
    # Extract coordinates
    if points.shape[0] > 2:
        X = points[:, 0]  # x1 coordinates
        Y = points[:, 1]  # x2 coordinates (depth)
    else:
        X = points[0, :]  # x1 coordinates
        Y = points[1, :]  # x2 coordinates (depth)
    
    # Initialize n
    n = np.zeros(len(X))
    
    # Term 1: Large circle
    center1 = (0, 5)
    radius1 = 2.0
    dist1 = np.sqrt((X - center1[0])**2 + (Y - center1[1])**2)
    chi_B2_1 = (dist1 <= radius1).astype(float)
    
    # Term 2: Small circle (same center)
    center2 = (0, 5)
    radius2 = 1.0
    dist2 = np.sqrt((X - center2[0])**2 + (Y - center2[1])**2)
    chi_B2_2 = (dist2 <= radius2).astype(float)
    
    # Term 3: Square region
    center3 = (3, 6)
    radius3 = 1.25
    chi_Binf = ((np.abs(X - center3[0]) <= radius3) & 
                (np.abs(Y - center3[1]) <= radius3)).astype(float)
    
    # Term 4: Sinusoidal boundary
    boundary_curve = 6.5 + np.sin(np.pi * X / 2)
    chi_sine_region = (Y >= boundary_curve).astype(float)
    
    # Combine all terms 
    n = chi_B2_1 - chi_B2_2 + chi_Binf + chi_sine_region
    
    return n


def process_single_t(k, t, phi_vals, dx, s_list, integrand_factor, x_origin=None, func=n_func, comments=False):
    res = [0] * len(s_list)
    contours = find_contours(phi_vals, t)
    
    if not contours:
        t_min, t_max = phi_vals.min(), phi_vals.max()
        #raise Exception("No contours found for t={:.2f}. Range is [{:.2f}, {:.2f}]".format(t, t_min, t_max))
        warnings.warn("No contours found for t={:.2f}. Range is [{:.2f}, {:.2f}]".format(t, t_min, t_max), Warning)
        return k, res
    elif len(contours) > 2:
        warnings.warn("more than two contours found for t={:.2f}".format(t), Warning)

    # If x_origin is not provided, assume grid starts at x[0] = 0
    if x_origin is None:
        x_origin = 0 
    
    for points in contours:
        # Convert grid coordinates to real coordinates
        if points[0, 0] > points[-1, 0]:
            grid_points = points
        else:
            grid_points = points[::-1]
            
        # Convert to real coordinates with proper origin
        real_points_base = grid_points * np.array(dx) + np.array([x_origin, 0])
        #print(real_points_base)
        for i, s in enumerate(s_list):
            # Apply s offset in real coordinates
            real_points = real_points_base + (s, 0)
            
            n = func(real_points)
            n_ind = n.nonzero()
            
            # For interpolation, use grid coordinates
            grid_points_nonzero = grid_points[n_ind]
            
            integrand = np.zeros(len(grid_points))
            integrand[n_ind] = interp_index(grid_points_nonzero, integrand_factor) * n[n_ind]
            
            # Use real_points for quadrature (physical distances)
            res[i] += (1 / np.sqrt(2)) * quadrature(real_points, integrand)
    
    return k, res


from joblib import Parallel, delayed


def generate_data(s_data, t_data, phi_vals, dx, I_M, x_origin, func):
    """
    Generate data g = Fn on data grid.

    Parameters:
    - s_data : 1D array with space discretization of data setup
    - t_data : 1D array with time discretization of data setup
    - phi_vals : travel times
    - dx : tuple with spacing in physical space (x, y)
    - I_M : integration factor
    - x_origin : first values in x
    - func : function for which we want to generate the GRT data

    Returns:
    - g : 2D array with the generated data
    """
    
    n_jobs = -1

    # Parallel processing: one job per time value
    results = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(process_single_t)(
        #delayed(process_single_t_improved)(
            k, t_data[k], phi_vals, dx, s_data, I_M, x_origin = x_origin, func = func
        )
        for k in range(len(t_data))
    )

    # Collect results
    g = np.zeros((len(s_data), len(t_data)))
    for m, vals_slice in results:
        g[:, m] = vals_slice

    return g
