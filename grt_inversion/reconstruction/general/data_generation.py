"""
For general setting, generate data g = Fn.
"""

from skimage.measure import find_contours
import numpy as np
import warnings
from joblib import Parallel, delayed


from ..translation_invariant.integration import quadrature, interp_index
from ..translation_invariant.data_generation import n_func


def process_single_s(k, t_list, dx, s, s_vals, c, alpha, tau_vals, a_vals, x_origin=None, func=n_func):
    # Determine source and receiver location
    source = np.array([np.argmin(np.abs(s_vals - (s - alpha))), 0])
    receiver = np.array([np.argmin(np.abs(s_vals - (s + alpha))), 0])
    #print(source, receiver, tau_vals.shape)

    # Access tau and a accordingly
    tau_s = tau_vals[source[0], :, :]
    tau_r = tau_vals[receiver[0], :, :]

    phi_vals = tau_s + tau_r
    
    a_s = a_vals[source[0], :, :] #solve_transport_equation(c, tau_s, X, Y, x_s, dx, np.zeros_like(X))
    a_r = a_vals[receiver[0], :, :] #solve_transport_equation(c, tau_r, X, Y, x_r, dx, np.zeros_like(X))
    
    grad_tau_s = np.gradient(tau_s, *dx, edge_order=2) 
    grad_tau_r = np.gradient(tau_r, *dx, edge_order=2)
    dot_product = (grad_tau_s[0] * grad_tau_r[0] + grad_tau_s[1] * grad_tau_r[1])

    denom = 1 + c**2 * dot_product
    
    # Check whether denominantor is positive (need to take square root)
    if np.any(denom < 0):
        warnings.warn("Negative denominator: forward scattering or numerical instability.", Warning)
        denom = np.maximum(denom, 1e-12)  # Prevent division by zero
        integrand_factor = (a_s * a_r) / (c * np.sqrt(denom))
        
        # Filter out problematic contour points
        valid_mask = (denom > 1e-10) & (np.abs(integrand_factor) < 1e6)
        integrand_factor[~valid_mask] = 0
    else:
        integrand_factor = (a_s * a_r) / (c * np.sqrt(denom))


    res = [0] * len(t_list)
    
    for i, t in enumerate(t_list):
        contours = find_contours(phi_vals, t)

        if not contours:
            t_min, t_max = phi_vals.min(), phi_vals.max()
            warnings.warn("No contours found for t={:.3f}.".format(t, t_min, t_max))
            continue

        elif len(contours) > 2:
            warnings.warn("More than two contours found for t={:.3f}.".format(t), Warning)

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
            real_points = grid_points * np.array(dx) + np.array([x_origin, 0])
            
            n = func(real_points)
            n_ind = n.nonzero()
                
            # For interpolation, use grid coordinates
            grid_points_nonzero = grid_points[n_ind]
                
            integrand = np.zeros(len(grid_points))
            integrand[n_ind] = interp_index(grid_points_nonzero, integrand_factor) * n[n_ind]
                
            # Use real_points for quadrature (physical distances)
            res[i] += (1 / np.sqrt(2)) * quadrature(real_points, integrand)
        
    return k, res


def generate_data_general(t_data, dx, s_data_extended, s_data, c, alpha, tau_vals, a_vals, x_origin=None, func=n_func):
    """
    Generate data g = Fn on data grid.

    Parameters:
    - t_data : 1D array with time discretization of data setup
    - dx : tuple with spacing in physical space (x, y)
    - s_data_extended : 1D array with extended space discretization of data setup
    - s_data : 1D array with space discretization of data setup
    - c : 2D array representing the background velocity on computational domain
    - alpha : common offset parameter
    - tau_vals : array of all computed solution for eikonal equation for all sources in s_data_extended
    - a_vals : array of all computed solution for transport equation for all sources in s_data_extended
    - x_origin : first values in x
    - func : function for which we want to generate the GRT data

    Returns:
    - g : 2D array with the generated data
    """
    n_jobs = -1

    # Parallel processing: one job per time value, parallelizing in s
    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(process_single_s)(
            k, t_data, dx, s_data[k], s_data_extended, c, alpha, tau_vals, a_vals, x_origin=x_origin, func=func
        )
        for k in range(len(s_data))
    )

    # Collect results
    g = np.zeros((len(s_data), len(t_data)))
    for m, vals_slice in results:
        g[m, :] = vals_slice

    return g




