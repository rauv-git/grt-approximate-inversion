"""
Computing reconstruction kernels in the general case.
"""

from .data_generation import generate_data_general
import numpy as np 
from joblib import Parallel, delayed
import gc


def K_star_egamma(points, gamma, p, q, k):
    """
    Evaluate the K_star_egamma function over a meshgrid (X, Y) for a fixed point p.

    Parameters:
    - X, Y: 2D arrays from np.meshgrid
    - p: 2-element array or list [px, py]
    - q, k: integers
    - gamma: float, radius parameter

    Returns:
    - Z: 2D array of the same shape as X, Y with evaluated function values
    """
    
    if points.shape[0] > 2:
        X = points[:, 0]  # x1 coordinates
        Y = points[:, 1]  # x2 coordinates (depth)
    else:
        X = points[0, :]  # x1 coordinates
        Y = points[1, :]  # x2 coordinates (depth)
        
    dx = X - p[0]
    dy = Y - p[1]
    r2 = dx**2 + dy**2  # squared distance from each point to p

    inside = r2 < gamma**2  # boolean mask for points within the gamma ball
    x1_q = Y**q  # Y is treated as depth coordinate

    coeff = (k + 1) / (np.pi * gamma**(2 * (k + 1)))
    factor = 4 * k * (gamma**2 - r2)**(k - 2) * (k * r2 - gamma**2)

    Z = np.zeros_like(X)
    Z[inside] = coeff * x1_q[inside] * factor[inside]

    return Z


def process_single_p(i, p, gamma, q, k, t_data, dx, s_data_extended, s_data, c, alpha, tau_vals, a_vals, x_origin = 0):
    '''
    results = Parallel(n_jobs=-1, verbose=1)(
        delayed(generate_data_general)(
            t_data, dx, s_data_extended, s_data, c, alpha, tau_vals, a_vals, x_origin=x_origin, 
            func=lambda points: K_star_egamma(points, gamma, p, q, k)
        )
        for j in range(len(t_data))
    )

    # Collect results
    vals = np.zeros((len(s_data), len(t_data)))
    for m, vals_slice in enumerate(results):
        vals[:, m] = vals_slice
    return i, vals
    '''

    return i, generate_data_general(
            t_data, dx, s_data_extended, s_data, c, alpha, tau_vals, a_vals, x_origin=x_origin, 
            func=lambda points: K_star_egamma(points, gamma, p, q, k))

def compute_kernels_parallel_batched_general(p_ref_vals, gamma, q, k, t_data, dx, s_data_extended, s_data, 
                                     c, alpha, tau_vals, a_vals, x_origin, batch_size=5, n_jobs=-1):
    """
    Compute kernels in parallel batches to balance memory usage and speed.

     Parameters:
    - p_ref_vals : discrete grid points where we want to compute reconstruction kernels
    - gamma: float, radius parameter
    - q, k: integers
    - t_data : 1D array with time discretization of data setup
    - dx : tuple with spacing in physical space (x, y)
    - s_data_extended : 1D array with extended space discretization of data setup
    - s_data : 1D array with space discretization of data setup
    - c : 2D array representing the background velocity on computational domain
    - alpha : common offset parameter
    - tau_vals : array of all computed solution for eikonal equation for all sources in s_data_extended
     - a_vals : array of all computed solution for transport equation for all sources in s_data_extended
    - x_origin : first values in x
    - batch_size : integer
    - n_jobs : integer

    Returns:
    - kernels : array with all reconstruction kernels
    """
    
    # Split p_ref_vals indices into batches
    n_p = len(p_ref_vals)
    batches = [list(range(i, min(i + batch_size, n_p))) for i in range(0, n_p, batch_size)]
    
    print(f"Processing {n_p} kernels in {len(batches)} batches of size ~{batch_size}")
    
    # Initialize output array
    kernels = np.zeros((len(p_ref_vals), len(s_data), len(t_data)))
    
    # Process each batch in parallel
    for batch_idx, batch in enumerate(batches):
        print(f"Processing batch {batch_idx + 1}/{len(batches)} with {len(batch)} kernels")
        
        # Parallel processing within each batch
        batch_results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(process_single_p)(
                #j, (s_mid, p_ref_vals[j]), gamma, q, k, t_vals, phi_vals, dx, s_ref, I_M, x_origin=x_origin
                j, p_ref_vals[j], gamma, q, k, t_data, dx, s_data_extended, s_data, c, alpha, tau_vals, a_vals, x_origin = x_origin
            )
            for j in batch
        )
        
        # Store results
        for original_j, vals in batch_results:
            kernels[original_j, :, :] = vals
        
        # Force garbage collection after each batch
        gc.collect()
    
    return kernels



