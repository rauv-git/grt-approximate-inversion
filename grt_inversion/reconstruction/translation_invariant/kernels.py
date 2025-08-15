"""
Compute reconstruction kernels.
"""

from skimage.measure import find_contours
import numpy as np
import warnings
import gc
import math

from .integration import quadrature, interp_index

from joblib import Parallel, delayed

warnings.filterwarnings('ignore')
def process_single_p(i, p, gamma, q, k, t_data, phi_vals, dx, s_data, I_M, x_origin = 0, beta = 0, comments=False):
    def K_star_egamma(points, gamma, p, q, k, beta = 0):
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
        x1_q = Y**q + beta # Y is treated as depth coordinate

        coeff = (k + 1) / (np.pi * gamma**(2 * (k + 1)))
        factor = 4 * k * (gamma**2 - r2)**(k - 2) * (k * r2 - gamma**2)

        Z = np.zeros_like(X)
        Z[inside] = coeff * x1_q[inside] * factor[inside]

        return Z


    def e_gamma(points, gamma, p, k):
        """
        Evaluate the e_gamma function over a meshgrid (X, Y) for a fixed point p.

        Parameters:
        - X, Y: 2D arrays from np.meshgrid
        - p: 2-element array or list [px, py]
        - k: integers
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

        coeff = (k + 1) / (np.pi * gamma**(2 * (k + 1)))
        factor = (gamma**2 - r2)**(k)

        Z = np.zeros_like(X)
        Z[inside] = coeff * factor[inside]

        return Z
       

    def process_single_t(j, t, phi_vals, dx, s_list, integrand_factor, x_origin=None, beta = 0):
        res = [0] * len(s_list)
        contours = find_contours(phi_vals, t)
        
        if not contours:
            t_min, t_max = phi_vals.min(), phi_vals.max()
            #raise Exception("No contours found for t={:.2f}. Range is [{:.2f}, {:.2f}]".format(t, t_min, t_max))
            warnings.warn("No contours found for t={:.2f}. Range is [{:.2f}, {:.2f}]".format(t, t_min, t_max), Warning)
        elif len(contours) > 2:
            warnings.warn("more than two contours found for t={:.2f}".format(t), Warning)

        
        # If x_origin is not provided, assume grid starts at x[0]
        if x_origin is None:
            x_origin = 0  # or whatever your actual x[0] is
        
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
                
                # With pseudodifferential operator K
                n = K_star_egamma(real_points, gamma, p, q, k, beta)
                # Without pseudodifferential operator K
                # n = e_gamma(real_points, gamma, p, k)

                n_ind = n.nonzero()
                
                # For interpolation, use grid coordinates
                grid_points_nonzero = grid_points[n_ind]
                
                integrand = np.zeros(len(grid_points))
                integrand[n_ind] = interp_index(grid_points_nonzero, integrand_factor) * n[n_ind]
                
                # Use real_points for quadrature (physical distances)
                res[i] += (1 / np.sqrt(2)) * quadrature(real_points, integrand)
                
        
        return j, res


    results = Parallel(n_jobs=-1, verbose=0)(
        delayed(process_single_t)(
            j, t_data[j], phi_vals, dx, s_data, I_M, x_origin = x_origin, beta = beta
        )
        for j in range(len(t_data))
    )

    # Collect results
    vals = np.zeros((len(s_data), len(t_data)))
    for m, vals_slice in results:
        vals[:, m] = vals_slice

    return i, vals



def compute_kernels_parallel_batched(p_ref_vals, s_mid, gamma, q, k, t_vals, phi_vals, dx, 
                                     s_ref, I_M, x_origin, beta = 0, batch_size=5, n_jobs=-1):
    """
    Compute kernels in parallel batches to balance memory usage and speed.

     Parameters:
    - p_ref_vals : discrete grid points where we want to compute reconstruction kernels
    - s_mid : midpoint
    - gamma: float, radius parameter
    - q, k: integers
    - t_vals : 1D array with time discretization
    - phi_vals : travel times
    - dx : tuple with spacing in physical space (x, y)
    - s_ref : s values
    - I_M: integration factor
    - x_origin : first values in x
    - beta : psuedodifferential operator parameter
    - batch_size : integer
    - n_jobs : integer

    Returns:
    - kernels: array with all reconstruction kernels
    """
    
    # Split p2 indices into batches
    n_p2 = len(p_ref_vals)
    batches = [list(range(i, min(i + batch_size, n_p2))) for i in range(0, n_p2, batch_size)]
    
    print(f"Processing {n_p2} kernels in {len(batches)} batches of size ~{batch_size}")
    
    # Initialize output array
    kernels = np.zeros((len(p_ref_vals), len(s_ref), len(t_vals)))
    
    # Process each batch in parallel
    for batch_idx, batch in enumerate(batches):
        print(f"Processing batch {batch_idx + 1}/{len(batches)} with {len(batch)} kernels")
        
        # Parallel processing within each batch
        batch_results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(process_single_p)(
                j, (s_mid, p_ref_vals[j]), gamma, q, k, t_vals, phi_vals, dx, s_ref, I_M, x_origin=x_origin, beta = beta
            )
            for j in batch
        )
        
        # Store results
        for original_j, vals in batch_results:
            kernels[original_j, :, :] = vals
        
        # Force garbage collection after each batch
        gc.collect()
        
        # Optional: print memory usage
        try:
            import psutil
            memory_percent = psutil.virtual_memory().percent
            print(f"Memory usage after batch {batch_idx + 1}: {memory_percent:.1f}%")
        except ImportError:
            pass
    
    return kernels


# Example usage
# kernels = compute_kernels_parallel_batched(p_ref_vals, s_mid, gamma, q, k, t_vals, phi_vals, dx, s_ref, I_M, x_origin=x[0], batch_size=15)

def compute_ref_kernels(p1_vals, p2_vals, p_ref_vals, s_vals, t_vals, 
                        s_mid, gamma, q, k, phi_vals, dx, s_ref, I_M, x_origin, beta=0):
    
    """
    Compute kernels.

     Parameters:
    - p1_vals, p2_vals, p_ref_vals : discrete grid points where we want to compute reconstruction kernels
    - s_vals, t_vals : arrays with space and time discretization
    - s_mid : midpoint
    - gamma: float, radius parameter
    - q, k: integers
    - phi_vals : travel times
    - dx : tuple with spacing in physical space (x, y)
    - s_ref : s values
    - I_M: integration factor
    - x_origin : first values in x
    - beta : psuedodifferential operator parameter

    Returns:
    - kernels: array with all reconstruction kernels
    """
    
    kernels = compute_kernels_parallel_batched(p_ref_vals, s_mid, gamma, q, k, 
                                               t_vals, phi_vals, dx, s_ref, I_M, 
                                               x_origin, beta, batch_size=15
                                               )

    # Interpolate kernel to p2_vals linearly if needed
    kernel_interp = np.zeros((len(p2_vals), len(s_ref), len(t_vals)))

    if np.any(p_ref_vals != p2_vals):
        for j, p2 in enumerate(p2_vals):
            pl_idx = np.argmin(abs(p2 - p_ref_vals))
            pl = p_ref_vals[pl_idx]

            pr_idx = pl_idx + 1
            if pr_idx >= len(p_ref_vals) - 1:
                if p2 > p_ref_vals[-1]:
                    print("p2 exceeds reference values")
                if p2 == p_ref_vals[-1]:
                    kernel_interp[j, :, :] = kernels[-1, :, :]
            else:
                pr = p_ref_vals[pr_idx]

                kernel_interp[j, :, :] = (pr - p2)/(pr - pl) * kernels[pl_idx, :, :] + (p2 - pl)/(pr - pl) * kernels[pr_idx, :, :]
        
        print("kernel interpolated to p2")

    else:
        kernel_interp = kernels

        print("no interpolation needed in p")

    # Slice kernels so that they are available on M_p
    ref_kernels = np.zeros((len(p1_vals), len(p2_vals), len(s_vals), len(t_vals)))

    # i_mid = (ns - 1)//2
    i_mid = (kernel_interp.shape[1] - 1)//2 # TODO: is that what we want to do?

    ds = abs(s_vals[1] - s_vals[0])
    ns = len(s_vals)

    for i, p1 in enumerate(p1_vals):
        i_p1 = math.floor((p1 - s_vals[0])/ds)
        ref_kernels[i, :, :, :] = kernel_interp[:, int(i_mid - i_p1):int(i_mid - i_p1 + ns), :]

    return ref_kernels
        

