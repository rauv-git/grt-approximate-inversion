"""
Help functions to interpolate kernels quickly.
"""

from joblib import Parallel, delayed
from scipy.interpolate import RectBivariateSpline
import numpy as np
import gc

def interpolate_kernel_batch_vectorized(batch_indices, ref_kernels, s_vals, t_vals, S_data, T_data):
    """
    Process multiple kernels in a single function call for better efficiency.
    """
    batch_results = []
    
    output_shape = S_data.shape
    
    for x_idx, y_idx in batch_indices:
        try:
            # Use lower-order splines for speed (kx=1, ky=1 is bilinear)
            kernel = RectBivariateSpline(s_vals, t_vals, ref_kernels[x_idx, y_idx, :, :], kx=1, ky=1)
            
            # Vectorized evaluation (faster than ev)
            interpolated = kernel(S_data[:, 0], T_data[0, :])
            batch_results.append((x_idx, y_idx, interpolated))
            
        except Exception as e:
            print(f"Error interpolating kernel [{x_idx}, {y_idx}]: {e}")
            batch_results.append((x_idx, y_idx, np.zeros(output_shape)))
    
    return batch_results

def interpolate_kernels_optimized(ref_kernels, s_vals, t_vals, S_data, T_data, 
                                 batch_size=30, n_jobs=-1, optimize_memory=True):
    """
    Optimized interpolation with larger batches and better memory management.
    """
    kernel_data = np.zeros((ref_kernels.shape[0], ref_kernels.shape[1], 
                           S_data.shape[0], S_data.shape[1]))
    
    # Check if interpolation is actually needed
    if (len(s_vals) == S_data.shape[0] and len(t_vals) == S_data.shape[1] and
        np.allclose(s_vals, S_data[:, 0], rtol=1e-10) and 
        np.allclose(t_vals, T_data[0, :], rtol=1e-10)):
        print("No interpolation needed - grids are identical")
        return ref_kernels
    
    # Create list of all (x_idx, y_idx) pairs
    all_indices = [(x_idx, y_idx) 
                   for x_idx in range(kernel_data.shape[0])
                   for y_idx in range(kernel_data.shape[1])]
    
    total_kernels = len(all_indices)
    
    # Use larger batches for better efficiency
    batches = [all_indices[i:i + batch_size] for i in range(0, len(all_indices), batch_size)]
    print(f"Processing in {len(batches)} batches of size ~{batch_size}")
    
    # Process batches in parallel with optimized settings
    for batch_idx, batch in enumerate(batches):
        if batch_idx % max(1, len(batches)//10) == 0:  # Print progress every 10%
            print(f"Processing batch {batch_idx + 1}/{len(batches)} ({100*batch_idx/len(batches):.1f}%)")
        
        try:
            # Use optimized batch processing
            batch_results = Parallel(n_jobs=n_jobs, verbose=0, backend='threading',
                                   batch_size=1, pre_dispatch='n_jobs')(
                delayed(interpolate_kernel_batch_vectorized)(
                    [indices], ref_kernels, s_vals, t_vals, S_data, T_data
                )
                for indices in batch
            )
            
            # Flatten batch results and store
            for batch_result in batch_results:
                for x_idx, y_idx, interpolated in batch_result:
                    kernel_data[x_idx, y_idx, :, :] = interpolated
                    
        except Exception as e:
            print(f"Parallel batch {batch_idx + 1} failed: {e}")
            print("Falling back to sequential processing...")
            
            # Sequential fallback
            for x_idx, y_idx in batch:
                try:
                    kernel = RectBivariateSpline(s_vals, t_vals,
                                               ref_kernels[x_idx, y_idx, :, :], 
                                               kx=1, ky=1)
                    kernel_data[x_idx, y_idx, :, :] = kernel(S_data[:, 0], T_data[0, :])
                except Exception as e2:
                    print(f"Sequential fallback failed for [{x_idx}, {y_idx}]: {e2}")
                    kernel_data[x_idx, y_idx, :, :] = np.zeros_like(S_data)
        
        # Memory management
        if optimize_memory and batch_idx % 5 == 0: # Every 5 batches
            gc.collect()
    
    print(f"Completed optimized interpolation of {total_kernels} kernels")
    return kernel_data

def interpolate_single_kernel_fast(x_idx, y_idx, ref_kernels, s_vals, t_vals, s_target, t_target):
    """
    Faster single kernel interpolation using direct coordinate arrays.
    """
    try:
        kernel = RectBivariateSpline(s_vals, t_vals, ref_kernels[x_idx, y_idx, :, :], kx=1, ky=1)
        # Use direct coordinate evaluation
        interpolated = kernel(s_target, t_target)
        return x_idx, y_idx, interpolated
    except Exception as e:
        return x_idx, y_idx, np.zeros((len(s_target), len(t_target)))

def interpolate_kernels_super_fast(ref_kernels, s_vals, t_vals, s_data, t_data, n_jobs=-1):
    """
    Maximum speed version - trades some memory safety for speed.
    """
    kernel_data = np.zeros((ref_kernels.shape[0], ref_kernels.shape[1], 
                           len(s_data), len(t_data)))
    
    # Quick identity check
    if (np.array_equal(s_vals, s_data) and np.array_equal(t_vals, t_data)):
        print("No interpolation needed - arrays are identical")
        return ref_kernels
    
    # Create all index pairs
    indices = [(x_idx, y_idx) 
               for x_idx in range(kernel_data.shape[0])
               for y_idx in range(kernel_data.shape[1])]
    
    # Use threading backend for I/O bound operations like interpolation
    results = Parallel(n_jobs=n_jobs, verbose=1, backend='threading', 
                      batch_size='auto', pre_dispatch='2*n_jobs')(
        delayed(interpolate_single_kernel_fast)(
            x_idx, y_idx, ref_kernels, s_vals, t_vals, s_data, t_data
        )
        for x_idx, y_idx in indices
    )
    
    # Store results
    for x_idx, y_idx, interpolated in results:
        kernel_data[x_idx, y_idx, :, :] = interpolated
    
    return kernel_data

def choose_interpolation_method(ref_kernels, s_vals, t_vals, s_data, t_data, n_jobs=-1, prefer_speed=True):
    """
    Automatically choose best interpolation method based on problem size.
    """
    total_kernels = ref_kernels.shape[0] * ref_kernels.shape[1]
    grid_size = len(s_data) * len(t_data)
    
    print(f"Problem size: {total_kernels} kernels, {grid_size} grid points each")
    
    # Rough estimate of memory usage
    estimated_memory_gb = (total_kernels * grid_size * 8) / (1024**3)  # 8 bytes per float64
    
    if estimated_memory_gb < 2.0 and prefer_speed:
        return interpolate_kernels_super_fast(ref_kernels, s_vals, t_vals, s_data, t_data, n_jobs)
    elif estimated_memory_gb < 8.0:
        S_data, T_data = np.meshgrid(s_data, t_data, indexing='ij')
        return interpolate_kernels_optimized(ref_kernels, s_vals, t_vals, S_data, T_data, 
                                           batch_size=50, n_jobs=n_jobs)
    else:
        S_data, T_data = np.meshgrid(s_data, t_data, indexing='ij')
        return interpolate_kernels_super_fast(ref_kernels, s_vals, t_vals, s_data, t_data, n_jobs)
    #interpolate_kernels_optimized(ref_kernels, s_vals, t_vals, S_data, T_data, 
                #                           batch_size=20, n_jobs=n_jobs, optimize_memory=True)

# Convenience wrapper
def interpolate_kernels_fast_safe(ref_kernels, s_vals, t_vals, s_data, t_data, n_jobs=-1, method='auto'):
    """
    Fast and safe kernel interpolation with automatic method selection.
    
    Parameters:
    -----------
    method : str
        'auto' - automatically choose based on problem size
        'fast' - fast, higher memory usage
        'balanced' - balance of speed and memory
        'safe' - prioritize memory safety
    """
    if method == 'auto':
        return choose_interpolation_method(ref_kernels, s_vals, t_vals, s_data, t_data, n_jobs, prefer_speed=True)
    elif method == 'fast':
        return interpolate_kernels_super_fast(ref_kernels, s_vals, t_vals, s_data, t_data, n_jobs)
    elif method == 'balanced':
        S_data, T_data = np.meshgrid(s_data, t_data, indexing='ij')
        return interpolate_kernels_optimized(ref_kernels, s_vals, t_vals, S_data, T_data, 
                                           batch_size=40, n_jobs=n_jobs)
    else:  # 'safe'
        S_data, T_data = np.meshgrid(s_data, t_data, indexing='ij')
        return interpolate_kernels_optimized(ref_kernels, s_vals, t_vals, S_data, T_data, 
                                           batch_size=15, n_jobs=n_jobs, optimize_memory=True)
    

