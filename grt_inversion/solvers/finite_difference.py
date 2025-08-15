"""
Finite difference implementations for transport solver using WENO schemes and Lax-Friedrichs sweeping as detailed in paper.
"""

import numpy as np
import matplotlib.pyplot as plt

def extrapolate_array_2d(u, ng = 2):
    """
    Extrapolate a 2D array by `ng` ghost cells on each side using linear extrapolation.
    """
    n1, n2 = u.shape
    u_ext = np.zeros((n1 + 2*ng, n2 + 2*ng))
    
    # Insert the interior
    u_ext[ng:-ng, ng:-ng] = u

    # Extrapolate top and bottom rows
    for i in range(ng):
        u_ext[i, ng:-ng] = 2 * u[0, :] - u[min(1, n1-1), :]
        u_ext[-1 - i, ng:-ng] = 2 * u[-1, :] - u[max(n1-2, 0), :]

    # Extrapolate left and right columns (including corners)
    for j in range(ng):
        u_ext[:, j] = 2 * u_ext[:, ng] - u_ext[:, ng + 1]
        u_ext[:, -1 - j] = 2 * u_ext[:, -ng - 1] - u_ext[:, -ng - 2]

    return u_ext

# TODO: Check these. Need a boundary condition probs
def add_ghost_nodes(u, boundary_type='extrapolate') -> np.ndarray:
    """
    Add ghost nodes around the domain for WENO stencil
    Returns extended array with 2 ghost nodes on each side
    """
    nx, ny = u.shape
    u_extended = np.zeros((nx + 4, ny + 4))
    
    # Copy interior values
    u_extended[2:-2, 2:-2] = u
    
    if boundary_type == 'extrapolate':
        # Left boundary (x-direction)
        u_extended[0, 2:-2] = 2*u[0, :] - u[1, :]
        u_extended[1, 2:-2] = u[0, :]
        
        # Right boundary (x-direction)
        u_extended[-2, 2:-2] = u[-1, :]
        u_extended[-1, 2:-2] = 2*u[-1, :] - u[-2, :]
        
        # Bottom boundary (y-direction)
        u_extended[2:-2, 0] = 2*u[:, 0] - u[:, 1]
        u_extended[2:-2, 1] = u[:, 0]
        
        # Top boundary (y-direction)
        u_extended[2:-2, -2] = u[:, -1]
        u_extended[2:-2, -1] = 2*u[:, -1] - u[:, -2]
        
        # Corners
        u_extended[0, 0] = u_extended[0, 2]
        u_extended[0, 1] = u_extended[0, 2]
        u_extended[1, 0] = u_extended[1, 2]
        u_extended[1, 1] = u_extended[1, 2]
        
        u_extended[0, -1] = u_extended[0, -3]
        u_extended[0, -2] = u_extended[0, -3]
        u_extended[1, -1] = u_extended[1, -3]
        u_extended[1, -2] = u_extended[1, -3]
        
        u_extended[-1, 0] = u_extended[-1, 2]
        u_extended[-1, 1] = u_extended[-1, 2]
        u_extended[-2, 0] = u_extended[-2, 2]
        u_extended[-2, 1] = u_extended[-2, 2]
        
        u_extended[-1, -1] = u_extended[-1, -3]
        u_extended[-1, -2] = u_extended[-1, -3]
        u_extended[-2, -1] = u_extended[-2, -3]
        u_extended[-2, -2] = u_extended[-2, -3]
    
    return u_extended

# WENO as detailed in Ganster & Rieder paper Approximate Inversion of a Class of Generalized Radon Transforms
def weno_reconstruction(v, h, direction='positive'):
    """
    WENO5 reconstruction as described in paper
    v should be a 5-point stencil: [v_{i-2}, v_{i-1}, v_i, v_{i+1}, v_{i+2}]
    Returns the reconstructed value at the interface
    """
    eps = 1e-6
    
    if direction == 'positive':  # u^+_{i, j}
        v0, v1, v2, v3, v4 = v # v_i-2, v_i-1, v_i, v_i+1, v_i+2
        
        gamma = (eps + (v4 - 2*v3 + v2)**2) / (eps + (v3 - 2*v2 + v1)**2)
        omega = 1 / (1 + 2*gamma**2)

        result = (1 - omega) * (v3 - v1)/(2*h) + omega * (-v4 + 4*v3 - 3*v2)/(2*h)
        
        
    else:  # direction == 'negative', u^-_{i, j}
        v0, v1, v2, v3, v4 = v # v_i-2, v_i-1, v_i, v_i+1, v_i+2
        
        gamma = (eps + (v2 - 2*v3 + v0)**2) / (eps + (v3 - 2*v2 + v1)**2)
        omega = 1 / (1 + 2*gamma**2)

        result = (1 - omega) * (v3 - v1)/(2*h) + omega * (3*v2 - 4*v1 + v0)/(2*h)

    return result


def compute_weno_derivatives(u_extended, h, sweep_x, sweep_y):
    """Compute WENO derivatives."""

    nx_ext, ny_ext = u_extended.shape
    nx, ny = nx_ext - 4, ny_ext - 4
    
    # Initialize derivative arrays
    u1_plus = np.zeros((nx, ny))   # u^+_{i-1,j} 
    u1_minus = np.zeros((nx, ny))  # u^-_{i+1,j}
    u2_plus = np.zeros((nx, ny))   # u^+_{i,j-1}
    u2_minus = np.zeros((nx, ny))  # u^-_{i,j+1}
    
    h1, h2 = h
    
    for i in range(nx):
        for j in range(ny):
            # Adjust indices for ghost nodes (interior points start at index 2)
            ii, jj = i + 2, j + 2
            
            # Debug: Check bounds before slicing
            if ii < 2 or ii >= nx_ext - 2 or jj < 2 or jj >= ny_ext - 2:
                continue
                
            # X-direction derivatives (equation 4.2 from screenshot)
            try:
                if sweep_x > 0:  # Forward sweep
                    # u^+_{i,j} using upwind stencil: need points [i-2, i-1, i, i+1, i+2]
                    stencil_x = u_extended[ii-2:ii+3, jj] 
                    if len(stencil_x) == 5:
                        u1_plus[i, j] = weno_reconstruction(stencil_x, h[0], 'positive')
                    
                    # u^-_{i,j} using downwind stencil: need points [i-1, i, i+1, i+2, i+3]  
                    if ii + 3 < nx_ext:
                        stencil_x = u_extended[ii-2:ii+3, jj] 
                        if len(stencil_x) == 5:
                            u1_minus[i, j] = weno_reconstruction(stencil_x, h[0], 'negative')
                else:  # Backward sweep
                    # Reverse the stencil direction
                    stencil_x = u_extended[ii-2:ii+3, jj]
                    if len(stencil_x) == 5:
                        u1_plus[i, j] = weno_reconstruction(stencil_x[::-1], h[0], 'negative')
                    
                    if ii + 3 < nx_ext:
                        stencil_x = u_extended[ii-2:ii+3, jj]
                        if len(stencil_x) == 5:
                            u1_minus[i, j] = weno_reconstruction(stencil_x[::-1], h[0], 'positive')
                
                # Y-direction derivatives  
                if sweep_y > 0:  # Forward sweep
                    # u^+_{i,j} using upwind stencil: need points [j-2, j-1, j, j+1, j+2]
                    stencil_y = u_extended[ii, jj-2:jj+3]  # Should give 5 points
                    if len(stencil_y) == 5:
                        u2_plus[i, j] = weno_reconstruction(stencil_y, h[1], 'positive')
                    
                    # u^-_{i,j} using downwind stencil: need points [j-1, j, j+1, j+2, j+3]
                    if jj + 3 < ny_ext:
                        stencil_y = u_extended[ii, jj-2:jj+3]  # Should give 5 points  
                        if len(stencil_y) == 5:
                            u2_minus[i, j] = weno_reconstruction(stencil_y, h[1], 'negative')
                else:  # Backward sweep
                    # Reverse the stencil direction
                    stencil_y = u_extended[ii, jj-2:jj+3]
                    if len(stencil_y) == 5:
                        u2_plus[i, j] = weno_reconstruction(stencil_y[::-1], h[1], 'negative')
                    
                    if jj + 3 < ny_ext:
                        stencil_y = u_extended[ii, jj-2:jj+3]
                        if len(stencil_y) == 5:
                            u2_minus[i, j] = weno_reconstruction(stencil_y[::-1], h[1], 'positive')
                            
            except Exception as e:
                print(f"Error at ({i}, {j}), extended indices ({ii}, {jj}): {e}")
                continue
    
    # Compute derivatives as on p.851
    dudx_plus = 1/2 * (u1_plus + u1_minus) # 2^{-1}(u^+_{i,j} + u^-_{i,j})
    dudx_minus = 1/2 * (u1_plus - u1_minus) # 2^{-1}(u^+_{i,j} - u^-_{i,j})
    
    dudy_plus = 1/2 * (u2_plus + u2_minus) # 2^{-1}(u^+_{i,j} - u^-_{i,j})
    dudy_minus = 1/2 * (u2_plus - u2_minus) # 2^{-1}(u^+_{i,j} - u^-_{i,j})
    
    return dudx_plus, dudx_minus, dudy_plus, dudy_minus

def compute_weno_derivatives_ptw(u_extended, u, h, sweep_x, sweep_y, i, j):
    """Compute WENO derivatives pointwise."""
    nx_ext, ny_ext = u_extended.shape
    nx, ny = nx_ext - 4, ny_ext - 4
    
    ii, jj = i + 2, j + 2

    try:
        if sweep_x > 0: # Forward sweep
            # u^+_{i,j} using upwind stencil: need points [i-2, i-1, i, i+1, i+2]
            stencil_x = np.concatenate([u[ii-2:ii, jj], u_extended[ii:ii+3, jj]])
            u1_plus = weno_reconstruction(stencil_x, h[0], 'positive')
            u1_minus = weno_reconstruction(stencil_x, h[0], 'negative')

        else: # Backward sweep
            # Reverse the stencil direction
            stencil_x = np.concatenate([u_extended[ii-2:ii+1, jj], u[ii+1:ii+3, jj]])
            u1_plus = weno_reconstruction(stencil_x, h[0], 'positive')
            u1_minus = weno_reconstruction(stencil_x, h[0], 'negative')
        
        # Y-direction derivatives  
        if sweep_y > 0: # Forward sweep
            # u^+_{i,j} using upwind stencil: need points [j-2, j-1, j, j+1, j+2]
            stencil_y = np.concatenate([u[ii, jj-2:jj], u_extended[ii, jj:jj+3]]) 
            u2_plus = weno_reconstruction(stencil_y, h[1], 'positive')
            u2_minus = weno_reconstruction(stencil_y, h[1], 'negative')

        else: # Backward sweep
            # Reverse the stencil direction
            stencil_y = np.concatenate([u_extended[ii, jj-2:jj+1], u[ii, jj+1:jj+3]])
            u2_plus = weno_reconstruction(stencil_y, h[1], 'positive')
            u2_minus = weno_reconstruction(stencil_y, h[1], 'negative')
                            
    except Exception as e:
        print(f"Error at ({i}, {j}), extended indices ({ii}, {jj}): {e}")
        
    
    # Compute derivatives as on p.851
    dudx_plus = 1/2 * (u1_plus + u1_minus) # 2^{-1}(u^+_{i,j} + u^-_{i,j})
    dudx_minus = 1/2 * (u1_plus - u1_minus) # 2^{-1}(u^+_{i,j} - u^-_{i,j})
    
    dudy_plus = 1/2 * (u2_plus + u2_minus) # 2^{-1}(u^+_{i,j} - u^-_{i,j})
    dudy_minus = 1/2 * (u2_plus - u2_minus) # 2^{-1}(u^+_{i,j} - u^-_{i,j})
    
    return dudx_plus, dudx_minus, dudy_plus, dudy_minus

def get_valid_neighbors_4(i, j, nx, ny):
    """Get valid 4-connected neighbors."""
    neighbors = []
    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]: #, (0,0)]:#, (1,1), (1,-1), (-1,1), (-1,-1)]:
        ni, nj = i + di, j + dj
        if 0 <= ni < nx and 0 <= nj < ny:
            neighbors.append((ni, nj))
    return neighbors

def laplacian_2d_alternative(f, dx=1.0, dy=1.0, boundary='zero'):
    """Compute the Laplacian of a 2D function using finite differences."""
    f = np.asarray(f)
    if f.ndim != 2:
        raise ValueError("Input must be a 2D array")
    
    
    laplacian = np.zeros_like(f)
    
    if boundary == 'zero':
        # Zero boundary conditions (f = 0 outside domain)
        # Interior points
        laplacian[1:-1, 1:-1] = (
            (f[1:-1, 2:] - 2*f[1:-1, 1:-1] + f[1:-1, :-2]) / dx**2 +
            (f[2:, 1:-1] - 2*f[1:-1, 1:-1] + f[:-2, 1:-1]) / dy**2
        )
        
    elif boundary == 'periodic':
        # Periodic boundary conditions
        # Use numpy roll for periodic boundaries
        f_right = np.roll(f, -1, axis=1)  # f(i, j+1)
        f_left = np.roll(f, 1, axis=1)    # f(i, j-1)
        f_up = np.roll(f, -1, axis=0)     # f(i+1, j)
        f_down = np.roll(f, 1, axis=0)    # f(i-1, j)
        
        laplacian = (
            (f_right - 2*f + f_left) / dx**2 +
            (f_up - 2*f + f_down) / dy**2
        )
        
    elif boundary == 'neumann':
        # Neumann boundary conditions (zero derivative at boundary)
        # Interior points
        laplacian[1:-1, 1:-1] = (
            (f[1:-1, 2:] - 2*f[1:-1, 1:-1] + f[1:-1, :-2]) / dx**2 +
            (f[2:, 1:-1] - 2*f[1:-1, 1:-1] + f[:-2, 1:-1]) / dy**2
        )
        
        # Edges (using one-sided differences for Neumann conditions)
        # Top and bottom edges
        laplacian[0, 1:-1] = (
            (f[0, 2:] - 2*f[0, 1:-1] + f[0, :-2]) / dx**2 +
            (2*f[1, 1:-1] - 2*f[0, 1:-1]) / dy**2
        )
        laplacian[-1, 1:-1] = (
            (f[-1, 2:] - 2*f[-1, 1:-1] + f[-1, :-2]) / dx**2 +
            (2*f[-2, 1:-1] - 2*f[-1, 1:-1]) / dy**2
        )
        
        # Left and right edges  
        laplacian[1:-1, 0] = (
            (2*f[1:-1, 1] - 2*f[1:-1, 0]) / dx**2 +
            (f[2:, 0] - 2*f[1:-1, 0] + f[:-2, 0]) / dy**2
        )
        laplacian[1:-1, -1] = (
            (2*f[1:-1, -2] - 2*f[1:-1, -1]) / dx**2 +
            (f[2:, -1] - 2*f[1:-1, -1] + f[:-2, -1]) / dy**2
        )
        
        # Corners
        laplacian[0, 0] = (2*f[0, 1] - 2*f[0, 0]) / dx**2 + (2*f[1, 0] - 2*f[0, 0]) / dy**2
        laplacian[0, -1] = (2*f[0, -2] - 2*f[0, -1]) / dx**2 + (2*f[1, -1] - 2*f[0, -1]) / dy**2
        laplacian[-1, 0] = (2*f[-1, 1] - 2*f[-1, 0]) / dx**2 + (2*f[-2, 0] - 2*f[-1, 0]) / dy**2
        laplacian[-1, -1] = (2*f[-1, -2] - 2*f[-1, -1]) / dx**2 + (2*f[-2, -1] - 2*f[-1, -1]) / dy**2

    elif boundary == 'extrapolate':
        f_extended = extrapolate_array_2d(f, 2)
        f_extended_laplacian = laplacian_2d(f_extended, dx, dy, 'zero')
        laplacian = f_extended_laplacian[2:-2, 2:-2]
        
    else:
        raise ValueError("boundary must be 'zero', 'periodic', 'neumann', or 'extrapolate")
    
    return laplacian

def laplacian_2d(f, dx, dy, boundary='extrapolate'):
    """
    Compute 2D Laplacian using finite differences.
    """
    laplacian = np.zeros_like(f)
    
    # Interior points
    laplacian[1:-1, 1:-1] = (
        (f[2:, 1:-1] - 2*f[1:-1, 1:-1] + f[:-2, 1:-1]) / dx**2 +
        (f[1:-1, 2:] - 2*f[1:-1, 1:-1] + f[1:-1, :-2]) / dy**2
    )
    
    # Handle boundaries based on boundary condition
    if boundary == 'extrapolate':
        # Simple extrapolation for boundaries
        laplacian[0, :] = laplacian[1, :]
        laplacian[-1, :] = laplacian[-2, :]
        laplacian[:, 0] = laplacian[:, 1]
        laplacian[:, -1] = laplacian[:, -2]
    elif boundary == 'zero':
        # Zero boundary conditions
        laplacian[0, :] = 0
        laplacian[-1, :] = 0
        laplacian[:, 0] = 0
        laplacian[:, -1] = 0
    
    return laplacian


def solve_transport_equation_fd(c, tau, X, Y, x_s, h, f):
    """
    Solve transport equation 2 grad a . grad tau + a laplace tau = 0 with full WENO scheme.

    Parameters:
    -----------
    c : ndarray
        Speed field
    tau : ndarray
        Travel time field from eikonal equation
    X, Y : ndarray
        Coordinate meshgrids
    x_s : array-like
        Source coordinates
    h : array-like
        Grid spacing
    f : ndarray
        Right-hand side
        
    Returns:
    --------
    tuple
        (amplitude, a_0, a_1) where amplitude = a_0 * a_1
    """
    if f.shape != c.shape:
        return False
    
    # Compute distance from every grid point to the source point
    distances = np.sqrt((X - x_s[0])**2 + (Y - x_s[1])**2)

    # Find the index of the minimum distance
    min_index = np.unravel_index(np.argmin(distances), X.shape)
    #print(min_index)
    source_index = [min_index[0], min_index[1]]

    nx, ny = X.shape[0], X.shape[1]
    
    # Initialize with good starting values according to paper
    dist = np.sqrt((X - x_s[0])**2 + (Y - x_s[1])**2)
    if np.any(dist == 0):
        dist[dist == 0] = 1  # Avoid division by zero
        #dist = np.maximum(dist, 1e-3)
        a_0 = 1 / (2 * np.sqrt(2 * np.pi) * np.sqrt(dist))
        a_0[source_index[0], source_index[1]] = np.max(np.delete(a_0.flatten(), source_index[0] * a_0.shape[1] + source_index[1]))
    else:
        a_0 = 1 / (2 * np.sqrt(2 * np.pi) * np.sqrt(dist))
        #a_0[source_index[0], source_index[1]] = np.max(np.delete(a_0.flatten(), source_index[0] * a_0.shape[1] + source_index[1]))

    u_old = np.sqrt(c) 
    
    # Mark fixed nodes around the source point
    '''
    valid_neighbors = get_valid_neighbors_4(source_index[0], source_index[1], nx, ny)
    memory = np.zeros_like(X)
    for neighbor in valid_neighbors:
        memory[neighbor] = 1
    memory[source_index[0], source_index[1]] = 1
    '''
    
    # Set boundary conditions in a radius around the actual source location
    boundary_radius = max(h[0], h[1]) * 1.5  # 1.5 grid spacings
    boundary_mask = distances <= boundary_radius
    memory = np.zeros_like(X)
    memory[boundary_mask] = 1

    # Initialize boundary values properly
    boundary_indices = np.where(boundary_mask)
    for i, j in zip(boundary_indices[0], boundary_indices[1]):
        u_old[i, j] = np.sqrt(c[i, j]) * a_0[i, j]

    h_1, h_2 = h[0], h[1]
    
    # Compute terms for Hamiltonian and artificial viscosities
    grad_tau_x, grad_tau_y = np.gradient(tau, h_1, h_2)
    laplacian_tau = laplacian_2d(tau, h_1, h_2, 'extrapolate') 
    grad_a0_x, grad_a0_y = np.gradient(a_0, h_1, h_2)
    
    # Factorize amplitude a = a_0 * a_1
    term_1 = 2 * (grad_a0_x * grad_tau_x + grad_a0_y * grad_tau_y) + a_0 * laplacian_tau
    term_2_x = 2 * a_0 * grad_tau_x  
    term_2_y = 2 * a_0 * grad_tau_y
    
    # Artificial viscosities
    alpha_1 = np.max(np.abs(2*grad_a0_x*grad_tau_x + a_0*laplacian_tau) + np.abs(2*a_0*grad_tau_x))
    alpha_2 = np.max(np.abs(2*grad_a0_y*grad_tau_y + a_0*laplacian_tau) + np.abs(2*a_0*grad_tau_y))
    
    # Add minimum viscosity for stability when source is off-grid
    alpha_1 = max(alpha_1, 0.1)
    alpha_2 = max(alpha_2, 0.1)

    # Hamiltonian function
    def H(u_val, p_val, q_val, i, j):
        return term_2_x[i, j] * p_val + term_2_y[i, j] * q_val + term_1[i, j] * u_val
    
    u = np.zeros_like(u_old)
    u[memory == 1] = u_old[memory == 1].copy()
    
    # Main iteration loop with Lax-Friedrichs sweeping
    max_iter = 500
    iteration = 0
    delta = 1e-2
    

    while True:
        # Lax-Friedrichs sweeping with 4 directions
        for sweep_x, sweep_y in [(1, 1), (-1, 1), (1, -1), (-1, -1)]:

            # Add ghost nodes for WENO
            # u_extended = add_ghost_nodes(u_old, 'extrapolate') # TODO: Check if this is ok to use; consider boundary conditions
            u_extended = extrapolate_array_2d(u = u_old, ng = 2)

            # check ranges
            i_range = range(0, nx) if sweep_x > 0 else range(nx-1, -1, -1)
            j_range = range(0, ny) if sweep_y > 0 else range(ny-1, -1, -1)
            
            for i in i_range:
                for j in j_range:
                    if memory[i, j] != 1:
                        
                        # Compute WENO derivatives
                        dudx_plus, dudx_minus, dudy_plus, dudy_minus = compute_weno_derivatives_ptw(
                                                                            u_extended, add_ghost_nodes(u), h, sweep_x, sweep_y, i, j)
                        # Apply the update formula from equation (4.3)
                        hamiltonian_val = H(u_old[i, j], dudx_plus, dudy_plus, i, j)
                        
                        # Update using Lax-Friedrichs scheme
                        u[i, j] = (1 / (alpha_1/h_1 + alpha_2/h_2) * 
                                  (f[i, j] - hamiltonian_val + alpha_1 * dudx_minus + alpha_2 * dudy_minus) + 
                                  u_old[i, j])
                        
        error_iter = np.linalg.norm(u - u_old, np.inf)/np.linalg.norm(u_old, np.inf)
        
        if error_iter < delta or iteration >= max_iter:
            break
        u_old = u.copy() 
        iteration += 1
    
    # print(f"Converged after {iteration} iterations")
    return a_0 * u, a_0, u

# TODO: clean this function up
def plot_transport_solver_fd():
    """Plot results from the finite difference transport solver."""
    x_min, x_max, nx = -5, 5, 121  
    y_min, y_max, ny = 0.0, 10, 121 

    #x_min, x_max, nx = -5, 5, 264 
    #y_min, y_max, ny = 0.0, 10, 264

    # Create 1D coordinate arrays
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    
    # Create 2D meshgrid: X has x values, Y has y values
    X, Y = np.meshgrid(x, y, indexing='ij')  # (nx, ny) shape
    dx = np.array([x[1]-x[0], y[1]-y[0]])
    source = (nx // 2, ny // 2)

    c = np.ones_like(X)

    x_s = np.array([(x[source[0] - 1] + x[source[0]])/2, (y[source[1] - 1] + y[source[1]])/2])
    #x_s = np.array([x[source[0]], y[source[1]]])
    tau = np.sqrt((X - x_s[0])**2 + (Y - x_s[1])**2)
    

    a_numerical, a_0, a_1 = solve_transport_equation_fd(c, tau, X, Y, x_s, dx, np.zeros_like(X))
    # a_numerical, a_0, a_1 = solve_transport_equation_fixed(c, tau, X, Y, x_s, dx, np.zeros_like(X))
    div = tau
    # div[tau == 0] = 1#e-2
    a_test = 1/(2*np.sqrt(2*np.pi)) * 1/np.sqrt(div)

    print("Transport equation solved successfully!")

    grad_tau_x, grad_tau_y = np.gradient(tau, dx[0], dx[1]) #compute_gradient(tau, dx[0], dx[1])
    #laplacian_tau = compute_laplacian(tau, dx[0], dx[1])
    #laplacian_tau = laplacian_2d(tau, dx[0], dx[1], 'zero')
    laplacian_tau = laplacian_2d(tau, dx[0], dx[1], 'extrapolate')

    grad_a_x, grad_a_y = np.gradient(a_1 * a_0, dx[0], dx[1])

    transport_residual = 2 * (grad_a_x * grad_tau_x + grad_a_y * grad_tau_y) + a_numerical * laplacian_tau
    #print(transport_residual)


    data_list = [c, tau, a_0, a_1, a_numerical, a_test]#np.mean(a_numerical.flatten() / a_test.flatten()) * a_test]
    vmin = min([np.min(data) for data in data_list])
    vmax = max([np.max(data) for data in data_list])

    #fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig, axes = plt.subplots(2, 4, figsize=(25, 10))
    axes = axes.flatten()

    # Use vmin/vmax in first six plots
    pcm0 = axes[0].pcolormesh(X, Y, c, cmap='inferno', shading='auto')#, vmin=vmin, vmax=vmax)
    axes[0].set_title(r"Background velocity $c$")
    fig.colorbar(pcm0, ax=axes[0], label='Time')

    pcm0 = axes[1].pcolormesh(X, Y, tau, cmap='inferno', shading='auto')#, vmin=vmin, vmax=vmax)
    axes[1].set_title(r"Travel Time $\tau$")
    fig.colorbar(pcm0, ax=axes[1], label='Time')

    pcm1 = axes[2].pcolormesh(X, Y, a_0, cmap='inferno', shading='auto')#, vmin=vmin, vmax=vmax)
    axes[2].set_title(r"Amplitude $a_0$")
    fig.colorbar(pcm1, ax=axes[2], label='Amplitude')

    pcm1 = axes[3].pcolormesh(X, Y, a_1, cmap='inferno', shading='auto')#, vmin=vmin, vmax=vmax)
    axes[3].set_title(r"Amplitude $a_1$")
    fig.colorbar(pcm1, ax=axes[3], label='Amplitude')

    #pcm1 = axes[4].pcolormesh(X, Y, a_test, cmap='inferno', shading='auto')#, vmin=vmin, vmax=vmax)
    pcm1 = axes[4].pcolormesh(X, Y, a_numerical, cmap='inferno', shading='auto')#, vmin=vmin, vmax=vmax)
    axes[4].set_title(r"Amplitude $a$")
    fig.colorbar(pcm1, ax=axes[4], label='Amplitude')

    pcm1 = axes[5].pcolormesh(X, Y, transport_residual, cmap='coolwarm', shading='auto')
    axes[5].set_title(r"plug numerical sol in eq")
    fig.colorbar(pcm1, ax=axes[5], label='Pointwise Error')
    #'''
    pcm1 = axes[6].pcolormesh(X, Y, a_test, cmap='inferno', shading='auto')
    axes[6].set_title(r"analytical solution")
    fig.colorbar(pcm1, ax=axes[6], label='amplitude')

    pcm1 = axes[7].pcolormesh(X, Y, a_numerical - a_test, cmap='coolwarm', shading='auto')
    axes[7].set_title(r"pointwise error")
    fig.colorbar(pcm1, ax=axes[7], label='Pointwise Error')
    #'''

    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    plt.show()

# TODO: clean up this function
def convergence_test_transport_solver_fd():
    """Test convergence of fast sweeping transport solver."""
    # List of mesh sizes (from coarse to fine)

    # TODO: place source not on grid point but just outside to circumvent singularity or having to smooth it
    meshsize_values = [2**k for k in range(2, 8)]

    errors_l2, errors_l1, errors_linf = [], [], []
    errors_l2_a1, errors_l1_a1, errors_linf_a1 = [], [], []
    errors_l2_a0, errors_l1_a0, errors_linf_a0 = [], [], []

    h_list = []

    # Loop over all mesh sizes to compute error
    for meshsize in meshsize_values:
        print(meshsize)

        # source in domain
        #x = y = np.linspace(0, 1, meshsize)
        #source = (meshsize // 2, meshsize // 2)
        #x_s = np.array([x[source[0]], y[source[1]]])
        
        # source just outside
        x = y = np.linspace(.01, 1, meshsize)
        #source = (meshsize // 2, 0)#meshsize // 2)
        x_s = np.array([x[meshsize//2], 0])

        X, Y = np.meshgrid(x, y, indexing='ij')

        c = np.ones_like(X)
        dx = x[1] - x[0]

        tau = np.sqrt((X - x_s[0])**2 + (Y - x_s[1])**2)

        div = tau
        div[tau == 0] = 1e-4
        a_ref_interp = 1/(2*np.sqrt(2*np.pi)) * 1/np.sqrt(div)

        # same as in solver
        #a_ref_interp[source[0], source[1]] = np.max(np.delete(a_ref_interp.flatten(), source[0] * a_ref_interp.shape[1] + source[1]))

        a_numerical, a0, a1 = solve_transport_equation_fd(c, tau, X, Y, x_s, [dx, dx], np.zeros_like(X))
        
        #error_l2 = np.linalg.norm(np.delete(np.abs(a_numerical - a_ref_interp), source))
        error_l2 = np.linalg.norm(np.abs(a_numerical - a_ref_interp))
        errors_l2.append(error_l2)

        #error_l1 = np.sum(np.delete(np.abs(a_numerical - a_ref_interp), source).flatten())
        error_l1 = np.sum(np.abs(a_numerical - a_ref_interp).flatten())
        errors_l1.append(error_l1)

        #error_linf = np.max(np.delete(np.abs(a_numerical - a_ref_interp), source).flatten())
        error_linf = np.max(np.abs(a_numerical - a_ref_interp).flatten())
        errors_linf.append(error_linf)

        #error_l2_a0 = np.linalg.norm(np.delete(np.abs(a0 - a_ref_interp), source))
        error_l2_a0 = np.linalg.norm(np.abs(a0 - a_ref_interp))
        errors_l2_a0.append(error_l2_a0)

        #error_l1_a0 = np.sum(np.delete(np.abs(a0 - a_ref_interp), source).flatten())
        error_l1_a0 = np.sum(np.abs(a0 - a_ref_interp).flatten())
        errors_l1_a0.append(error_l1_a0)

        #error_linf_a0 = np.max(np.delete(np.abs(a0 - a_ref_interp), source).flatten())
        error_linf_a0 = np.max(np.abs(a0 - a_ref_interp).flatten())
        errors_linf_a0.append(error_linf_a0)

        #error_l2_a1 = np.linalg.norm(np.delete(np.abs(a1 - a_ref_interp), source))
        error_l2_a1 = np.linalg.norm(np.abs(a1 - a_ref_interp))
        errors_l2_a1.append(error_l2_a1)

        #error_l1_a1 = np.sum(np.delete(np.abs(a1 - a_ref_interp), source).flatten())
        error_l1_a1 = np.sum(np.abs(a1 - a_ref_interp).flatten())
        errors_l1_a1.append(error_l1_a1)

        #error_linf_a1 = np.max(np.delete(np.abs(a1 - a_ref_interp), source).flatten())
        error_linf_a1 = np.max(np.abs(a1 - a_ref_interp).flatten())
        errors_linf_a1.append(error_linf_a1)

        h_list.append(dx)

        print(f"Mesh: {meshsize}x{meshsize}, h = {dx:.5f}, L2 error = {error_l2:.5e}, L1 error = {error_l1:.5e}, Linf error = {error_linf:.5e}")

        # Convergence of a_1 also interesting (more what we expect I think)

    fig, axes = plt.subplots(1, 3, figsize=(20, 7)) #plt.figure(figsize=(6, 4))
    start = 0
    stop = None

    axes[0].loglog(h_list[start:stop], np.array(errors_l2[start:stop])*np.array(h_list[start:stop])**1, 'o-', label=r"$L^2$ error")
    axes[0].loglog(h_list[start:stop], np.array(errors_l1[start:stop])*np.array(h_list[start:stop])**2, 'o-', label=r"$L^1$ error")
    #axes[0].loglog(h_list[start:stop], np.array(errors_linf[start:stop])*np.array(h_list[start:stop])**0, 'o-', label=r"$L^\infty$ error")
    axes[0].loglog(h_list[start:stop], np.array(h_list[start:stop])**1, '--', label="1st-order reference")
    axes[0].loglog(h_list[start:stop], np.array(h_list[start:stop])**2, '--', label="2nd-order reference")
    #axes[0].loglog(h_list[start:stop], np.array(h_list[start:stop])**(1/4), '--', label="3/4st-order reference")
    #axes[0].loglog(h_list[start:stop], np.array(h_list[start:stop])**3, '--', label="3rd-order reference")

    axes[1].loglog(h_list[start:stop], np.array(errors_l2_a1[start:stop])*np.array(h_list[start:stop])**1, 'o-', label=r"$L^2$ error")
    axes[1].loglog(h_list[start:stop], np.array(errors_l1_a1[start:stop])*np.array(h_list[start:stop])**2, 'o-', label=r"$L^1$ error")
    #axes[1].loglog(h_list[start:stop], np.array(errors_linf_a1[start:stop])*np.array(h_list[start:stop])**0, 'o-', label=r"$L^\infty$ error")
    axes[1].loglog(h_list[start:stop], np.array(h_list[start:stop])**1, '--', label="1st-order reference")
    axes[1].loglog(h_list[start:stop], np.array(h_list[start:stop])**2, '--', label="2nd-order reference")
    #axes[1].loglog(h_list[start:stop], np.array(h_list[start:stop])**(1/4), '--', label="1/4th-order reference")
    #axes[1].loglog(h_list[start:stop], np.array(h_list[start:stop])**0, '--', label="0-order reference")

    axes[2].loglog(h_list[start:stop], np.array(errors_l2_a0[start:stop])*np.array(h_list[start:stop])**1, 'o-', label=r"$L^2$ error")
    axes[2].loglog(h_list[start:stop], np.array(errors_l1_a0[start:stop])*np.array(h_list[start:stop])**2, 'o-', label=r"$L^1$ error")
    #axes[2].loglog(h_list[start:stop], np.array(errors_linf_a0[start:stop])*np.array(h_list[start:stop])**0, 'o-', label=r"$L^\infty$ error")
    axes[2].loglog(h_list[start:stop], np.array(h_list[start:stop])**1, '--', label="1st-order reference")
    axes[2].loglog(h_list[start:stop], np.array(h_list[start:stop])**2, '--', label="2nd-order reference")
    #axes[2].loglog(h_list[start:stop], np.array(h_list[start:stop])**(1/4), '--', label="3/4st-order reference")
    #axes[2].loglog(h_list[start:stop], np.array(h_list[start:stop])**0, '--', label="0-order reference")


    for i in range(3):
        axes[i].set_xlabel(r"$h$")
        axes[i].set_ylabel(r"$\|a - a_{ref}\|$")
        axes[i].legend()

    axes[0].set_title(r"Convergence of $a = a_0 a_1$")
    axes[1].set_title(r"Convergence of $a_1$")
    axes[2].set_title(r"Convergence of $a_0$")
    plt.show()

