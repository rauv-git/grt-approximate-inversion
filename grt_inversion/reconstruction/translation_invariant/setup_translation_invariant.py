"""
Methods to extend domain suitably and solve required equations.
"""

import numpy as np
import math

from ...core.eikonal import solve_eikonal
from ...core.transport import solve_transport_equation


def extend_and_solve(x, y, dx, alpha, velocity):
    """Extend grid suitably and assemble integration factor and travel time function."""
    
    alpha_ind = math.ceil(alpha / dx[0])
    x_left_extension = []
    for i in range(1, alpha_ind + 1):
        x_left_extension.append(x[0] - i * dx[0])

    x_left = np.append(np.sort(np.array(x_left_extension)), x)

    x_right_extension = []
    for i in range(1, alpha_ind + 1):
        x_right_extension.append(x[-1] + i * dx[0])

    x_right = np.append(x, np.sort(np.array(x_right_extension)))

    x_extended = np.unique(np.append(x_left, x_right))

    # Generate extended meshgrid
    X, Y = np.meshgrid(x, y, indexing='ij')
    X_extended, Y_extended = np.meshgrid(x_extended, y, indexing='ij')
    
    # Background velocity on extended mesh
    c_extended = velocity(X_extended, Y_extended)
    
    # Determine sources
    source = (np.argmin(abs(x_extended)), np.where(y == 0)[0][0])
    
    # Compute eikonal equation and slice
    tau = solve_eikonal(c_extended, dx, source)   

    tau_minus = tau[2*alpha_ind:len(x)+2*alpha_ind, :]
    tau_plus = tau[:len(x), :]

    x_s = [x_extended[source[0]], y[source[1]]]
    
    # Do same for transport equation
    a_extended = solve_transport_equation(c_extended, tau, X_extended, Y_extended, x_s, dx, np.zeros_like(X_extended))
    
    a_minus = a_extended[alpha_ind:len(x)+alpha_ind, :]
    a_plus = a_extended[:len(x), :]
 
    # Compute integrand factor I_M
    grad_tau_minus = np.gradient(tau_minus, *dx, edge_order=2) 
    grad_tau_plus = np.gradient(tau_plus, *dx, edge_order=2)
    dot_product = (grad_tau_minus[0] * grad_tau_plus[0] + grad_tau_minus[1] * grad_tau_plus[1])

    c = velocity(X, Y)

    denom = 1 + c**2 * dot_product
    
    # Check whether denominantor is positive (need to take square root)
    if np.any(denom < 0):
        print("Negative denominator: forward scattering or numerical instability.")
        denom = np.maximum(denom, 1e-12)  # Prevent division by zero
        I_M = (a_minus * a_plus) / (c * np.sqrt(denom))
        #I_M = (c * a_minus * a_plus) / (np.sqrt(denom))
        # Filter out problematic contour points
        valid_mask = (denom > 1e-10) & (np.abs(I_M) < 1e6)
        I_M[~valid_mask] = 0
    else:
        I_M = (a_minus * a_plus) / (c * np.sqrt(denom))

    phi_vals = tau_minus + tau_plus

    return I_M, phi_vals


def extend_and_solve_alternative(x, y, dx, alpha, velocity):
    """Extend grid suitably and assemble integration factor and travel time function."""
    
    alpha_ind = math.ceil(alpha / dx[0])
    x_left_extension = []
    for i in range(1, alpha_ind + 1):
        x_left_extension.append(x[0] - i * dx[0])

    x_left = np.append(np.sort(np.array(x_left_extension)), x)

    x_right_extension = []
    for i in range(1, alpha_ind + 1):
        x_right_extension.append(x[-1] + i * dx[0])

    x_right = np.append(x, np.sort(np.array(x_right_extension)))

    # Generate extended meshgrid
    X, Y = np.meshgrid(x, y, indexing='ij')
    X_left, Y_left = np.meshgrid(x_left, y, indexing='ij')
    X_right, Y_right = np.meshgrid(x_right, y, indexing='ij')

    # Background velocity on extended mesh
    c_left = velocity(X_left, Y_left)
    c_right = velocity(X_right, Y_right)

    # Determine sources
    source_left = (np.argmin(abs(x_left + alpha)), np.where(y == 0)[0][0])
    source_right = (np.argmin(abs(x_right - alpha)), np.where(y == 0)[0][0])

    # Compute eikonal equation and slice
    tau_left = solve_eikonal(c_left, dx, source_left)   
    tau_right = solve_eikonal(c_right, dx, source_right) 

    left_ind = len(x_left) - len(x)
    right_ind = len(x_right) - len(x)

    tau_minus = tau_left[left_ind:]
    tau_plus = tau_right[:-right_ind]

    x_s = [x_left[source_left[0]], y[source_left[1]]]
    x_r = [x_right[source_right[0]], y[source_right[1]]]

    # Do same for transport equation
    a_left = solve_transport_equation(c_left, tau_left, X_left, Y_left, x_s, dx, np.zeros_like(X_left))
    a_right = solve_transport_equation(c_right, tau_right, X_right, Y_right, x_r, dx, np.zeros_like(X_right))

    a_minus = a_left[left_ind:]
    a_plus = a_right[:-right_ind]
 
    # Compute integrand factor I_M
    grad_tau_minus = np.gradient(tau_minus, *dx, edge_order=2) 
    grad_tau_plus = np.gradient(tau_plus, *dx, edge_order=2)
    dot_product = (grad_tau_minus[0] * grad_tau_plus[0] + grad_tau_minus[1] * grad_tau_plus[1])

    c = velocity(X, Y)

    denom = 1 + c**2 * dot_product
    
    # Check whether denominantor is positive (need to take square root)
    if np.any(denom < 0):
        print("Negative denominator: forward scattering or numerical instability.")
        denom = np.maximum(denom, 1e-12)  # Prevent division by zero
        I_M = (a_minus * a_plus) / (c * np.sqrt(denom))
        #I_M = (c * a_minus * a_plus) / (np.sqrt(denom))
        # Filter out problematic contour points
        valid_mask = (denom > 1e-10) & (np.abs(I_M) < 1e6)
        I_M[~valid_mask] = 0
    else:
        I_M = (a_minus * a_plus) / (c * np.sqrt(denom))

    phi_vals = tau_minus + tau_plus

    return I_M, phi_vals
