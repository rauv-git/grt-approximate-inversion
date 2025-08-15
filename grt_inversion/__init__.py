
# grt_inversion/__init__.py
"""
Generalized Radon Transform (GRT) Approximate Inversion Package

This package implements the methodology described in Ganster & Rieder
for the approximate inversion of Generalized Radon Transforms with 
layered background velocities.
"""

# Import key functions to make them available at package level

from .core import layered_velocity, bilinear_velocity, one_velocity, solve_eikonal, solve_transport_equation
from .reconstruction.translation_invariant import extend_and_solve, generate_data
from .reconstruction.general import generate_data_general, compute_tau_and_a
from .utils import plot, cutoff

# Optionally define what gets imported with "from grt_inversion import *"
__all__ = [
    'layered_velocity',
    'bilinear_velocity',
    'one_velocity',
    'solve_eikonal',
    'solve_transport_equation',
    'extend_and_solve',
    'generate_data',
    'generate_data_general',
    'compute_tau_and_a',
    'plot',
    'cutoff'
]

