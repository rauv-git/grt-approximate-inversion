"""
Functions to use for the general background velocity case, i.e., we do not assume the background velcity to be layered here.
In particular, we need to accomodate for the fact that none of the equations, isochrones, etc. are translationally invariant
in this setting. Hence, we need to compute the solution for tau and a for all possible source and receiver locations.
"""

from .data_generation import generate_data_general
from .setup_general import compute_tau_and_a
from .kernels import compute_kernels_parallel_batched_general

__all__ = [
    'generate_data_general',
    'compute_tau_and_a',
    'compute_kernels_parallel_batched_general'
]