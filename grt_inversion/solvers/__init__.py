"""
Finite difference and finite elemnt method solver implementations for the transport equation.
"""

from .finite_difference import solve_transport_equation_fd
from .finite_element import solve_transport_equation_fem

__all__ = ['solve_transport_equation_fd', 'solve_transport_equation_fem']

