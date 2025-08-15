"""
Transport equation solver. Choice between fast sweeping (finite difference) and finite element method.
"""

from ..solvers.finite_difference import solve_transport_equation_fd
from ..solvers.finite_element import solve_transport_equation_fem

def solve_transport_equation(c, tau, X, Y, x_s, h, f, method='fd'):
    """Compute solution of transport equation using Fast sweeping or FEM."""

    if method == None:
        method = 'fd' # Use fast sweeping as standard solver if not specified otherwise
    
    if method == 'fd':
        a, a_0, a_1 = solve_transport_equation_fd(c, tau, X, Y, x_s, h, f)
        return a
    else: # 'fem'
        a_solution, a_solution_grid = solve_transport_equation_fem(c, tau, X, Y, x_s, h, f)
        return a_solution_grid
