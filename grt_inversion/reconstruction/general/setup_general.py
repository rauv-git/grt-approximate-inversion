"""
Compute all solutions we need to handle the general case.
"""

import numpy as np
from scipy.interpolate import RectBivariateSpline

from ...core.eikonal import solve_eikonal
from ...core.transport import solve_transport_equation


def compute_tau_and_a(s, x, y, velocity):
    """
    Compute all eikonal and transport equations required to compute GRT in general setting.
    
    Parameters:
        - s : 1D array of source locations
        - x, y : 1D arrays
        - velocity : function

    Returns: 
        - tau_vals, a_vals : solutions for eikonal and transport equation
        - c : velocity field
    """
    try:
        from joblib import Parallel, delayed
    except ImportError:
        print("joblib not installed. Install with: pip install joblib")
        return None
    
    dx = np.array([abs(x[1] - x[0]), abs(y[1] - y[0])])

    X, Y = np.meshgrid(x, y, indexing='ij')
    c = velocity(X, Y)

    #dx_extended = ((x[-1] - x[0])/(len(s)), (y[-1] - y[0])/(len(y)))

    # x_extended = np.linspace(x[0], x[-1], len(s))
    x_extended = s #np.linspace(s[0], s[-1], len(s))
    y_extended = y #np.linspace(y[0], y[-1], len(y))
    dx_extended = (x_extended[1] - x_extended[0], y_extended[1] - y_extended[0])

    X_extended, Y_extended = np.meshgrid(x_extended, y_extended, indexing='ij')
    c_extended = velocity(X_extended, Y_extended)

    def process_single_s(i):
        """compute tau and a for given source"""
        #print(f"Processing s = {s}")

        #source = (i, np.argmin(abs(y)))
        source = (i, 0)
        # solve eikonal eq on suitably fine mesh and then project onto coarser mesh
        tau_s = solve_eikonal(c_extended, dx_extended, source, factored=True)

        tau_s_interp = RectBivariateSpline(x_extended, y_extended, tau_s, kx=1, ky=1)
        tau_s_projected = tau_s_interp.ev(X, Y)

        x_s = [s[source[0]], 0]# y[source[1]]]
        
        #a_s = solve_transport_equation(c, tau_s_projected, X, Y, x_s, dx, np.zeros_like(X))
        
        a_s_extended = solve_transport_equation(c_extended, tau_s, X_extended, Y_extended, x_s, dx_extended, np.zeros_like(X_extended))

        a_s_interp = RectBivariateSpline(x_extended, y_extended, a_s_extended, kx=1, ky=1)
        a_s = a_s_interp.ev(X, Y)

        return i, tau_s_projected, a_s
        
    
    # Parallel execution with joblib
    n_jobs = -1 #min(cpu_count(), len(x) - 2)
    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(process_single_s)(i)#, s[i]) 
        for i in range(len(s))
    )
    
    tau_vals = np.zeros((len(s), len(x), len(y)))
    a_vals = np.zeros((len(s), len(x), len(y)))
    # Collect results
    for i, tau, a in results:
        tau_vals[i, :, :] = tau
        a_vals[i, :, :] = a
    
    return tau_vals, a_vals, c






