"""
Cutoff functions.
"""

import numpy as np

def phi(r, L_low, L_top, R_low, R_top):
    """Smooth cutoff function."""
    def f(r):
        if r > 0:
            return np.exp(-1/r)
        else:
            return 0
    
    def p(r, R_low, R_top):
        return f(r - R_low) / (f(r - R_low) + f(R_top - r))
    
    def q(r, R_low, R_top):
        return f(R_top - r) / (f(R_top - r) + f(r - R_low))
    
    if r <= L_low or r >= R_top:
        return 0
    elif r > L_low and r < L_top:
        return p(r, L_low, L_top)
    elif r > R_low and r < R_top:
        return q(r, R_low, R_top)
    elif r >= L_top and r <= R_low: 
        return 1

def psi(s, t, L_low_s, L_top_s, R_low_s, R_top_s, L_low_t, L_top_t, R_low_t, R_top_t):
    """2D cutoff function."""
    try:
        return phi(s, L_low_s, L_top_s, R_low_s, R_top_s) * phi(t, L_low_t, L_top_t, R_low_t, R_top_t)
    except:
        if phi(s, L_low_s, L_top_s, R_low_s, R_top_s) == 0 or phi(t, L_low_t, L_top_t, R_low_t, R_top_t) == 0:
            return 0
        elif phi(s, L_low_s, L_top_s, R_low_s, R_top_s) == None or phi(t, L_low_t, L_top_t, R_low_t, R_top_t) == None:
            return 0



def cutoff(vals, s_val, t_val):
    """Apply cutoff function to data."""
    L_low_s, L_top_s, R_low_s, R_top_s = s_val[0], s_val[0] + .5, s_val[-1] + .5, s_val[-1]
    L_low_t, L_top_t, R_low_t, R_top_t = t_val[0], t_val[0] + .5, t_val[-1] + .5, t_val[-1]
    psi_vals = vals
    for i in range(len(s_val)):
        s = s_val[i]
        #print(A)
        for j in range(len(t_val)):
            t = t_val[j]
            psi_vals[i, j] = psi(s, t, L_low_s, L_top_s, R_low_s, R_top_s, L_low_t, L_top_t, R_low_t, R_top_t) * vals[i, j]
    return psi_vals

