"""
Plot data on a meshgrid.
"""

import numpy as np
import matplotlib.pyplot as plt

def plot(x, y, data):
    """
    Plot data on a meshgrid.
    """
    X, Y = np.meshgrid(x, y, indexing='ij')
    #plt.pcolormesh(S_data, T_data, g_cutoff.toarray())
    plt.pcolormesh(X, Y, data)
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.show()