"""
Fast Marching Method implementation for solving the eikonal equation using eikonalfm.
"""

import numpy as np
import matplotlib.pyplot as plt


def solve_eikonal(c, dx, source, factored=True) -> np.ndarray:
    """
    Main interface for solving the eikonal equation.
    
    Parameters:
    -----------
    c : ndarray
        Speed field
    dx : array-like
        Grid spacing
    source : tuple
        Source point indices
    factored : bool
        Whether to use factored fast marching
        
    Returns:
    --------
    tau : ndarray
        Travel time field
    """
    try: 
        import eikonalfm

        if factored: # Solve factored fast marching using eikonalfm
            tau1_ffm = eikonalfm.factored_fast_marching(c, source, dx, order=2)
            tau_ffm = eikonalfm.distance(tau1_ffm.shape, dx, source, indexing="ij") * tau1_ffm
            return tau_ffm
        else:
            tau_fm = eikonalfm.fast_marching(c, source, dx, order=2)
            return tau_fm
    
    except ImportError:
        raise ImportError(
            "eikonalfm not installed. Install with:\n"
            "pip install git+https://github.com/kevinganster/eikonalfm.git"
        )




# Testing and convergence plot functions

def test():
    """Test the eikonal solver with a simple example."""
    # Run example
    x_min, x_max, nx = -5.0, 5.0, 201  # horizontal
    y_min, y_max, ny = 0.0, 10.0, 201   # depth

    # Create 1D coordinate arrays
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)

    # Create 2D meshgrid: X has x values, Y has y values
    X, Y = np.meshgrid(x, y, indexing='ij')  # (nx, ny) shape

    dx = np.array([x[1]-x[0], y[1]-y[0]])
    
    source = (nx // 2, ny // 2 )
    
    c = np.ones_like(X)

    tau_result = solve_eikonal(c, dx, source)

    test = np.sqrt((X - X[source])**2 + (Y - Y[source])**2)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    # Fast Marching
    pcm0 = axes[0].pcolormesh(X, Y, tau_result, cmap='inferno', shading='auto')
    axes[0].set_title("Fast Marching Travel Times")
    fig.colorbar(pcm0, ax=axes[0], label='Time')

    # True Solution
    pcm1 = axes[1].pcolormesh(X, Y, test, cmap='inferno', shading='auto')
    axes[1].set_title("True Solution")
    fig.colorbar(pcm1, ax=axes[1], label='Time')

    # Error Map
    error = np.abs(tau_result - test)
    pcm2 = axes[2].pcolormesh(X, Y, error, cmap='viridis', shading='auto')
    axes[2].set_title("Absolute Error")
    fig.colorbar(pcm2, ax=axes[2], label='|Error|')

    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    plt.show()

# TODO: tidy up plots
def convergence_test_eikonal():
    """Test convergence of eikonal solver."""
    meshsize_values = [2**k + 1 for k in range(3,12)]
    #print(meshsize_values)

    errors = []
    h_list = []

    for meshsize in meshsize_values:
        # Create 1D coordinate arrays
        x = y = np.linspace(-1, 1, meshsize)

        # Create 2D meshgrid: X has x values, Y has y values
        X, Y = np.meshgrid(x, y, indexing='ij')

        source = (meshsize // 2, meshsize // 2 )
        c = 1 / np.sqrt((X - X[source])**2 + (Y - Y[source])**2)
        dx = np.array([x[1]-x[0], y[1]-y[0]])
        h = dx[0]

        tau_result = solve_eikonal(c, [h, h], source)

        test = 1/2 * ((X - X[source])**2 + (Y - Y[source])**2)
        #err = np.sum(np.abs(tau_result - test).flatten())
        err = np.linalg.norm(tau_result - test)
        print(err, h)
        #errors.append(np.linalg.norm(tau_result - test))
        errors.append(err)
        h_list.append(h)

    plt.loglog(h_list, np.array(errors) * np.array(h_list)*1, label=f"$\log$ of errors in $L^2$-Norm")
    plt.loglog(h_list, np.array(h_list)**2, label=f"$2$-nd order reference")
    plt.xlabel(f"$h$")
    plt.ylabel('error')
    plt.legend(loc='best')
    plt.show()  
