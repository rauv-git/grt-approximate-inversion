"""
Mesh generation example and velocity field definitions.
"""

import numpy as np
import matplotlib.pyplot as plt

def mesh_example():
    """Generate example rectangular mesh."""
    # Define the domain and resolution
    x_min, x_max, nx = -5.0, 5.0, 201  # horizontal
    y_min, y_max, ny = 0.0, 5.0, 101   # depth

    # Create 1D coordinate arrays
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)

    # Create 2D meshgrid: X has x values, Y has y values
    X, Y = np.meshgrid(x, y, indexing='ij')  # (nx, ny) shape

    # Flatten for vectorized operations if needed
    points = np.stack([X.ravel(), Y.ravel()], axis=-1)  # shape (nx*ny, 2)

    return X, Y


def layered_velocity(x, y, m=0.1, b=0.5):
    """Generate layered background velocity model."""
    return m * y + b

def bilinear_velocity(x, y, mx=0.01, my=0.1, b=0.5):
    """Generate laterally varying bilinear velocity model."""
    return mx*x + my*y + b

def one_velocity(x, y):
    """Generate constant background velocity model."""
    return np.ones_like(x)

def velocity_example():
    """Plot example velocity field."""
    X, Y = mesh_example()
    # visualize velocity field
    #velocity = layered_velocity(X, Y)
    velocity = bilinear_velocity(X, Y)

    plt.figure(figsize=(6, 5))
    plt.pcolormesh(X, Y, velocity, shading='auto', cmap='viridis')
    plt.colorbar(label='Velocity')
    plt.xlabel('x')
    plt.ylabel('y (depth)')
    #plt.title('Velocity Field')
    plt.show()

#velocity_example()