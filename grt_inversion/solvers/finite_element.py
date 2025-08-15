"""
Finite element implementation using scikit-fem.
"""

import numpy as np
import matplotlib.pyplot as plt

try:
    from skfem import *
    from skfem.helpers import *
    from skfem.helpers import dot, grad
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import spsolve
    SKFEM_AVAILABLE = True
except ImportError:
    SKFEM_AVAILABLE = False
    print("scikit-fem not available. FEM solver will not work.")

import warnings
warnings.filterwarnings('ignore')


class TransportSolverSkFem:
    """
    Finite element solver for transport equation using scikit-fem.
    """

    def __init__(self, mesh, xs, c, regularization_eps=1e-3):
        """
        Initialize FEM transport solver.
        
        Args:
            mesh: scikit-fem mesh
            xs: Source point coordinates (x, y)
            c: Speed function values at mesh nodes
            regularization_eps: Regularization parameter for epsilon ball radius
        """
        if not SKFEM_AVAILABLE:
            raise ImportError("scikit-fem is required for FEM solver")
        
        self.mesh = mesh
        self.xs = np.array(xs)
        self.eps = regularization_eps

        self.tol = 1e-13
        
        # Create basis (P1 elements)
        self.basis = Basis(mesh, ElementTriP1())
        
        # Store speed function - ensure it's interpolated to the mesh
        if isinstance(c, np.ndarray) and c.shape == (self.basis.N,):
            self.c = c
        else:
            self.c = self.basis.interpolate(c)
        
    def solve_transport_bilinear(self, tau, laplace_tau):
    #def solve_transport_bilinear(self, tau, laplace_tau, grad_tau):
        """
        Solve transport equation using strongly imposed boundary conditions.
        
        The transport equation is:
        2 \nabla a \cdot \nabla \tau + a \Delta \tau = 0 in Omega
        
        with inflow boundary condition on epsilon ball boundary.
        """
        
        # Create interpolated functions for use in bilinear form
        tau_basis = self.basis.interpolate(tau)
        laplace_tau_basis = self.basis.interpolate(laplace_tau)

    
        # Define the bilinear form 
        @BilinearForm
        def transport_bilinear(u, v, w):
            return - 2.0 * u * dot(grad(tau_basis), grad(v)) - u * laplace_tau_basis * v
            
        # Assemble main system
        A = transport_bilinear.assemble(self.basis)
        b = np.zeros(self.basis.N)

        # Compute boundary integral:
        outer_facets = self.mesh.boundary_facets() 
        boundary_basis = FacetBasis(self.mesh, ElementTriP1(), facets=outer_facets)
        
        tau_boundary = boundary_basis.interpolate(tau) 

        @BilinearForm
        def boundary_bilinear(u, v, w):
            return 2 * u * dot(grad(tau_boundary), w.n) * v 

        A_boundary = boundary_bilinear.assemble(boundary_basis)
        A += A_boundary
        
        # Apply inner boundary conditions
        self._apply_boundary_conditions(A, b)
        
        # Apply strong conditions on epsilon ball
        self._apply_strong_conditions(A, b)

        # Solve the system
        try:
            a_solution = spsolve(A, b)
        except:
            # If direct solve fails, try with regularization
            print("Direct solve failed, adding regularization...")
            import scipy.sparse as sp
            A_reg = A + 1e-8 * sp.eye(A.shape[0])
            a_solution = spsolve(A_reg, b)
        
        return a_solution
    
    def find_outer_boundary_nodes(self):
        """
        Find nodes on the outer rectangular boundary (not on epsilon ball boundary).
        """
        x_coords = self.mesh.p[0, :]
        y_coords = self.mesh.p[1, :]
        
        # Find min/max coordinates of rectangular domain
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        
        tol = 1e-10
        boundary_nodes = []
        
        for i in range(len(x_coords)):
            x, y = x_coords[i], y_coords[i]
            
            # Check if node is on rectangular boundary
            if (abs(x - x_min) < tol or abs(x - x_max) < tol or 
                abs(y - y_min) < tol or abs(y - y_max) < tol) and \
                (abs(x - self.xs[0]) > self.eps or abs(y - self.xs[1]) > self.eps):
                boundary_nodes.append(i)
        
        return np.array(boundary_nodes)
    
    def find_outer_boundary_nodes_and_normals(self):
        """
        Improved boundary detection that works for rectangular meshes.
        """
        outer_boundary_nodes = self.find_outer_boundary_nodes()
        
        normals = {}
        
        # Get domain bounds
        x_coords = self.mesh.p[0, :]
        y_coords = self.mesh.p[1, :]
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        
        # Use adaptive tolerance based on actual grid spacing
        x_unique = np.unique(x_coords)
        y_unique = np.unique(y_coords)
        tol_x = 0.5 * (x_unique[1] - x_unique[0]) if len(x_unique) > 1 else 1e-10
        tol_y = 0.5 * (y_unique[1] - y_unique[0]) if len(y_unique) > 1 else 1e-10
        
        # Normals for outer boundary nodes
        for node in outer_boundary_nodes:
            x, y = self.mesh.p[:, node]
            
            normal = None
            
            # Check boundaries with appropriate tolerances
            on_left = abs(x - x_min) < tol_x
            on_right = abs(x - x_max) < tol_x  
            on_bottom = abs(y - y_min) < tol_y
            on_top = abs(y - y_max) < tol_y
            
            # Handle corners first (priority to corners)
            if on_left and on_bottom:
                normal = np.array([-1/np.sqrt(2), -1/np.sqrt(2)])  # Corner: SW
            elif on_right and on_bottom:
                normal = np.array([1/np.sqrt(2), -1/np.sqrt(2)])   # Corner: SE
            elif on_left and on_top:
                normal = np.array([-1/np.sqrt(2), 1/np.sqrt(2)])   # Corner: NW
            elif on_right and on_top:
                normal = np.array([1/np.sqrt(2), 1/np.sqrt(2)])    # Corner: NE
            # Then handle edges
            elif on_left:
                normal = np.array([-1, 0])
            elif on_right:
                normal = np.array([1, 0])
            elif on_bottom:
                normal = np.array([0, -1])
            elif on_top:
                normal = np.array([0, 1])
            
            if normal is not None:
                normals[node] = normal
            else:
                print(f"Warning: Could not determine normal for boundary node {node} at ({x:.6f}, {y:.6f})")
        
        return outer_boundary_nodes, normals

    def find_epsilon_and_inflow_boundary_nodes(self):
        """
        Improved epsilon boundary detection for rectangular meshes.
        """
        x_coords = self.mesh.p[0, :]
        y_coords = self.mesh.p[1, :]
        
        epsilon_boundary_nodes = []
        
        # Use actual grid spacing for tolerances
        x_unique = np.unique(x_coords)
        y_unique = np.unique(y_coords)
        
        if len(x_unique) > 1:
            dx = x_unique[1] - x_unique[0]
        else:
            dx = 1e-3
            
        if len(y_unique) > 1:
            dy = y_unique[1] - y_unique[0]
        else:
            dy = 1e-3
        
        tol_x = 1.1 * dx  
        tol_y = 1.1 * dy
        
        epsilon_nodes = self.find_epsilon_nodes()

        for i in range(len(x_coords)):
            x, y = x_coords[i], y_coords[i]
            
            dist_x = abs(x - self.xs[0])
            dist_y = abs(y - self.xs[1])
            
            # Check if node is on epsilon boundary (not interior)
            if (abs(dist_x - self.eps) < tol_x and dist_y <= self.eps + tol_y) or \
            (abs(dist_y - self.eps) < tol_y and dist_x <= self.eps + tol_x):
                if i not in epsilon_nodes:
                    epsilon_boundary_nodes.append(i)
        
        return np.array(epsilon_boundary_nodes)

    def find_epsilon_nodes(self):
        """
        Find nodes in the epsilon ball.
        """
        x_coords = self.mesh.p[0, :]
        y_coords = self.mesh.p[1, :]
        
        epsilon_nodes = []
        
        for i in range(len(x_coords)):
            x, y = x_coords[i], y_coords[i]

            # square around source
            dist_x = (x - self.xs[0])
            dist_y = (y - self.xs[1])
            # Check if node is in epsilon ball
            if abs(dist_x) <= self.eps + self.tol:
                if abs(dist_y) <= self.eps + self.tol:
                    epsilon_nodes.append(i)
            
        return np.array(epsilon_nodes)
    
    def _apply_boundary_conditions(self, A, b):
        """
        Apply boundary conditions:
        - Inflow boundary condition on epsilon ball boundary
        """
        epsilon_and_inflow_boundary_nodes = self.find_epsilon_and_inflow_boundary_nodes()
        
        # Apply inflow boundary condition on epsilon ball boundary
        for dof in epsilon_and_inflow_boundary_nodes:
            x, y = self.mesh.p[:, dof]
            dist = np.sqrt((x - self.xs[0])**2 + (y - self.xs[1])**2)
            dist = max(dist, self.eps)
            
            # Inflow boundary condition: a = sqrt(c) / (2*sqrt(2*pi)*sqrt(|x-xs|))
            bc_value = (1.0 / (2.0 * np.sqrt(2.0 * np.pi) * np.sqrt(dist))) * np.sqrt(self.c[dof])
            
            A[dof, :] = 0
            A[dof, dof] = 1
            b[dof] = bc_value
          
    def _apply_strong_conditions(self, A, b):
        epsilon_nodes = self.find_epsilon_nodes()
        
        # Get coordinates of epsilon nodes
        coords = self.mesh.p[:, epsilon_nodes]  # shape (2, N)

        # Compute distances from source to each node
        dists = np.sqrt((coords[0] - self.xs[0])**2 + (coords[1] - self.xs[1])**2)
        
        # Identify indices where distance is zero (i.e., exactly at the source)
        zero_idx = np.where(dists == 0)[0]  # array of indices
        
        # Replace zero distances with smallest non-zero value to avoid singularity
        non_zero_dists = dists[dists > 0]
        if len(non_zero_dists) == 0:
            raise ValueError("All distances are zero; cannot apply strong condition safely.")
        min_non_zero_dist = np.min(non_zero_dists)
        dists[zero_idx] = min_non_zero_dist

        # Compute all bc_values
        bc_values = np.zeros_like(dists)
        for i, dof in enumerate(epsilon_nodes):
            dist = dists[i]
            bc_values[i] = (1.0 / (2.0 * np.sqrt(2.0 * np.pi) * np.sqrt(dist))) * np.sqrt(self.c[dof])

        # Replace bc_value at zero_idx with max of the other values
        if len(zero_idx) > 0:
            other_values = np.delete(bc_values, zero_idx)
            max_bc_value = np.max(other_values)
            bc_values[zero_idx] = max_bc_value

        # Apply the strong condition to the system
        for i, dof in enumerate(epsilon_nodes):
            A[dof, :] = 0
            A[dof, dof] = 1
            b[dof] = bc_values[i]

        print(f"Applied strong condition to {len(epsilon_nodes)} nodes in epsilon ball")


    def plot_solution(self, tau, a_solution):
        """
        Plot the solutions using matplotlib with triangulation.
        """
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # Get mesh coordinates
        x_coords = self.mesh.p[0, :]
        y_coords = self.mesh.p[1, :]
        triangles = self.mesh.t.T  # Transpose for matplotlib
        
        # Create triangulation
        triang = plt.matplotlib.tri.Triangulation(x_coords, y_coords, triangles)
        
        # Plot tau (eikonal solution)
        im1 = axes[0].tripcolor(triang, tau)
        axes[0].set_title('Eikonal Solution τ(x)')
        axes[0].set_aspect('equal')
        plt.colorbar(im1, ax=axes[0])
        
        # Plot a (transport solution)
        im2 = axes[1].tripcolor(triang, a_solution)
        axes[1].set_title('Transport Solution a(x)')
        axes[1].set_aspect('equal')
        plt.colorbar(im2, ax=axes[1])
        
        # Plot boundary nodes
        outer_boundary_nodes = self.find_outer_boundary_nodes()
        inflow_boundary_nodes = self.find_epsilon_and_inflow_boundary_nodes()
        axes[2].triplot(x_coords, y_coords, triangles, 'k-', linewidth=0.3, alpha=0.3)
        if len(outer_boundary_nodes) > 0:
            axes[2].plot(x_coords[outer_boundary_nodes], y_coords[outer_boundary_nodes], 'bo', markersize=3, label=f'Outer boundary ({len(outer_boundary_nodes)})')
        if len(inflow_boundary_nodes) > 0:
            axes[2].plot(x_coords[inflow_boundary_nodes], y_coords[inflow_boundary_nodes], 'ro', markersize=4, label=f'Inflow boundary ({len(inflow_boundary_nodes)})')
        axes[2].plot(self.xs[0], self.xs[1], 'go', markersize=8, label='Source')
        
        # Draw epsilon ball
        epsilon_nodes = self.find_epsilon_nodes()
        circle3 = plt.Circle(self.xs, self.eps, fill=False, color='red', linestyle='--', linewidth=2)
        axes[2].add_patch(circle3)
        axes[2].set_title('Boundary Nodes')
        if len(epsilon_nodes) > 0:
            axes[2].plot(x_coords[epsilon_nodes], y_coords[epsilon_nodes], 'go', markersize=3, label=f'Epsilon nodes ({len(epsilon_nodes)})')
        axes[2].legend()
        axes[2].set_aspect('equal')
        
        # Plot mesh around epsilon ball
        axes[3].triplot(x_coords, y_coords, triangles, 'k-', linewidth=0.3, alpha=0.3)
        axes[3].plot(self.xs[0], self.xs[1], 'go', markersize=10, label='Source')
        
        # Draw epsilon ball
        circle4 = plt.Circle(self.xs, self.eps, fill=False, color='red', linestyle='--', linewidth=2)
        axes[3].add_patch(circle4)
        axes[3].set_title(f'Mesh around Epsilon Ball (radius = {self.eps:.3f})')
        axes[3].legend()
        axes[3].set_aspect('equal')
        axes[3].set_xlim(self.xs[0] - 3*self.eps, self.xs[0] + 3*self.eps)
        axes[3].set_ylim(self.xs[1] - 3*self.eps, self.xs[1] + 3*self.eps)
        
        plt.tight_layout()
        plt.show()

    def compute_error(self, a_solution, a_ref, fullmesh=True):
        """
        Compute L1, L2, and L_inf errors with option to exclude epsilon ball region.
        
        Parameters:
        a_solution: numerical solution values at mesh nodes
        a_ref: reference solution values at mesh nodes  
        fullmesh: if True, compute error on full mesh; if False, exclude epsilon ball
        
        Returns:
        tuple: (error_l2, error_l1, error_linf)
        """
        
        def triangle_area(vertices):
            """Compute area of triangle given 3 vertices."""
            if vertices.shape[0] != 3:
                raise ValueError(f"Expected 3 vertices, got {vertices.shape[0]}")
            if vertices.shape[1] != 2:
                raise ValueError(f"Expected 2D vertices, got shape {vertices.shape}")
            
            v1, v2, v3 = vertices
            return 0.5 * abs((v2[0] - v1[0]) * (v3[1] - v1[1]) - (v3[0] - v1[0]) * (v2[1] - v1[1]))

        def get_valid_triangles_and_nodes(mesh, epsilon_nodes):
            """
            Get triangles and nodes that don't intersect with epsilon ball.
            
            Returns:
            valid_triangles: array of triangle indices to include in error computation
            valid_nodes: array of node indices that appear in valid triangles
            node_mapping: dict mapping original node indices to compressed indices
            """
            epsilon_nodes_set = set(epsilon_nodes)
            
            # Find triangles that don't contain any epsilon nodes
            valid_triangles = []
            points = mesh.p.T  # Convert to (N x 2) format
            triangles = mesh.t.T  # Convert to (M x 3) format
            
            for tri_idx, triangle in enumerate(triangles):
                # Check if any vertex of this triangle is in epsilon ball
                if not any(node in epsilon_nodes_set for node in triangle):
                    valid_triangles.append(tri_idx)
            
            valid_triangles = np.array(valid_triangles)
            
            # Get all nodes that appear in valid triangles
            if len(valid_triangles) > 0:
                valid_nodes = np.unique(triangles[valid_triangles].flatten())
            else:
                valid_nodes = np.array([])
            
            # Create mapping from original node indices to compressed indices
            node_mapping = {node: i for i, node in enumerate(valid_nodes)}
            
            print(f"Full mesh: {len(triangles)} triangles, {mesh.p.shape[1]} nodes")
            print(f"Excluding epsilon ball: {len(valid_triangles)} triangles, {len(valid_nodes)} nodes")
            print(f"Excluded: {len(triangles) - len(valid_triangles)} triangles, {len(epsilon_nodes)} epsilon nodes")
            
            return valid_triangles, valid_nodes, node_mapping

        def compute_l2_error_fem(mesh, u_numerical, u_reference, valid_triangles=None):
            """Compute L2 error using FEM integration over triangles."""
            error_squared = 0.0
            
            points = mesh.p.T  # Convert to (N x 2) format
            triangles = mesh.t.T  # Convert to (M x 3) format
            
            # Use valid triangles if provided, otherwise use all triangles
            triangle_indices = valid_triangles if valid_triangles is not None else range(len(triangles))
            
            for tri_idx in triangle_indices:
                triangle = triangles[tri_idx]
                
                # Get triangle vertices
                vertices = points[triangle]  # Shape (3, 2)
                area = triangle_area(vertices)
                
                # Get solution values at triangle nodes
                u_num_tri = u_numerical[triangle]
                u_ref_tri = u_reference[triangle]
                
                # Compute difference at nodes
                diff = u_num_tri - u_ref_tri
                
                # Simple integration: average over triangle nodes weighted by area
                error_squared += area * np.mean(diff**2)
            
            return np.sqrt(error_squared)

        def compute_l1_error_fem(mesh, u_numerical, u_reference, valid_triangles=None):
            """Compute L1 error using FEM integration over triangles."""
            error_l1 = 0.0
            
            points = mesh.p.T  # Convert to (N x 2) format
            triangles = mesh.t.T  # Convert to (M x 3) format
            
            # Use valid triangles if provided, otherwise use all triangles
            triangle_indices = valid_triangles if valid_triangles is not None else range(len(triangles))
            
            for tri_idx in triangle_indices:
                triangle = triangles[tri_idx]
                
                # Get triangle vertices
                vertices = points[triangle]  # Shape (3, 2)
                area = triangle_area(vertices)
                
                # Get solution values at triangle nodes
                u_num_tri = u_numerical[triangle]
                u_ref_tri = u_reference[triangle]
                
                # Compute absolute difference at nodes
                diff = np.abs(u_num_tri - u_ref_tri)
                
                # Simple integration: average over triangle nodes weighted by area
                error_l1 += area * np.mean(diff)
            
            return error_l1

        def compute_linf_error_fem(mesh, u_numerical, u_reference, valid_nodes=None):
            """Compute L_inf error (maximum pointwise error) on specified nodes."""
            if valid_nodes is not None:
                # Only consider error at valid nodes (excluding epsilon ball)
                return np.max(np.abs(u_numerical[valid_nodes] - u_reference[valid_nodes]))
            else:
                # Consider all nodes
                return np.max(np.abs(u_numerical - u_reference))

        if fullmesh:
            # Compute error on full mesh
            error_l2 = compute_l2_error_fem(self.mesh, a_solution, a_ref)
            error_l1 = compute_l1_error_fem(self.mesh, a_solution, a_ref)
            error_linf = compute_linf_error_fem(self.mesh, a_solution, a_ref)
        else:
            # Compute error excluding epsilon ball
            epsilon_nodes = self.find_epsilon_nodes()
            valid_triangles, valid_nodes, node_mapping = get_valid_triangles_and_nodes(self.mesh, epsilon_nodes)
            
            if len(valid_triangles) == 0:
                print("Warning: No valid triangles found after excluding epsilon ball")
                return 0.0, 0.0, 0.0
            
            error_l2 = compute_l2_error_fem(self.mesh, a_solution, a_ref, valid_triangles)
            error_l1 = compute_l1_error_fem(self.mesh, a_solution, a_ref, valid_triangles)
            error_linf = compute_linf_error_fem(self.mesh, a_solution, a_ref, valid_nodes)

        return error_l2, error_l1, error_linf

    def get_valid_triangles_and_nodes(self, mesh, epsilon_nodes):
        """
        Get triangles and nodes that don't intersect with epsilon ball.
        
        Returns:
        valid_triangles: array of triangle indices to include in error computation
        valid_nodes: array of node indices that appear in valid triangles
        node_mapping: dict mapping original node indices to compressed indices
        """
        epsilon_nodes_set = set(epsilon_nodes)
        
        # Find triangles that don't contain any epsilon nodes
        valid_triangles = []
        points = mesh.p.T  # Convert to (N x 2) format
        triangles = mesh.t.T  # Convert to (M x 3) format
        
        for tri_idx, triangle in enumerate(triangles):
            # Check if any vertex of this triangle is in epsilon ball
            if not any(node in epsilon_nodes_set for node in triangle):
                valid_triangles.append(tri_idx)
        
        valid_triangles = np.array(valid_triangles)
        
        # Get all nodes that appear in valid triangles
        if len(valid_triangles) > 0:
            valid_nodes = np.unique(triangles[valid_triangles].flatten())
        else:
            valid_nodes = np.array([])
        
        # Create mapping from original node indices to compressed indices
        node_mapping = {node: i for i, node in enumerate(valid_nodes)}
        
        print(f"Full mesh: {len(triangles)} triangles, {mesh.p.shape[1]} nodes")
        print(f"Excluding epsilon ball: {len(valid_triangles)} triangles, {len(valid_nodes)} nodes")
        print(f"Excluded: {len(triangles) - len(valid_triangles)} triangles, {len(epsilon_nodes)} epsilon nodes")
            
        return valid_triangles, valid_nodes, node_mapping

    def visualize_mesh_exclusion(self):
        """
        Visualize which parts of the mesh are excluded when computing error without epsilon ball.
        """
        import matplotlib.pyplot as plt
        
        # Get mesh data
        x_coords = self.mesh.p[0, :]
        y_coords = self.mesh.p[1, :]
        triangles = self.mesh.t.T  # Transpose for matplotlib
        
        # Find epsilon nodes and valid triangles
        epsilon_nodes = self.find_epsilon_nodes()
        valid_triangles, valid_nodes, _ = self.get_valid_triangles_and_nodes(self.mesh, epsilon_nodes)
        
        # Create masks
        all_triangles = np.arange(len(triangles))
        excluded_triangles = np.setdiff1d(all_triangles, valid_triangles)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Show included vs excluded triangles
        triang = plt.matplotlib.tri.Triangulation(x_coords, y_coords, triangles)
        
        # Plot all triangles in light gray
        axes[0].triplot(triang, color='lightgray', linewidth=0.5, alpha=0.3)
        
        # Plot valid triangles in blue
        if len(valid_triangles) > 0:
            valid_triang = plt.matplotlib.tri.Triangulation(x_coords, y_coords, triangles[valid_triangles])
            axes[0].triplot(valid_triang, color='blue', linewidth=0.8, alpha=0.7)
        
        # Plot excluded triangles in red
        if len(excluded_triangles) > 0:
            excluded_triang = plt.matplotlib.tri.Triangulation(x_coords, y_coords, triangles[excluded_triangles])
            axes[0].triplot(excluded_triang, color='red', linewidth=0.8, alpha=0.7)
        
        # Mark epsilon nodes
        axes[0].plot(x_coords[epsilon_nodes], y_coords[epsilon_nodes], 'ro', markersize=4, label='Epsilon nodes')
        
        # Mark source and epsilon ball
        axes[0].plot(self.xs[0], self.xs[1], 'go', markersize=10, label='Source')
        circle = plt.Circle(self.xs, self.eps, fill=False, color='red', linestyle='--', linewidth=2)
        axes[0].add_patch(circle)
        
        axes[0].set_title('Mesh Exclusion for Error Computation')
        axes[0].legend(['Valid triangles', 'Excluded triangles', 'Epsilon nodes', 'Source'])
        axes[0].set_aspect('equal')
        
        # Plot 2: Zoom around epsilon ball
        axes[1].triplot(triang, color='lightgray', linewidth=0.5, alpha=0.3)
        if len(valid_triangles) > 0:
            axes[1].triplot(valid_triang, color='blue', linewidth=0.8, alpha=0.7)
        if len(excluded_triangles) > 0:
            axes[1].triplot(excluded_triang, color='red', linewidth=0.8, alpha=0.7)
        
        axes[1].plot(x_coords[epsilon_nodes], y_coords[epsilon_nodes], 'ro', markersize=6)
        axes[1].plot(self.xs[0], self.xs[1], 'go', markersize=12)
        circle2 = plt.Circle(self.xs, self.eps, fill=False, color='red', linestyle='--', linewidth=3)
        axes[1].add_patch(circle2)
        
        axes[1].set_title(f'Zoom: Epsilon Ball (radius = {self.eps:.3f})')
        axes[1].set_xlim(self.xs[0] - 3*self.eps, self.xs[0] + 3*self.eps)
        axes[1].set_ylim(self.xs[1] - 3*self.eps, self.xs[1] + 3*self.eps)
        axes[1].set_aspect('equal')
        
        plt.tight_layout()
        plt.show()
        
        print(f"Total triangles: {len(triangles)}")
        print(f"Valid triangles (for error computation): {len(valid_triangles)}")
        print(f"Excluded triangles: {len(excluded_triangles)}")
        print(f"Epsilon nodes: {len(epsilon_nodes)}")
        print(f"Valid nodes: {len(valid_nodes)}")


def create_rectangular_mesh_ij_consistent(x, y):
    """
    Create a rectangular mesh consistent with meshgrid(indexing='ij').
    With 'ij' indexing: X[i,j] corresponds to (x[i], y[j])
    """
    from skfem import MeshTri
    import numpy as np
    
    nx = len(x)
    ny = len(y)
    
    print(f"Creating mesh: nx={nx}, ny={ny}")
    
    # Create points consistent with 'ij' indexing
    # For meshgrid with 'ij': X[i,j] = x[i], Y[i,j] = y[j]
    points = []
    
    # Create points: for each i (x-index), for each j (y-index)
    for i in range(nx):
        for j in range(ny):
            points.append([x[i], y[j]])
    
    points = np.array(points).T
    print(f"Created {points.shape[1]} points")
    
    # Create triangles 
    triangles = []
    for i in range(nx - 1):  
        for j in range(ny - 1):  
            # Node indices for the quad at (i,j)
            p1 = i * ny + j          
            p2 = i * ny + (j + 1)   
            p3 = (i + 1) * ny + j    
            p4 = (i + 1) * ny + (j + 1)  
            
            # Two triangles per quad
            triangles.append([p1, p2, p3])  # Lower triangle
            triangles.append([p2, p4, p3])  # Upper triangle
    
    triangles = np.array(triangles).T
    print(f"Created {triangles.shape[1]} triangles")
    
    mesh = MeshTri(points, triangles)
    return mesh, points


def meshgrid_to_triangulation_ij(x, y, f_values, mesh):
    """
    Interpolate function values from meshgrid to triangulation mesh points.
    Consistent with meshgrid(indexing='ij') where f_values[i,j] corresponds to (x[i], y[j]).
    """
    from scipy.interpolate import RegularGridInterpolator
    
    # With 'ij' indexing, f_values[i,j] corresponds to (x[i], y[j])
    # So f_values has shape (nx, ny) and f_values[i,j] is the value at (x[i], y[j])
    interp = RegularGridInterpolator((x, y), f_values,
                                   method='cubic',
                                   bounds_error=False,
                                   fill_value=0.0)
    
    # Get mesh points
    mesh_points = mesh.p
    points_to_interp = np.column_stack((mesh_points[0, :], mesh_points[1, :]))
    
    # Interpolate
    f_interp = interp(points_to_interp)
    
    return f_interp


def triangulation_to_meshgrid_ij(mesh, f_values, nx, ny):
    """
    Convert triangulation solution back to meshgrid format.
    Returns array consistent with meshgrid(indexing='ij') where result[i,j] corresponds to (x[i], y[j]).
    """
    # Get mesh coordinates
    x_coords = mesh.p[0, :]
    y_coords = mesh.p[1, :]
    
    # Create result array
    result = np.zeros((nx, ny))
    
    # For mesh created consistent with 'ij' indexing:
    # Points were created as: for i in range(nx): for j in range(ny)
    # So node_idx = i * ny + j corresponds to grid position (i, j)
    
    for node_idx in range(len(x_coords)):
        # Extract i, j from node index
        i = node_idx // ny 
        j = node_idx % ny  
        
        if i < nx and j < ny:
            result[i, j] = f_values[node_idx]
    
    return result

def laplacian_2d(f, dx, dy, boundary='extrapolate'):
    """
    Compute 2D Laplacian using finite differences.
    """
    nx, ny = f.shape
    laplacian = np.zeros_like(f)
    
    # Interior points
    laplacian[1:-1, 1:-1] = (
        (f[2:, 1:-1] - 2*f[1:-1, 1:-1] + f[:-2, 1:-1]) / dx**2 +
        (f[1:-1, 2:] - 2*f[1:-1, 1:-1] + f[1:-1, :-2]) / dy**2
    )
    
    # Handle boundaries based on boundary condition
    if boundary == 'extrapolate':
        # Simple extrapolation for boundaries
        laplacian[0, :] = laplacian[1, :]
        laplacian[-1, :] = laplacian[-2, :]
        laplacian[:, 0] = laplacian[:, 1]
        laplacian[:, -1] = laplacian[:, -2]
    elif boundary == 'zero':
        # Zero boundary conditions
        laplacian[0, :] = 0
        laplacian[-1, :] = 0
        laplacian[:, 0] = 0
        laplacian[:, -1] = 0
    
    return laplacian


def solve_transport_equation_fem(c, tau_grid, X, Y, x_s, dx, f):
    """
    Main function to run the transport equation solver with epsilon ball exclusion.
    """
    
    x = X[:, 0]
    y = Y[0, :]
    
    # Epsilon radius (should be larger than grid spacing)
    # dx = (abs(x[1] - x[0]), abs(y[1] - y[0]))
    eps_radius = max(dx) * 1
    
    # Create mesh with consistent indexing
    mesh, points = create_rectangular_mesh_ij_consistent(x, y)
    
    # Interpolate speed function to mesh
    c_interpolated = meshgrid_to_triangulation_ij(x, y, c, mesh)
    
    # Initialize solver
    solver = TransportSolverSkFem(mesh, x_s, c_interpolated, regularization_eps=eps_radius)
    
    # Solve eikonal equation on regular grid
    print("Solving eikonal equation...")
    
    laplace_tau_grid = laplacian_2d(tau_grid, dx[0], dx[1], 'extrapolate') # Doesn't work well with laplace_2d which extends original 
    # array to compute second order finite difference approximation of laplacian; only occurs for grid with nx not = ny

    # Interpolate to mesh using consistent functions
    tau = meshgrid_to_triangulation_ij(x, y, tau_grid, mesh)
    laplace_tau = meshgrid_to_triangulation_ij(x, y, laplace_tau_grid, mesh)
    
    # Solve transport equation
    print("Solving transport equation...")
    a_solution = solver.solve_transport_bilinear(tau, laplace_tau)
    # a_solution = solver.solve_transport_bilinear(tau, laplace_tau, grad_tau)
    
    # Plot results
    print("Plotting results...")
    solver.plot_solution(tau, a_solution)

    # Convert back to grid for comparison
    nx, ny = X.shape
    a_solution_grid = triangulation_to_meshgrid_ij(mesh, a_solution, nx, ny)

    return a_solution, a_solution_grid


    