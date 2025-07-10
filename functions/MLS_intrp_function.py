from scipy.spatial.distance import euclidean
import numpy as np
import torch


class MLS_interpolation:
    """
    Class for performing Moving Least Squares (MLS) interpolation in 3D.
    """

    def __init__(self, x_src, y_src, z_src, x_dst, y_dst, z_dst):
        """
        Initialize the MLS_interpolation class with source and destination coordinates.

        Args:
            x_src (array-like): Source x-coordinates.
            y_src (array-like): Source y-coordinates.
            z_src (array-like): Source z-coordinates.
            x_dst (float): Destination x-coordinate.
            y_dst (float): Destination y-coordinate.
            z_dst (float): Destination z-coordinate.
        """
        self.x_src = x_src  # Store source x-coordinates
        self.y_src = y_src  # Store source y-coordinates
        self.z_src = z_src  # Store source z-coordinates
        self.x_dst = x_dst  # Store destination x-coordinate
        self.y_dst = y_dst  # Store destination y-coordinate
        self.z_dst = z_dst  # Store destination z-coordinate

    def gaussian_weight(self, distance, h):
        """
        Compute the Gaussian weight for a given distance and smoothing parameter h.

        Args:
            distance (float or array-like): Distance(s) between points.
            h (float): Smoothing parameter.

        Returns:
            np.ndarray: Gaussian weights.
        """
        return np.exp(-distance**2 / h**2)  # Calculate Gaussian weight

    def polynomial_basis(self, x):
        """
        Construct the polynomial basis vector for a given 3D point.

        Args:
            x (array-like): 3D point(s), shape (..., 3).

        Returns:
            np.ndarray: Polynomial basis vector(s).
        """
        # Return the polynomial basis vector for each point
        return np.array([
            x[0]**0,          # 1 (constant term)
            x[0],             # x
            x[1],             # y
            x[2],             # z
            x[0]**2,          # x^2
            x[1]**2,          # y^2
            x[2]**2,          # z^2
            x[0]*x[1],        # xy
            x[0]*x[2],        # xz
            x[1]*x[2]         # yz
        ]).T

    def calculate_distances(self):
        """
        Calculate Euclidean distances from each source point to the destination point.

        Returns:
            list: List of distances.
        """
        src_points = np.array(list(zip(self.x_src, self.y_src, self.z_src)))  # Stack source coordinates
        dst_point = np.array([self.x_dst, self.y_dst, self.z_dst])            # Destination point as array
        distances = [euclidean(src_point, dst_point) for src_point in src_points]  # Compute distances
        return distances

    def interpolation(self, dist=None):
        """
        Perform MLS interpolation to compute coefficients for the destination point.

        Args:
            dist (list or None): Optional precomputed distances. If None, distances are calculated.

        Returns:
            list: List of torch tensors representing interpolation coefficients.
        """
        if dist:
            distance_eucl = dist  # Use provided distances
        else:
            distance_eucl = self.calculate_distances()  # Calculate distances if not provided

        W = np.diag(self.gaussian_weight(abs(np.array(distance_eucl)), h=1.0))  # Weight matrix
        P = self.polynomial_basis(np.array([self.x_src, self.y_src, self.z_src]))  # Design matrix for source points
        a = self.polynomial_basis((self.x_dst, self.y_dst, self.z_dst))  # Basis vector for destination point

        b = np.linalg.pinv(np.dot(P.T, np.dot(W, P)))  # Pseudo-inverse of weighted design matrix
        c = np.dot(P.T, W)  # Weighted design matrix transpose

        coefficients = np.dot(np.dot(a, b), c)  # Compute coefficients
        coefficients = [torch.tensor(coefficient, dtype=torch.float32) for coefficient in coefficients]  # Convert to torch tensors

        return coefficients  # Return list of coefficients

