import numpy as np 
from scipy.spatial import cKDTree, distance
import torch
from functions import MLS_intrp_function


class Codec:
    """
    Codec class for dimensionality reduction and connectivity analysis using Euclidean and Mahalanobis distances.
    """

    def __init__(self, normalized_dataset, point_id_, coordinates):
        """
        Initialize the Codec object.

        Args:
            normalized_dataset (np.ndarray): The normalized dataset.
            point_id_ (np.ndarray): Array of point IDs.
            coordinates (np.ndarray): Array of coordinates with point IDs.
        """
        self.normalized_dataset = normalized_dataset  # Store the normalized dataset
        self.point_id_ = point_id_  # Store the point IDs
        self.coordinates = coordinates  # Store the coordinates

    def reduceSpace(self):
        """
        Reduce the space by associating each normalized data point with its coordinates.

        Returns:
            list: List of tuples containing (normalized_data, point_id, x, y, z)
        """
        reduced_space = []  # Initialize the reduced space list
        np.random.seed(0)  # Set random seed for reproducibility
        for i in range(self.normalized_dataset.shape[0]):  # Iterate over all data points
            # Find the coordinates corresponding to the current point ID
            coordinates_reduced = self.coordinates[np.isin(self.coordinates[:, -1], [self.point_id_[i]])][0]
            # Append the tuple to the reduced space
            reduced_space.append((self.normalized_dataset[i],
                                  self.point_id_[i],
                                  coordinates_reduced[0],
                                  coordinates_reduced[1],
                                  coordinates_reduced[2]))
        return reduced_space  # Return the reduced space

    def calc_euc_dist(self, src_space_coords, src_point_id, dest_space_coords, dest_point_id, num_neighbors=250):
        """
        Calculate the Euclidean distance between source and destination coordinates using a KD-tree.

        Args:
            src_space_coords (np.ndarray): Source coordinates.
            src_point_id (np.ndarray): Source point IDs.
            dest_space_coords (np.ndarray): Destination coordinates.
            dest_point_id (np.ndarray): Destination point IDs.
            num_neighbors (int): Number of nearest neighbors to consider.

        Returns:
            dict: Mapping from source point ID to list of nearest destination point IDs.
        """
        tree = cKDTree(dest_space_coords)  # Build KD-tree for destination coordinates
        points_search_space = {}  # Initialize the search space dictionary
        for i, points in enumerate(src_space_coords):  # Iterate over source coordinates
            _, indices = tree.query(points, k=num_neighbors)  # Find nearest neighbors
            # Map source point ID to destination point IDs
            points_search_space[src_point_id[i]] = [dest_point_id[j] for j in indices]
        return points_search_space  # Return the mapping

    def calc_Mahal_dist(self, dest_point_id, points_search_space, num_neigh_connectivity, return_conn_matrix=False):
        """
        Calculate Mahalanobis and Euclidean distances for connectivity.

        Args:
            dest_point_id (np.ndarray): Destination point IDs.
            points_search_space (dict): Mapping from point ID to neighbor IDs.
            num_neigh_connectivity (int): Number of neighbors to keep for connectivity.
            return_conn_matrix (bool): Whether to return the connectivity matrix.

        Returns:
            tuple: (connectivity dictionary, coordinate dictionary) or (connectivity matrix, coordinate dictionary)
        """
        # Build a dictionary mapping point ID to coordinates
        coordinate_dict = {int(coord[3]): (float(coord[0]), float(coord[1]), float(coord[2])) for coord in self.coordinates}
        fine_points = np.array([list(point) for point in coordinate_dict.values()])  # All coordinates as array
        cov_matrix = np.cov(fine_points, rowvar=False)  # Covariance matrix of coordinates
        cov_inv = np.linalg.pinv(cov_matrix)  # Pseudo-inverse of covariance matrix
        new_conn = {}  # Initialize connectivity dictionary
        for p in dest_point_id:
            new_conn[p] = []  # Initialize neighbor list

        for point_id_1 in new_conn:  # For each destination point
            x1, y1, z1 = coordinate_dict[point_id_1]  # Get coordinates
            for point_id_2 in points_search_space[point_id_1]:  # For each neighbor
                x2, y2, z2 = coordinate_dict[point_id_2]  # Get neighbor coordinates
                euc_dist = distance.euclidean((x1, y1, z1), (x2, y2, z2))  # Euclidean distance
                delta = np.array([x2 - x1, y2 - y1, z2 - z1])  # Difference vector
                m_dist = np.sqrt(np.dot(delta, np.dot(cov_inv, delta)))  # Mahalanobis distance
                new_conn[point_id_1].append((point_id_2, m_dist, euc_dist))  # Store neighbor info

        connectivity_data = []  # List to store connectivity rows
        for pt in new_conn:
            new_conn[pt].sort(key=lambda x: x[1])  # Sort neighbors by Mahalanobis distance
            new_conn[pt] = new_conn[pt][:num_neigh_connectivity]  # Keep only closest neighbors
            for tup in new_conn[pt]:
                row = np.array([pt, tup[0], tup[2]])  # Store (point, neighbor, Euclidean distance)
                connectivity_data.append(row)

        if not return_conn_matrix:
            return new_conn, coordinate_dict  # Return dictionary if matrix not requested

        connectivity = np.array(connectivity_data)  # Convert to NumPy array
        return connectivity, coordinate_dict  # Return matrix and coordinate dictionary

    def reduced_connectivity(self, mode):
        """
        Compute reduced connectivity using either encoder or decoder mode.

        Args:
            mode (str): Either 'encoder' or 'decoder'.

        Returns:
            tuple: (interpolation coefficients dictionary, point ID data dictionary)
        """
        if mode.lower() not in ['encoder', 'decoder']:
            raise ValueError('Mode parameter must be "encoder" or "decoder" ')  # Validate mode

        reduced_space1 = self.reduceSpace()  # Reduce the space
        reduced_space1 = np.array(reduced_space1)  # Convert to NumPy array
        point_id_reduced = reduced_space1[:, 1].astype(int)  # Extract reduced point IDs

        if mode.lower() == 'encoder':
            # Calculate search space from reduced to full coordinates
            encoded_points_search_space = self.calc_euc_dist(
                src_space_coords=reduced_space1[:, 2:5],
                src_point_id=point_id_reduced,
                dest_space_coords=self.coordinates[:, :3],
                dest_point_id=self.coordinates[:, -1].astype(int)
            )
            # Calculate connectivity using Mahalanobis distance
            full_connectivity, coordinate_dict = self.calc_Mahal_dist(
                dest_point_id=point_id_reduced,
                points_search_space=encoded_points_search_space,
                num_neigh_connectivity=12
            )
        if mode.lower() == 'decoder':
            # Calculate search space from full to reduced coordinates
            encoded_points_search_space = self.calc_euc_dist(
                src_space_coords=self.coordinates[:, :3],
                src_point_id=self.coordinates[:, -1].astype(int),
                dest_space_coords=reduced_space1[:, 2:5],
                dest_point_id=point_id_reduced
            )
            # Calculate connectivity using Mahalanobis distance
            full_connectivity, coordinate_dict = self.calc_Mahal_dist(
                dest_point_id=self.coordinates[:, -1].astype(int),
                points_search_space=encoded_points_search_space,
                num_neigh_connectivity=12
            )

        # Compute interpolation coefficients for each point
        intrp_coeffs = {}  # Dictionary for interpolation coefficients
        point_id_data = {}  # Dictionary for point ID data
        for key, values in full_connectivity.items():
            point_ids = [torch.tensor((value[0]), dtype=torch.int32) for value in values]  # Neighbor point IDs

            # Prepare coordinates for MLS interpolation
            x = [list(coordinate_dict[point_id[0]])[0] for point_id in values]
            y = [list(coordinate_dict[point_id[0]])[1] for point_id in values]
            z = [list(coordinate_dict[point_id[0]])[2] for point_id in values]

            # Perform Moving Least Squares interpolation
            MLS = MLS_intrp_function.MLS_interpolation(
                x_src=x,
                y_src=y,
                z_src=z,
                x_dst=list(coordinate_dict[key])[0],
                y_dst=list(coordinate_dict[key])[1],
                z_dst=list(coordinate_dict[key])[2]
            )
            intrp_coeffs[key] = MLS.interpolation()  # Store interpolation coefficients
            point_id_data[key] = point_ids  # Store neighbor point IDs

        return intrp_coeffs, point_id_data  # Return results
