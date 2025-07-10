import numpy as np 
import pandas as pd
from scipy.spatial import cKDTree, distance

# The Euclidean distance selects 50 points from the coarse mesh to calculate Mahalanobis distances using the covariance matrix of the fine mesh.

class WriteConnectivity:
    def __init__(self, normalized_dataset, point_id_, coordinates, num_neighbors=50, num_connection_per_node=6):
        """
        Initialize the WriteConnectivity class.

        Args:
            normalized_dataset (np.ndarray): Normalized dataset for the mesh.
            point_id_ (np.ndarray): Array of point IDs.
            coordinates (np.ndarray): Array of coordinates with point IDs.
            num_neighbors (int): Number of nearest neighbors to consider (default: 50).
            num_connection_per_node (int): Number of connections per node (default: 6).
        """
        self.normalized_dataset = normalized_dataset
        self.point_id_ = point_id_
        self.coordinates = coordinates
        self.num_neighbors = num_neighbors
        self.num_connection_per_node = num_connection_per_node

    def reduceSpace(self):
        """
        Reduce the space by associating each normalized dataset entry with its coordinates and point ID.

        Returns:
            list: List of tuples containing (normalized_data, point_id, x, y, z).
        """
        reduced_space = []
        np.random.seed(0)  # Set random seed for reproducibility
        for i in range(self.normalized_dataset.shape[0]):
            # Find the coordinates for the current point ID
            coordinates_reduced = self.coordinates[np.isin(self.coordinates[:, -1], [self.point_id_[i]])][0]
            reduced_space.append((
                self.normalized_dataset[i],
                self.point_id_[i],
                coordinates_reduced[0],
                coordinates_reduced[1],
                coordinates_reduced[2]
            ))
        return reduced_space

    def reduced_connectivity(self):
        """
        Compute the reduced connectivity using Mahalanobis distance for the nearest neighbors.

        Returns:
            pd.DataFrame: DataFrame containing point_i, point_j, and normalized inverse Euclidean distance.
        """
        reduced_space1 = self.reduceSpace()  # Reduce the space to relevant points
        reduced_space1 = np.array(reduced_space1)  # Convert to numpy array

        coarse_mesh_points = reduced_space1[:, 2:5]  # Extract coordinates
        point_id_reduced = reduced_space1[:, 1].astype(int)  # Extract point IDs

        # Initialize a KD-tree for efficient neighbor search
        tree = cKDTree(coarse_mesh_points)

        coarse_points_search_space = {}
        # For each point, find the nearest neighbors in the mesh
        for i, coarse_point in enumerate(coarse_mesh_points):
            _, indices = tree.query(coarse_point, k=self.num_neighbors)  # Find nearest neighbors
            coarse_points_search_space[point_id_reduced[i]] = [point_id_reduced[j] for j in indices]

        # Create a dictionary mapping point IDs to their coordinates
        coordinate_dict = {int(coord[3]): (float(coord[0]), float(coord[1]), float(coord[2])) for coord in self.coordinates}
        reduced_space1 = np.array(reduced_space1)

        # Convert the dictionary values to a numpy array for covariance calculation
        fine_points = np.array([list(point) for point in coordinate_dict.values()])

        # Calculate the covariance matrix of the fine mesh points
        cov_matrix = np.cov(fine_points, rowvar=False)

        # Compute the pseudo-inverse of the covariance matrix
        cov_inv = np.linalg.pinv(cov_matrix)

        new_conn = {}
        for p in point_id_reduced:
            new_conn[p] = []

        # For each point, compute Mahalanobis and Euclidean distances to its neighbors
        for point_id_1 in new_conn:
            x1, y1, z1 = coordinate_dict[point_id_1]
            for point_id_2 in coarse_points_search_space[point_id_1]:
                if point_id_1 != point_id_2:
                    x2, y2, z2 = coordinate_dict[point_id_2]
                    euc_dist = distance.euclidean((x1, y1, z1), (x2, y2, z2))  # Euclidean distance
                    delta = np.array([x2 - x1, y2 - y1, z2 - z1])
                    m_dist = np.sqrt(np.dot(delta, np.dot(cov_inv, delta)))  # Mahalanobis distance
                    new_conn[point_id_1].append((point_id_2, m_dist, euc_dist))

        # Prepare the reduced connectivity data
        reduced_connectivity_data = []

        for pt in new_conn:
            # Sort neighbors by Mahalanobis distance and keep the closest ones
            new_conn[pt].sort(key=lambda x: x[1])
            new_conn[pt] = new_conn[pt][:self.num_connection_per_node]
            for tup in new_conn[pt]:
                row = np.array([pt, tup[0], tup[2]])  # Store point_i, point_j, and Euclidean distance
                reduced_connectivity_data.append(row)

        # Convert the list of rows into a NumPy array
        reduced_connectivity = np.array(reduced_connectivity_data)
        # Find the maximum inverse distance for normalization
        max_distance = np.max(1.0 / reduced_connectivity[:, 2])

        # Normalize the inverse distances between 0 and 1
        normalized_distances = reduced_connectivity.copy()
        inv_dist = 1.0 / normalized_distances[:, 2]  # Inverse of distances
        normalized_distances[:, 2] = inv_dist / max_distance  # Normalize

        # Create a DataFrame for the connectivity
        df = pd.DataFrame(normalized_distances, columns=['point_i', 'point_j', 'distance'])

        # Ensure that 'point_i' and 'point_j' are integer columns
        df['point_i'] = df['point_i'].astype(int)
        df['point_j'] = df['point_j'].astype(int)

        return df
