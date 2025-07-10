import numpy as np
import pandas as pd
import torch
from functions.Reduced_space_PDF_function import get_reduced_space
from functions.Codec import Codec
from functions.Connectivity import WriteConnectivity

class Space:
    """
    Class for handling dimensionality reduction, connectivity, and interpolation matrices
    for a given set of points and their associated data.
    """

    def create(self, npoints, magnitude, coordinates, point_id, edge_indices, edge_distances, space2=False):
        """
        Perform dimensionality reduction and prepare all necessary data structures
        for further processing, including connectivity and interpolation matrices.

        Args:
            npoints (int): Number of points to select in reduced space.
            magnitude (np.ndarray): Data associated with each point.
            coordinates (np.ndarray): Coordinates of each point.
            point_id (np.ndarray): Unique identifiers for each point.
            edge_indices (pd.DataFrame): DataFrame with 'point_i' and 'point_j' columns for edges.
            edge_distances (np.ndarray): Distances between edges.
            space2 (bool): Optional flag for alternate normalization.
        """
        self.point_id_orig = point_id  # Store original point IDs

        # Dimensionality reduction: get reduced magnitude and coordinates
        magnitude_reduced, coordinates_reduced = get_reduced_space(
            magnitude, coordinates, point_id, p1=0.2, pn=1.0, num_selected=npoints
        )
        point_id_reduced = coordinates_reduced[:, -1].astype(int)  # Extract reduced point IDs

        # Normalize dataset for reduced space
        if not space2:
            normalized_dataset = magnitude_reduced[:, 0]  # Use first column if not space2
        else:
            normalized_dataset = np.array([el[0] for el in magnitude_reduced[:, 0]])  # Alternate normalization

        # Create Codec object for encoding/decoding
        reduced_space = Codec(normalized_dataset, point_id_reduced, coordinates)
        intrp_coeffs_encoder, point_id_data_encoder = reduced_space.reduced_connectivity(mode='encoder')  # Encoder interpolation
        intrp_coeffs_decoder, point_id_data_decoder = reduced_space.reduced_connectivity(mode='decoder')  # Decoder interpolation

        # Write connectivity matrix using WriteConnectivity
        reduced_space = WriteConnectivity(normalized_dataset, point_id_reduced, coordinates)
        reduced_conn = reduced_space.reduced_connectivity()  # Get reduced connectivity DataFrame

        self.magnitude_reduced = magnitude_reduced  # Store reduced magnitude
        coordinates_reduced[:, -1] = np.argsort(coordinates_reduced[:, -1].astype(int))  # Reorder last column
        self.coordinates_reduced = coordinates_reduced  # Store reduced coordinates
        self.point_id_reduced = np.argsort(coordinates_reduced[:, -1].astype(int))  # Store sorted reduced IDs

        # Store interpolation coefficients and point ID data for pooling/unpooling
        self.unpool_interp_coeff_list = intrp_coeffs_decoder
        self.unpool_point_id_data_list = point_id_data_decoder
        self.pool_interp_coeff_list = intrp_coeffs_encoder
        self.pool_point_id_data_list = point_id_data_encoder
        self.reduced_connectivity_array = reduced_conn

        self.edge_distances = edge_distances  # Store original edge distances
        self.edge_indices_reduced_orig = self.reduced_connectivity_array[['point_i', 'point_j']]  # Reduced edge indices
        self.edge_distances_reduced = self.reduced_connectivity_array[['distance']]  # Reduced edge distances
        self.point_id_reduced_orig = pd.unique(self.reduced_connectivity_array['point_i'])  # Unique reduced point IDs
        self.point_id_index_reduced_orig = np.where(np.isin(point_id, self.point_id_reduced_orig))[0]  # Indices of reduced IDs in original

        # Rename edge indices to match new point ID ordering
        self.edge_indices = self.rename_edge_indices(point_id, edge_indices)
        self.edge_indices_reduced = self.rename_edge_indices(self.point_id_reduced_orig, self.edge_indices_reduced_orig)

        # Prepare unpooling data: rename and convert to tensors, then sort
        self.unpool_point_id_data_list_ = self._convert_dict_tensor(
            self._rename_data_values(self.unpool_point_id_data_list, self.point_id_reduced_orig)
        )
        self.unpool_interp_coeff_list_ = self._convert_dict_tensor(self.unpool_interp_coeff_list)
        point_id_orig_temp = np.argsort(point_id)  # Indices to sort original point IDs
        self.unpool_point_id_data_list_sorted = {
            k: self.unpool_point_id_data_list_[k]
            for k in sorted(self.unpool_point_id_data_list_.keys(), key=lambda x: point_id_orig_temp[x])
        }
        self.unpool_interp_coeff_list_sorted = {
            k: self.unpool_interp_coeff_list_[k]
            for k in sorted(self.unpool_interp_coeff_list_.keys(), key=lambda x: point_id_orig_temp[x])
        }

        # Prepare pooling data: rename and convert to tensors, then sort
        self.pool_point_id_data_list_ = self._convert_dict_tensor(
            self._rename_data_values(self.pool_point_id_data_list, point_id)
        )
        self.pool_interp_coeff_list_ = self._convert_dict_tensor(self.pool_interp_coeff_list)
        self.pool_point_id_data_list_sorted = {
            k: self.pool_point_id_data_list_[k]
            for k in sorted(self.pool_point_id_data_list_.keys(), key=lambda x: point_id_orig_temp[x])
        }
        self.pool_interp_coeff_list_sorted = {
            k: self.pool_interp_coeff_list_[k]
            for k in sorted(self.pool_interp_coeff_list_.keys(), key=lambda x: point_id_orig_temp[x])
        }

    def _get_interpolation_matrix(self):
        """
        Create sparse interpolation matrices for encoder and decoder.

        Returns:
            tuple: (encoder_sparse_interpolation, decoder_sparse_interpolation)
        """
        _, encoder_sparse_interpolation = self._create_interpolation_matrix(
            self.pool_interp_coeff_list_sorted, self.pool_point_id_data_list_sorted, self.point_id_orig, decoder=False
        )
        _, decoder_sparse_interpolation = self._create_interpolation_matrix(
            self.unpool_interp_coeff_list_sorted, self.unpool_point_id_data_list_sorted, self.point_id_reduced_orig, decoder=True
        )
        return (encoder_sparse_interpolation, decoder_sparse_interpolation)

    def rename_edge_indices(self, point_id, edge_indices):
        """
        Rename edge indices to match the new ordering of point IDs.

        Args:
            point_id (np.ndarray): Array of point IDs.
            edge_indices (pd.DataFrame): DataFrame with 'point_i' and 'point_j'.

        Returns:
            pd.DataFrame: DataFrame with renamed indices.
        """
        if not set(edge_indices['point_i']).issubset(point_id) or not set(edge_indices['point_j']).issubset(point_id):
            raise ValueError("At least one point in edge_indices does not exist in point_id")
        sorted_points = np.sort(point_id)  # Sort point IDs
        label_to_index = {label: idx for idx, label in enumerate(sorted_points)}  # Map label to index
        edge_indices_copy = edge_indices.copy()  # Copy DataFrame
        edge_indices_copy['point_i'] = edge_indices_copy['point_i'].map(label_to_index)  # Rename point_i
        edge_indices_copy['point_j'] = edge_indices_copy['point_j'].map(label_to_index)  # Rename point_j
        return edge_indices_copy

    def _convert_dict_tensor(self, dict):
        """
        Convert all lists in a dictionary to stacked torch tensors.

        Args:
            dict (dict): Dictionary with lists of tensors.

        Returns:
            dict: Dictionary with stacked tensors.
        """
        for key in dict:
            dict[key] = torch.stack(dict[key])  # Stack list of tensors
        return dict

    def _rename_data_values(self, unpool_point_id_data_list, point_id):
        """
        Rename point IDs in data lists to a new range (0 to max) for GCN compatibility.

        Args:
            unpool_point_id_data_list (dict): Dictionary of lists of point IDs.
            point_id (np.ndarray): Array of point IDs.

        Returns:
            dict: Dictionary with renamed point IDs as tensors.
        """
        sorted_points = np.sort(point_id)  # Sort point IDs
        label_to_index = {label: idx for idx, label in enumerate(sorted_points)}  # Map label to index
        renamed_data_list = {}
        for key, data_list in unpool_point_id_data_list.items():
            renamed_data_list[key] = [
                torch.tensor(label_to_index.get(value.item(), value), dtype=torch.int64) for value in data_list
            ]  # Rename each value
        return renamed_data_list

    def _create_interpolation_matrix(self, data_list, point_id_data_list_, point_orig, decoder=False):
        """
        Create an interpolation matrix and its sparse representation.

        Args:
            data_list (dict): Dictionary of weights.
            point_id_data_list_ (dict): Dictionary of indices.
            point_orig (np.ndarray): Original point IDs.
            decoder (bool): Whether this is for the decoder.

        Returns:
            tuple: (interpolation_matrix, sparse_interpolation_matrix)
        """
        interpolation_matrix = np.zeros((len(point_id_data_list_), len(point_orig)), dtype=float)  # Initialize matrix
        i = 0
        for row_key, indices in point_id_data_list_.items():
            weights = data_list[row_key]  # Get weights for this row
            interpolation_matrix[i, indices] = weights  # Assign weights to matrix
            i += 1
        interpolation_matrix = torch.tensor(interpolation_matrix, dtype=torch.float32)  # Convert to tensor
        sparse_interpolation_matrix = interpolation_matrix.to_sparse_coo()  # Convert to sparse
        return interpolation_matrix, sparse_interpolation_matrix

    def _get_edge_indices(self):
        """
        Get edge indices as a torch tensor.

        Returns:
            torch.Tensor: Edge indices tensor.
        """
        return torch.tensor(np.array(self.edge_indices), dtype=torch.int64).t().contiguous()

    def _get_edge_distances(self):
        """
        Get edge distances as a torch tensor.

        Returns:
            torch.Tensor: Edge distances tensor.
        """
        return torch.tensor(np.array(self.edge_distances).squeeze(), dtype=torch.float32).t().contiguous()

    def _get_indices_reduced(self):
        """
        Get reduced edge indices as a torch tensor.

        Returns:
            torch.Tensor: Reduced edge indices tensor.
        """
        return torch.tensor(np.array(self.edge_indices_reduced), dtype=torch.int64).t().contiguous()

    def _get_edge_distances_reduced(self):
        """
        Get reduced edge distances as a torch tensor.

        Returns:
            torch.Tensor: Reduced edge distances tensor.
        """
        return torch.tensor(np.array(self.edge_distances_reduced).squeeze(), dtype=torch.float32).t().contiguous()

    def _get_point_id_index_reduced(self):
        """
        Get indices of reduced point IDs in the original set as a torch tensor.

        Returns:
            torch.Tensor: Indices tensor.
        """
        return torch.tensor(np.array(self.point_id_index_reduced_orig), dtype=torch.int64).t().contiguous()

    def save(self, path):
        """
        Save all relevant tensors to disk.

        Args:
            path (str): File path prefix for saving.
        """
        torch.save(self._get_edge_indices(), f"{path}_edge_indices.pt")  # Save edge indices
        torch.save(self._get_edge_distances(), f"{path}_edge_distances.pt")  # Save edge distances
        torch.save(self._get_indices_reduced(), f"{path}_indices_reduced.pt")  # Save reduced indices
        torch.save(self._get_edge_distances_reduced(), f"{path}_edge_distances_reduced.pt")  # Save reduced distances
        torch.save(self._get_point_id_index_reduced(), f"{path}_point_id_index_reduced.pt")  # Save reduced indices
        encoder_sparse_interpolation, decoder_sparse_interpolation = self._get_interpolation_matrix()  # Get interpolation matrices
        torch.save(encoder_sparse_interpolation, f"{path}_encoder_sparse_interpolation.pt")  # Save encoder matrix
        torch.save(decoder_sparse_interpolation, f"{path}_decoder_sparse_interpolation.pt")  # Save decoder matrix

    def load(self, path):
        """
        Load all relevant tensors from disk.

        Args:
            path (str): File path prefix for loading.
        """
        self.edge_indices = torch.load(f"{path}_edge_indices.pt")  # Load edge indices
        self.edge_distances = torch.load(f"{path}_edge_distances.pt")  # Load edge distances
        self.indices_reduced = torch.load(f"{path}_indices_reduced.pt")  # Load reduced indices
        self.edge_distances_reduced = torch.load(f"{path}_edge_distances_reduced.pt")  # Load reduced distances
        self.point_id_index_reduced = torch.load(f"{path}_point_id_index_reduced.pt")  # Load reduced indices
        self.encoder_sparse_interpolation = torch.load(f"{path}_encoder_sparse_interpolation.pt")  # Load encoder matrix
        self.decoder_sparse_interpolation = torch.load(f"{path}_decoder_sparse_interpolation.pt")  # Load decoder matrix