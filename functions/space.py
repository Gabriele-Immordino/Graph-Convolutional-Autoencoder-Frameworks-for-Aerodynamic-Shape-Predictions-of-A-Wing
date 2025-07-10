import numpy as np
import pandas as pd
import torch
from functions.Reduced_space_PDF_function import get_reduced_space
from functions.Codec import Codec
from functions.Connectivity import WriteConnectivity

class Space:
    """
    Class for handling reduced space operations, including dimensionality reduction,
    pooling/unpooling interpolation, and connectivity management for graph-based models.
    """

    def create(self, npoints, magnitude, coordinates, point_id, edge_indices, edge_distances, space2=False):
        """
        Perform dimensionality reduction and prepare all relevant data structures for the reduced space.

        Args:
            npoints (int): Number of points to select in reduced space.
            magnitude (np.ndarray): Magnitude data for all points.
            coordinates (np.ndarray): Coordinates for all points.
            point_id (np.ndarray): Original point IDs.
            edge_indices (pd.DataFrame): DataFrame with 'point_i' and 'point_j' columns for edges.
            edge_distances (np.ndarray): Distances for each edge.
            space2 (bool, optional): Flag for alternate handling of magnitude. Defaults to False.
        """
        self.point_id_orig = point_id  # Store original point IDs

        # Find Points in Reduced Space using dimensionality reduction
        magnitude_reduced, coordinates_reduced = get_reduced_space(
            magnitude, coordinates, point_id, p1=0.2, pn=1.0, num_selected=npoints
        )
        point_id_reduced = coordinates_reduced[:, -1].astype(int)  # Extract reduced point IDs

        # Prepare normalized dataset for Codec
        if not space2:
            normalized_dataset = magnitude_reduced[:, 0]  # Use first column if not space2
        else:
            normalized_dataset = np.array([el[0] for el in magnitude_reduced[:, 0]])  # Alternate handling

        # Create Codec for pooling/unpooling interpolation coefficients
        reduced_space = Codec(normalized_dataset, point_id_reduced, coordinates)
        intrp_coeffs_encoder, point_id_data_encoder = reduced_space.reduced_connectivity(mode='encoder')
        intrp_coeffs_decoder, point_id_data_decoder = reduced_space.reduced_connectivity(mode='decoder')

        # Write connectivity matrix for reduced space
        reduced_space = WriteConnectivity(normalized_dataset, point_id_reduced, coordinates)
        reduced_conn = reduced_space.reduced_connectivity()

        self.magnitude_reduced = magnitude_reduced  # Store reduced magnitudes

        # Sort reduced coordinates by their last column (point IDs)
        coordinates_reduced[:, -1] = np.argsort(coordinates_reduced[:, -1].astype(int))
        self.coordinates_reduced = coordinates_reduced  # Store reduced coordinates
        self.point_id_reduced = np.argsort(coordinates_reduced[:, -1].astype(int))  # Store sorted reduced point IDs

        # Store pooling/unpooling interpolation coefficients and point ID data
        self.unpool_interp_coeff_list = intrp_coeffs_decoder
        self.unpool_point_id_data_list = point_id_data_decoder
        self.pool_interp_coeff_list = intrp_coeffs_encoder
        self.pool_point_id_data_list = point_id_data_encoder
        self.reduced_connectivity_array = reduced_conn

        self.edge_distances = edge_distances  # Store edge distances
        self.edge_indices_reduced_orig = self.reduced_connectivity_array[['point_i', 'point_j']]  # Reduced edge indices
        self.edge_distances_reduced = self.reduced_connectivity_array[['distance']]  # Reduced edge distances
        self.point_id_reduced_orig = pd.unique(self.reduced_connectivity_array['point_i'])  # Unique reduced point IDs
        self.point_id_index_reduced_orig = np.where(np.isin(point_id, self.point_id_reduced_orig))[0]  # Indices in original IDs

        # Rename edge indices for original and reduced spaces
        self.edge_indices = self.rename_edge_indices(point_id, edge_indices)
        self.edge_indices_reduced = self.rename_edge_indices(self.point_id_reduced_orig, self.edge_indices_reduced_orig)

        # --- Unpooling ---
        # Convert and rename unpooling data for tensor operations and GCN compatibility
        self.unpool_point_id_data_list_ = self._convert_dict_tensor(
            self._rename_data_values(self.unpool_point_id_data_list, self.point_id_reduced_orig)
        )
        self.unpool_interp_coeff_list_ = self._convert_dict_tensor(self.unpool_interp_coeff_list)

        # Sort unpooling dictionaries by original point ID order
        point_id_orig_temp = np.argsort(point_id)
        self.unpool_point_id_data_list_sorted = {
            k: self.unpool_point_id_data_list_[k]
            for k in sorted(self.unpool_point_id_data_list_.keys(), key=lambda x: point_id_orig_temp[x])
        }
        self.unpool_interp_coeff_list_sorted = {
            k: self.unpool_interp_coeff_list_[k]
            for k in sorted(self.unpool_interp_coeff_list_.keys(), key=lambda x: point_id_orig_temp[x])
        }

        # --- Pooling ---
        # Convert and rename pooling data for tensor operations
        self.pool_point_id_data_list_ = self._convert_dict_tensor(
            self._rename_data_values(self.pool_point_id_data_list, point_id)
        )
        self.pool_interp_coeff_list_ = self._convert_dict_tensor(self.pool_interp_coeff_list)

        # Sort pooling dictionaries by original point ID order
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
        Create sparse interpolation matrices for pooling (encoder) and unpooling (decoder).

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
        Rename edge indices to match the new index range after reduction.

        Args:
            point_id (np.ndarray): Array of point IDs to map to.
            edge_indices (pd.DataFrame): DataFrame with 'point_i' and 'point_j' columns.

        Returns:
            pd.DataFrame: Edge indices with renamed point indices.
        """
        if not set(edge_indices['point_i']).issubset(point_id) or not set(edge_indices['point_j']).issubset(point_id):
            raise ValueError("At least one point in edge_indices does not exist in point_id")
        sorted_points = np.sort(point_id)  # Sort point IDs
        label_to_index = {label: idx for idx, label in enumerate(sorted_points)}  # Map label to new index
        edge_indices_copy = edge_indices.copy()
        edge_indices_copy['point_i'] = edge_indices_copy['point_i'].map(label_to_index)
        edge_indices_copy['point_j'] = edge_indices_copy['point_j'].map(label_to_index)
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
            dict[key] = torch.stack(dict[key])  # Stack list of tensors into one tensor
        return dict

    def _rename_data_values(self, unpool_point_id_data_list, point_id):
        """
        Rename point IDs in data lists to a new contiguous range (0 to max).

        Args:
            unpool_point_id_data_list (dict): Dictionary of data lists with point IDs.
            point_id (np.ndarray): Array of point IDs to map to.

        Returns:
            dict: Dictionary with renamed point IDs.
        """
        sorted_points = np.sort(point_id)
        label_to_index = {label: idx for idx, label in enumerate(sorted_points)}
        renamed_data_list = {}
        for key, data_list in unpool_point_id_data_list.items():
            renamed_data_list[key] = [
                torch.tensor(label_to_index.get(value.item(), value), dtype=torch.int64) for value in data_list
            ]
        return renamed_data_list

    def _create_interpolation_matrix(self, data_list, point_id_data_list_, point_orig, decoder=False):
        """
        Create an interpolation matrix for pooling or unpooling.

        Args:
            data_list (dict): Dictionary of interpolation coefficients.
            point_id_data_list_ (dict): Dictionary of point ID data.
            point_orig (np.ndarray): Original point IDs.
            decoder (bool, optional): Whether to create decoder (unpooling) matrix. Defaults to False.

        Returns:
            tuple: (dense interpolation matrix, sparse interpolation matrix)
        """
        interpolation_matrix = np.zeros((len(point_id_data_list_), len(point_orig)), dtype=float)
        i = 0
        for row_key, indices in point_id_data_list_.items():
            weights = data_list[row_key]
            interpolation_matrix[i, indices] = weights  # Assign weights to appropriate indices
            i += 1
        interpolation_matrix = torch.tensor(interpolation_matrix, dtype=torch.float32)
        sparse_interpolation_matrix = interpolation_matrix.to_sparse_coo()  # Convert to sparse format
        return interpolation_matrix, sparse_interpolation_matrix

    def _get_edge_indices(self):
        """
        Get edge indices as a torch tensor.

        Returns:
            torch.Tensor: Edge indices tensor (2, num_edges).
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
            torch.Tensor: Reduced edge indices tensor (2, num_edges).
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
        Get indices of reduced point IDs in the original point ID array.

        Returns:
            torch.Tensor: Indices tensor.
        """
        return torch.tensor(np.array(self.point_id_index_reduced_orig), dtype=torch.int64).t().contiguous()

    def save(self, path):
        """
        Save all relevant tensors to disk.

        Args:
            path (str): Path prefix for saving files.
        """
        torch.save(self._get_edge_indices(), f"{path}_edge_indices.pt")  # Save edge indices
        torch.save(self._get_edge_distances(), f"{path}_edge_distances.pt")  # Save edge distances
        torch.save(self._get_indices_reduced(), f"{path}_indices_reduced.pt")  # Save reduced edge indices
        torch.save(self._get_edge_distances_reduced(), f"{path}_edge_distances_reduced.pt")  # Save reduced edge distances
        torch.save(self._get_point_id_index_reduced(), f"{path}_point_id_index_reduced.pt")  # Save reduced point ID indices
        encoder_sparse_interpolation, decoder_sparse_interpolation = self._get_interpolation_matrix()
        torch.save(encoder_sparse_interpolation, f"{path}_encoder_sparse_interpolation.pt")  # Save encoder interpolation
        torch.save(decoder_sparse_interpolation, f"{path}_decoder_sparse_interpolation.pt")  # Save decoder interpolation

    def load(self, path):
        """
        Load all relevant tensors from disk.

        Args:
            path (str): Path prefix for loading files.
        """
        self.edge_indices = torch.load(f"{path}_edge_indices.pt")  # Load edge indices
        self.edge_distances = torch.load(f"{path}_edge_distances.pt")  # Load edge distances
        self.edge_indices_reduced = torch.load(f"{path}_indices_reduced.pt")  # Load reduced edge indices
        self.edge_distances_reduced = torch.load(f"{path}_edge_distances_reduced.pt")  # Load reduced edge distances
        self.point_id_index_reduced = torch.load(f"{path}_point_id_index_reduced.pt")  # Load reduced point ID indices
        self.encoder_sparse_interpolation = torch.load(f"{path}_encoder_sparse_interpolation.pt")  # Load encoder interpolation
        self.decoder_sparse_interpolation = torch.load(f"{path}_decoder_sparse_interpolation.pt")  # Load decoder interpolation
