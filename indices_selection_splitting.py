import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt


def _max_strc_modes_idx(dataset):
    """
    Returns the indices of the top 30 samples with the highest combined maximum absolute values
    across all variables for each sample in the dataset.
    
    Args:
        dataset (np.ndarray): Input data of shape (samples, ..., variables).
        
    Returns:
        np.ndarray: Indices of the top 30 samples.
    """
    max_combined_values = np.sum(np.abs(np.max(dataset, axis=2)), axis=1)  # Sum of max abs values per sample
    sorted_indices = np.argsort(max_combined_values)[::-1]  # Indices sorted by descending value
    return sorted_indices[:30]  # Top 30 indices

def _move_indices(X, Y, train_indices, val_indices, test_indices):
    """
    Adjusts the train, validation, and test indices to ensure the top 30 samples (by max structure modes)
    are distributed among the splits, avoiding duplicates.
    
    Args:
        X (np.ndarray): Feature data.
        Y (np.ndarray): Target data.
        train_indices (list): Initial training indices.
        val_indices (list): Initial validation indices.
        test_indices (list): Initial test indices.
        
    Returns:
        tuple: Updated (train_indices, val_indices, test_indices).
    """
    additional_indices = _max_strc_modes_idx(X[:,:,3:])  # Get top 30 indices
    random.shuffle(additional_indices)  # Shuffle for randomness

    additional_train = additional_indices[:10]  # First 10 for training
    additional_val = additional_indices[10:20]  # Next 10 for validation
    additional_test = additional_indices[20:]  # Remaining for testing

    # Remove these indices from existing splits to avoid duplicates
    train_indices = sorted([idx for idx in train_indices if idx not in additional_indices])
    val_indices = sorted([idx for idx in val_indices if idx not in additional_indices])
    test_indices = sorted([idx for idx in test_indices if idx not in additional_indices])

    # Add the new indices to each split
    train_indices.extend(add_idx for add_idx in additional_train if add_idx not in train_indices)
    val_indices.extend(add_idx for add_idx in additional_val if add_idx not in val_indices)
    test_indices.extend(add_idx for add_idx in additional_test if add_idx not in test_indices)

    # Sort indices for consistency
    train_indices = sorted(train_indices)
    val_indices = sorted(val_indices)
    test_indices = sorted(test_indices)

    return train_indices, val_indices, test_indices

def _strat_kmeans(X, Y, initial_indices, modes, k, seed_value, train_ratio, val_ratio, test_ratio):
    """
    Performs stratified splitting using KMeans clustering on the provided modes.
    Adjusts splits to ensure balance and calculates a splitting score.
    
    Args:
        X (np.ndarray): Feature data.
        Y (np.ndarray): Target data.
        initial_indices (np.ndarray): Array of sample indices.
        modes (np.ndarray): Data for clustering.
        k (int): Number of clusters.
        seed_value (int): Random seed.
        train_ratio (float): Proportion for training set.
        val_ratio (float): Proportion for validation set.
        test_ratio (float): Proportion for test set.
        
    Returns:
        tuple: (X_train, X_val, X_test, Y_train, Y_val, Y_test, (train_indices, val_indices, test_indices), sp_score)
    """
    kmeans = KMeans(n_clusters=k, random_state=seed_value, n_init=10)  # KMeans clustering
    strat_labels = kmeans.fit_predict(modes)  # Cluster labels

    # Stratified split: train vs temp
    X_train, X_temp, Y_train, Y_temp, train_indices, temp_indices, strat_train, strat_temp = train_test_split(
        X, Y, initial_indices, strat_labels, test_size=(1 - train_ratio), random_state=seed_value, stratify=strat_labels
    )
    temp_val_ratio = val_ratio / (val_ratio + test_ratio)  # Proportion of val in temp
    # Stratified split: val vs test
    X_val, X_test, Y_val, Y_test, val_indices, test_indices, strat_val, strat_test = train_test_split(
        X_temp, Y_temp, temp_indices, strat_temp, test_size=(1 - temp_val_ratio), random_state=seed_value, stratify=strat_temp
    )

    # Adjust indices for balance
    train_indices, val_indices, test_indices = _move_indices(X, Y, train_indices, val_indices, test_indices)
    X_train, X_val, X_test = X[train_indices], X[val_indices], X[test_indices]  # Update splits
    Y_train, Y_val, Y_test = Y[train_indices], Y[val_indices], Y[test_indices]

    # Calculate correlation score for each split
    total_corr_value = 0
    for split_name, data in zip(['Train', 'Validation', 'Test'], [X_train, X_val, X_test]):
        modes_data = data[:, :, 3:9].reshape(-1, 6)  # Extract and reshape modes
        correlation_matrix = np.corrcoef(modes_data, rowvar=False)  # Correlation matrix
        overall_correlation = np.sum(np.square(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]))  # Sum squared off-diagonal
        total_corr_value += overall_correlation

    sp_score = 1 - total_corr_value / 45  # Splitting score
    print(f"(k={k}) Splitting score: {sp_score}")

    return X_train, X_val, X_test, Y_train, Y_val, Y_test, (list(train_indices), list(val_indices), list(test_indices)), sp_score

def split_dataset_stratified(X, Y, k=28, seed_value=50, train_ratio=0.52, val_ratio=0.24, test_ratio=0.24):
    """
    Splits the dataset into stratified train, validation, and test sets using KMeans clustering.
    
    Args:
        X (np.ndarray): Feature data.
        Y (np.ndarray): Target data.
        k (int): Number of clusters for stratification.
        seed_value (int): Random seed.
        train_ratio (float): Proportion for training set.
        val_ratio (float): Proportion for validation set.
        test_ratio (float): Proportion for test set.
        
    Returns:
        tuple: (train_indices, val_indices, test_indices)
    """
    np.random.seed(seed_value)  # Set numpy random seed
    random.seed(seed_value)  # Set python random seed

    initial_indices = np.arange(len(X))  # Indices for all samples
    modes = X[:, 0, 3:]  # Modes for clustering

    # Perform stratified splitting
    X_train, X_val, X_test, Y_train, Y_val, Y_test, (train_indices, val_indices, test_indices), sp_score = _strat_kmeans(
        X, Y, initial_indices, modes, k, seed_value, train_ratio, val_ratio, test_ratio
    )

    return train_indices, val_indices, test_indices