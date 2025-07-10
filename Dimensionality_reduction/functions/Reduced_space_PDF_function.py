import numpy as np

def calculate_probability_array(elements, p1, pn):
    """
    Calculate probabilities for an array of elements based on a custom exponential formula.

    Parameters:
    - elements: numpy array of mesh node indices
    - p1: probability for the smallest element
    - pn: probability for the largest element

    Returns:
    - Numpy array of calculated probabilities
    """
    n = len(elements)  # Number of elements
    exponent_terms = -2 * (elements / n)  # Exponential scaling for each element
    probabilities = 1 + ((1 - np.exp(exponent_terms)) / (1 - np.exp(-2))) * (pn - p1) + p1  # Probability formula
    return probabilities  # Return the computed probabilities


def get_reduced_space(variable: np.ndarray, coordinates, point_id, p1, pn, num_selected):
    """
    Reduce the variable space by probabilistically selecting elements based on a custom PDF.

    Parameters:
    - variable: numpy array of variable values
    - coordinates: numpy array of coordinates (last column should match point_id)
    - point_id: numpy array of point IDs corresponding to variable
    - p1: probability for the smallest element
    - pn: probability for the largest element
    - num_selected: number of elements to select

    Returns:
    - variable_reduced: reduced variable array (2D column vector)
    - coordinates_reduced: coordinates corresponding to reduced variable
    """
    rumore = np.random.uniform(low=0, high=variable.min() * 0.00001, size=variable.shape)  # Add small random noise
    variable_noisy = variable + rumore  # Noisy variable to avoid ties
    variables_sorted = np.sort(variable_noisy)[::-1]  # Sort in descending order
    indices_sorted = np.arange(len(variables_sorted))  # Indices for sorted array

    result_probabilities = calculate_probability_array(indices_sorted, p1, pn)  # Calculate probabilities
    normalized_probabilities = result_probabilities / np.sum(result_probabilities)  # Normalize probabilities

    selected_indices = np.random.choice(
        np.arange(len(variables_sorted)),
        size=num_selected,
        replace=False,
        p=normalized_probabilities
    )  # Randomly select indices based on probabilities

    variables_sorted_reduced = variables_sorted[selected_indices]  # Get reduced sorted variables

    # Find common elements and their indices in the noisy variable and reduced set
    common_elements, indices_variable, indices_reduced = np.intersect1d(
        variable_noisy, variables_sorted_reduced, return_indices=True
    )
    indices_variable = np.sort(indices_variable)  # Sort indices for consistent order

    variable_reduced = variable[indices_variable]  # Get reduced variable values from original array
    ptids = point_id[indices_variable]  # Get corresponding point IDs
    variable_reduced = np.array([[x] for x in variable_reduced])  # Convert to column vector

    coordinates_reduced = coordinates[np.isin(coordinates[:, -1], ptids)]  # Filter coordinates by selected point IDs

    return variable_reduced, coordinates_reduced  # Return reduced variable and coordinates
