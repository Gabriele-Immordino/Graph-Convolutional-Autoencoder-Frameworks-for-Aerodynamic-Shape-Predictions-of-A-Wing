import numpy as np

def calculate_probability_array(elements, p1, pn):
    """
    Calculate probabilities for an array of elements based on a custom exponential formula.

    Parameters:
    elements (np.ndarray): Array of mesh node indices.
    p1 (float): Probability for the smallest element.
    pn (float): Probability for the largest element.

    Returns:
    np.ndarray: Array of calculated probabilities.
    """
    n = len(elements)  # Get the number of elements
    exponent_terms = -2 * (elements / n)  # Compute exponent terms for the formula
    probabilities = 1 + ((1 - np.exp(exponent_terms)) / (1 - np.exp(-2))) * (pn - p1) + p1  # Calculate probabilities
    return probabilities  # Return the probability array

def get_reduced_space(variable, coordinates, point_id, p1, pn, num_selected):
    """
    Reduce the variable and coordinates arrays based on a probability distribution.

    Parameters:
    variable (np.ndarray): Array of variable values.
    coordinates (np.ndarray): Array of coordinates, last column should be point IDs.
    point_id (np.ndarray): Array of point IDs corresponding to variable.
    p1 (float): Probability for the smallest element.
    pn (float): Probability for the largest element.
    num_selected (int): Number of elements to select.

    Returns:
    tuple: (variable_reduced, coordinates_reduced)
        variable_reduced (np.ndarray): Reduced variable array.
        coordinates_reduced (np.ndarray): Reduced coordinates array.
    """
    variables_sorted = np.array(sorted(variable))[::-1]  # Sort variable in descending order
    indices_sorted = np.arange(len(variables_sorted))  # Create an array of sorted indices

    result_probabilities = calculate_probability_array(indices_sorted, p1, pn)  # Calculate probabilities for sorted indices
    normalized_probabilities = result_probabilities / np.sum(result_probabilities)  # Normalize probabilities

    selected_indices = np.random.choice(
        np.arange(len(variables_sorted)),
        size=num_selected,
        replace=False,
        p=normalized_probabilities
    )  # Randomly select indices based on probabilities

    variables_sorted_reduced = variables_sorted[selected_indices]  # Get reduced variable values

    common_elements, indices_variable, indices_reduced = np.intersect1d(
        variable, variables_sorted_reduced, return_indices=True
    )  # Find common elements and their indices

    indices_variable = np.sort(indices_variable)  # Sort the indices for consistent ordering

    variable_reduced = variable[indices_variable]  # Get reduced variable array
    ptids = point_id[indices_variable]  # Get corresponding point IDs
    variable_reduced = np.array([[x] for x in variable_reduced])  # Reshape to column vector

    coordinates_reduced = coordinates[np.isin(coordinates[:, -1], ptids)]  # Filter coordinates by selected point IDs


    return variable_reduced, coordinates_reduced  # Return reduced arrays

