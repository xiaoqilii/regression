import numpy as np

def create_dummy_variables(categories):
    """Converts a list of categorical values into a matrix of dummy variables."""
    unique_categories = np.unique(categories)
    dummy_matrix = np.zeros((len(categories), len(unique_categories)))

    for i, category in enumerate(categories):
        category_index = np.where(unique_categories == category)[0][0]
        dummy_matrix[i, category_index] = 1

    return dummy_matrix

def sort_by_group(categories, y):
    """Reorder y values so that values are grouped by the categories."""
    sorted_indices = np.argsort(categories)
    sorted_y = y[sorted_indices]
    
    unique_categories = np.unique(categories)
    sorted_idx = np.argsort(unique_categories)
    sorted_categories = unique_categories[sorted_idx]
    
    return sorted_categories, sorted_y
