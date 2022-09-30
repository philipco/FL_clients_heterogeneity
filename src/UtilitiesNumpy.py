"""Created by Constantin Philippenko, 30th September 2022."""
import math

import numpy as np


def remove_diagonal(distribution: np.array, symmetric_matrix) -> np.array:

    # If the matrix is symmetric, we also need to remove symmetric elements.
    if symmetric_matrix:
        r, c = np.triu_indices(len(distribution), 1)
        return distribution[r, c]

    distrib_without_diag = distribution.flatten()
    return np.delete(distrib_without_diag, range(0, len(distrib_without_diag), len(distribution) + 1), 0)


def create_matrix_with_zeros_diagonal_from_array(distribution: np.array) -> np.array:
    d = list(np.concatenate(distribution))
    size = (1 + math.sqrt(1 + 4 * len(d))) / 2
    assert size.is_integer(), "The size is not an integer."
    size = int(size)
    for i in range(size):
        d.insert(i * size + i, 0)
    return np.array(d).reshape((size, size))