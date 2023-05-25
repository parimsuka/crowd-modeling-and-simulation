"""
Module implementing principal component analysis

author: Simon Blöchinger, Talwinder Singh
"""

import typing

import numpy as np


def pca_forward(data_matrix: list[list[float]],
                component_count: typing.Optional[int] = None,
                full_matrices: bool = False
                ) -> (np.ndarray, np.ndarray, np.ndarray, float, list[float], np.ndarray):
    """
    Performs principal component analysis of a data_matrix with a number of components.

    :param data_matrix: Data matrix with rows x_i.
    :param component_count: The number of components used for the PCA.
    :param full_matrices: If the full matrices should be returned by np.linalg.svd.
    :return: u,s,vh of the singular vector decomposition, as well as the energy of this decomposition,
        the energy of the components and the removed mean that was removed to center the matrix.
    """
    # create a copy of the data_matrix
    data_matrix_cp = np.copy(data_matrix)

    # 1. Form data matrix with rows x_i
    # Nothing to do here, as the pca_dataset.txt file is already in the correct format

    # 2. Center matrix

    # compute mean of each column
    column_mean = np.mean(data_matrix_cp, axis=0)

    # center the matrix
    data_matrix_cp -= column_mean

    # 3. Decompose centered matrix into singular vectors
    u, s, vh = np.linalg.svd(data_matrix_cp, full_matrices=full_matrices)

    if component_count:
        partial_s = np.copy(s)
        np.put(partial_s, [i for i in range(component_count, len(partial_s))], 0)
    else:
        partial_s = s

    # 4. Compute energy (explained variance)

    # calculate the energy of the pca using component_count components
    energy = np.power(np.sum(np.square(s)), -1) * np.sum(np.square(partial_s))

    # calculate the energy per component
    energy_per_component = []
    for component in s:
        energy_per_component.append(np.power(np.sum(np.square(s)), -1) * np.sum(np.square(component)))

    return u, partial_s, vh, energy, energy_per_component, column_mean


def pca_reverse(u: np.ndarray, s: np.ndarray, vh: np.ndarray, removed_mean: np.ndarray) -> np.ndarray:
    """
    Reconstructs an original matrix using the singular value decomposition matrices U, S and V.

    :param u: The singular value decomposition matrix U containing the left singular vectors.
    :param s: The singular value decomposition matrix S containing the singular values as a list.
    :param vh: The singular value decomposition matrix V containing the right singular vectors.
    :param removed_mean: The mean that was removed previously, which will be added again in this step.
    :return: The reconstructed matrix.
    """
    # reconstruct the matrix
    rec_matrix = u * s @ vh

    # re-add the mean
    rec_matrix += removed_mean

    return rec_matrix
