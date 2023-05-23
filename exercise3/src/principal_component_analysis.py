"""
Module implementing principal component analysis

author: Simon Blöchinger, Talwinder Singh
"""

import typing

import numpy as np


def pca_forward(data_matrix: list[list[float]],
                component_count: typing.Optional[int] = None,
                full_matrices: bool = False):
    """
    Performs principal component analysis of a data_matrix with a number of components.

    :param data_matrix: Data matrix with rows x_i.
    :param component_count: The number of components used for the PCA.
    :param full_matrices: If the full matrices should be returned by np.linalg.svd.
    :return: u,s,vh of the singular vector decomposition and the energy.
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

    # 4. Compute energy (explained variance)

    # If a number of components is given: create a partial_s with only that number of components
    if component_count:
        partial_s = s[:component_count]
    else:
        partial_s = s

    # calculate the energy of the pca using component_count components
    energy = np.power(np.sum(np.square(s)), -1) * np.sum(np.square(partial_s))

    energy_per_component = []
    for component in s:
        energy_per_component.append(np.power(np.sum(np.square(s)), -1) * np.sum(np.square(component)))

    return u, s, vh, energy, energy_per_component


def pca_reverse():
    None  #todo (?)
