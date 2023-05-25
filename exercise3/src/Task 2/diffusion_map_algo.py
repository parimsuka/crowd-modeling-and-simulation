"""
Implementation of the diffusion map algorithm.
"""

import numpy as np
import scipy as sp

EPS = 0.05


def diffusion_map_algo(
    data: np.ndarray, radius: float, L: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Implementation of the diffusion map algorithm.

    Parameters:
    data (np.ndarray): The input data, a 2D array-like object where each row is a data
        point.
    radius (float): The maximum distance for the sparse distance matrix.
    L (int): The number of dimensions to reduce the data to.

    Returns:
    Tuple[np.ndarray, np.ndarray]: A tuple of two 1D arrays: the eigenvalues and the
        eigenvectors.
    """

    # Create KDTree from the data for efficient nearest neighbor queries
    datatree = sp.spatial.KDTree(data)

    # Compute sparse distance matrix according to given radius
    sparse_D = datatree.sparse_distance_matrix(datatree, max_distance=radius).tocoo().tocsr()

    # Compute epsilon according to EPS and max distance in sparse_D
    epsilon = EPS * sparse_D.data.max()

    # Compute zero division epsilon for numerical stability
    zero_div_eps = 1e-3 * np.median(sparse_D.data)

    # Create kernel matrix W from D
    W = sparse_D.copy()
    W.data **= 2
    W.data = np.exp(-W.data / epsilon)

    # Form diagonal normalization matrix
    P = create_diag_norm_matrix(W)

    # Compute inverse of P and add zero division epsilon for numerical stability
    P_inv = sp.sparse.diags(1.0 / (P.diagonal() + zero_div_eps))

    # Normalize W to form kernel matrix K
    K = P_inv @ W @ P_inv

    # Form diagonal normalization matrix
    Q = create_diag_norm_matrix(K)

    # Compute square root of the inverse of Q and add zero division epsilon for
    # numerical stability
    Q_inv_sqrt = sp.sparse.diags(1.0 / (np.sqrt(Q.diagonal()) + zero_div_eps))

    # Form symmetric matrix T_hat
    T_hat = Q_inv_sqrt @ K @ Q_inv_sqrt

    # Compute L + 1 largest eigvalues and corresponding eigvectors of T_hat
    eigenvalues, eigenvectors = sp.sparse.linalg.eigsh(T_hat, k=L + 1, which="LM")

    # Reverse order of eigvalues and eigenvectors because they are returned in ascending
    # order
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]

    # Compute eigvalues of T_hat^(1/epsilon)
    lambda_values = eigenvalues ** (1 / epsilon)

    # Compute eigenvectors of T
    phi = Q_inv_sqrt @ eigenvectors

    return lambda_values, phi


def create_diag_norm_matrix(mat):
    """
    Create a diagonal normalization matrix for a given sparse matrix.

    Parameters:
    mat (scipy.sparse.csr_matrix): The input sparse matrix.

    Returns:
    scipy.sparse.csr_matrix: The diagonal normalization matrix.
    """

    row_sums = np.array(mat.sum(axis=1)).ravel()

    diag_norm_matrix = sp.sparse.diags(row_sums)

    return diag_norm_matrix
