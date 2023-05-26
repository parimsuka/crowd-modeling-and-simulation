"""
Implementation of the diffusion map algorithm.
"""

import numpy as np
from scipy.spatial import distance_matrix
from sklearn.datasets import make_swiss_roll
import matplotlib.pyplot as plt
import scipy as sp
from math import pi
from sklearn.decomposition import PCA



EPS = 0.05

def diffusion_map(
    data: np.ndarray, L: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Implementation of the diffusion map algorithm.

    Parameters:
    data (np.ndarray): The input data, a 2D array-like object where each row is a data
        point.
    L (int): The number of dimensions to reduce the data to.

    Returns:
    Tuple[np.ndarray, np.ndarray]: A tuple of two 1D arrays: the eigenvalues and the
        eigenvectors.
    """

    # Form a distance matrix D
    D = distance_matrix(data, data) 
    
    # Set ϵ to 5% of the diameter of the dataset
    epsilon = 0.05*np.max(D)

    # Form the kernel matrix W
    W = np.exp(-np.square(D)/epsilon)
    
    #  Form the diagonal normalization matrix P
    P = np.diag(np.sum(W, axis=1))

    # Compute P inverse
    P_inv = np.linalg.inv(P)
    
    # Normalize W to form the kernel matrix K 
    K = P_inv @ W @ P_inv

    #  Form the diagonal normalization matrix Q
    Q = np.diag(np.sum(K, axis=1))
    
    # Compute Q inverse square root
    Q_inv_sqrt = np.linalg.inv(np.sqrt(Q))

    #  Form the symmetric matrix T_hat
    T_hat = Q_inv_sqrt @ K @ Q_inv_sqrt

    # Compute L + 1 largest eigvalues and corresponding eigvectors of T_hat
    eigenvalues, eigenvectors = np.linalg.eigh(T_hat)

    # Reverse order of eigvalues and eigvectors because they are returned in ascending
    # order
    eigenvalues = eigenvalues[-L:][::-1]
    eigenvectors = eigenvectors[:, -L:][:, ::-1]

    # Compute lambda and phi from the eigvalues and eigvectors
    lambda_values = eigenvalues**(1/(2*epsilon))
    phi = Q_inv_sqrt @ eigenvectors

    return lambda_values, phi

'''

def diffusion_map(
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
'''


def create_periodic_data(N=1000):
    """
    Create the periodic data for the first part of the task.

    Parameters:
    N (int): The number of data points to create.

    Returns:
    Tuple[np.ndarray, np.ndarray]: A tuple of two 1D arrays
    """
    t_k = 2 * pi * np.array(np.arange(0, N)) / (N + 1)
    x_k = np.array([np.cos(t_k), np.sin(t_k)])
    return x_k.T, t_k.T


def create_swiss_roll_data(N= 5000):
    """
    Create the swiss roll data for the second part of the task using make swiss roll from
    sklearn.

    Parameters:
    N (int): The number of data points to create.

    Returns:
    Tuple[np.ndarray, np.ndarray]: A tuple of two 1D arrays
    """
    x_k, t_k = make_swiss_roll(N)
    return x_k, t_k


def plot_eigenfunctions(phi, t_k, L, save=False, filename="plot", task=1):
    """
    Plot the eigenfunctions.

    Parameters:
    phi (np.ndarray): The eigenfunctions.
    t_k (np.ndarray): The data points.
    L (int): The number of eigenfunctions.
    save (bool): Whether to save the plot.
    filename (str): The filename to save the plot to.
    task (int): The task number.

    Returns:
    None
    """

    for i in range(L):
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot()
        if task == 1: # Plot eigenfunctions for task 1
            ax.scatter(t_k, phi[:, i], c=t_k)
            ax.set_xlabel(r'$t_{k}$')
        else: # Plot eigenfunctions wrt to phi_1 for task 2
            ax.scatter(phi[:, 1], phi[:, i], c=t_k)
            ax.set_xlabel(r'$\phi_{1}(x_{k})$')
            
        ax.set_ylabel(r'$\phi_{' + str(i) + '}(x_{k})$')
        ax.set_title(r'Eigenfunction $\phi_{' + str(i) + '}$')
        ax.set_ylim([-0.5, 0.5])
        if save:
            fig.savefig(filename + "_" + str(i) + ".png")
        plt.show()



def plot_3d_data(x, t, title= '', xlabel= 'x', ylabel='y', zlabel = 'z', save = False,  filename = "plot"):
    """
    Plot the data in 3D.

    Parameters:
    x (np.ndarray): The data points.
    t (np.ndarray): The data points.
    title (str): The title of the plot.
    xlabel (str): The label of the x-axis.
    ylabel (str): The label of the y-axis.
    zlabel (str): The label of the z-axis.
    save (bool): Whether to save the plot.
    filename (str): The filename to save the plot to.

    Returns:
    None
    """
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=t)
    if save:
        fig.savefig(filename + ".png")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    plt.show()


def get_reconstructed_data_from_pca(x, n_components):
    """
    Get the reconstructed data from PCA.

    Parameters:
    x (np.ndarray): The data points.
    n_components (int): The number of principal components.

    Returns:
    np.ndarray: The reconstructed data.
    """

    # Compute the principal components
    pca = PCA(n_components=n_components)
    x_pca = pca.fit_transform(x)

    # Compute the explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_
    # Print the explained variance ratio
    for i, ratio in enumerate(explained_variance_ratio):
        print(f"Explained variance ratio for PC{i + 1}: {ratio:.4f}")

    # Compute the reconstructed data
    x_reconstructed = pca.inverse_transform(x_pca)

    return x_reconstructed