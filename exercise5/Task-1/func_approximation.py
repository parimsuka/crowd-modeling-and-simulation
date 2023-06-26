import numpy as np
from numpy.linalg import lstsq as np_lstsq
import matplotlib.pyplot as plt


def radial_basis_function(x, center, epsilon) -> float:
    """Radial basis function with center c and width epsilon

    Args:
        x (float): The input value
        center (float): The center of the radial basis function
        epsilon (float): The width of the radial basis function

    Returns:
        float: The value of the radial basis function at x
    """

    return np.exp(-np.linalg.norm(x - center) ** 2 / epsilon**2)


def radial_basis_function_approximation(
    X, f, centers, epsilon, reg = 1e-3
) -> tuple[float, np.ndarray]:
    """Approximate the function f with radial basis functions.

    Args:
        X (np.ndarray): The input data
        f (np.ndarray): The function values
        centers (np.ndarray): The centers of the radial basis functions
        epsilon (float): The width of the radial basis functions

    Returns:
        C (np.ndarray): The coefficients of the linear combination of radial 
        basis functions
        Phi (np.ndarray): The matrix Phi

    """
    # Create the matrix Phi
    Phi = np.zeros((len(X), len(centers)))
    for i, x in enumerate(X):
        for j, c in enumerate(centers):
            Phi[i, j] = radial_basis_function(x, c, epsilon)

    # Perform least squares linear regression
    C, _, _, _ = np_lstsq(Phi, f, reg)

    return C, Phi


def plot_approximations(X, f, L_values, epsilon_values, centers_range) -> None:
    """Plot the original data and the approximated function for different
    values of L and epsilon.

    Args:
        X (np.ndarray): The input data
        f (np.ndarray): The function values
        L_values (list[int]): The values of L
        epsilon_values (list[float]): The values of epsilon
        centers_range (tuple[float, float]): The range of the centers
    """
    # Sort the data for plotting
    sort_idx = np.argsort(X.flatten())
    X_sorted = X[sort_idx]
    f_sorted = f[sort_idx]

    # Loop over the L values
    for L in L_values:
        # Define the centers
        centers = np.linspace(centers_range[0], centers_range[1], L)

        # Create a new figure for this L value
        plt.figure(figsize=(10, 6))
        plt.scatter(X_sorted, f_sorted, color="blue", label="Original Data")

        # Loop over the epsilon values
        for epsilon in epsilon_values:
            # Perform the radial basis function approximation
            C, Phi = radial_basis_function_approximation(X, f, centers, epsilon)

            # Predict the function values
            f_pred = Phi @ C
            f_pred_sorted = f_pred[sort_idx]

            # Plot the approximated function
            plt.plot(
                X_sorted,
                f_pred_sorted,
                label=f"Approximated Function (epsilon={epsilon}, centers={L})",
            )

        # Add labels and title to the plot
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.title(
            f"Approximation of the Function with Radial Basis Functions (centers={L})"
        )
        plt.legend()
        plt.show()


def plot_function_data(X, f, f_pred, title, label_pred) -> None:
    """Plot the original data and the approximated function
    
    Args:
        X (np.ndarray): The input data
        f (np.ndarray): The function values
        f_pred (np.ndarray): The predicted function values
        title (str): The title of the plot
        label_pred (str): The label of the predicted function
    """
    # Sort the data for plotting
    sort_idx = np.argsort(X.flatten())
    X_sorted = X[sort_idx]
    f_sorted = f[sort_idx]
    f_pred_sorted = f_pred[sort_idx]

    # Plot the original data and the approximated function
    plt.figure(figsize=(10, 6))
    plt.scatter(X_sorted, f_sorted, color='blue', label='Original Data')
    plt.plot(X_sorted, f_pred_sorted, color='red', label=label_pred)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(title)
    plt.legend()
    plt.show()
