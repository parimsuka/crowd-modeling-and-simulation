# Define the logistic map function
import numpy as np
import matplotlib.pyplot as plt

def logistic_map(x, r):
    """Return the next iteration of the logistic map.
    Parameters
    ----------
    x : float
        The current value of the logistic map.
    r : float
        The value of the parameter r."""
    return r * x * (1 - x)


def plot_bifurcation_diagram(r_start, r_end, x_init=0.5, num_iterations=1000, last_n_iterations=100):
    """Plot the bifurcation diagram of the logistic map.

    Parameters
    ----------
    r_start : float
        The starting value of the parameter r.
    r_end : float
        The ending value of the parameter r.
    x_init : float, optional
        The initial condition for the logistic map (default is 0.5).
    num_iterations : int, optional
        The total number of iterations to run the logistic map (default is 1000).
    last_n_iterations : int, optional
        The number of final iterations to plot (default is 100).
    """
    # Set up parameters for the analysis
    r_values = np.linspace(r_start, r_end, 1000)

    # Initialize an array to store the steady states
    steady_states = np.zeros((len(r_values), last_n_iterations))

    # Iterate the logistic map and store the steady states
    for i, r in enumerate(r_values):
        x = x_init
        for j in range(num_iterations):
            x = logistic_map(x, r)
            if j >= num_iterations - last_n_iterations:
                steady_states[i, j - (num_iterations - last_n_iterations)] = x

    # Plot the bifurcation diagram
    plt.figure(figsize=(10, 7))
    plt.plot(r_values, steady_states, ',k', alpha=0.1)
    plt.title('Bifurcation Diagram of the Logistic Map')
    plt.xlabel('r')
    plt.ylabel('Steady States')
    plt.show()