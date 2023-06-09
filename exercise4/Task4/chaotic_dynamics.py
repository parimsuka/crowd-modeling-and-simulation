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


def lorenz(X = [10, 10, 10], sigma=10, beta=8/3, rho=28):
    """Return the derivatives of the Lorenz system.
    Parameters
    ----------

    X : array_like, shape (3,)
        The current state of the system (x, y, and z).
    sigma : float, optional
        The value of the parameter sigma (default is 10).
    beta : float, optional
        The value of the parameter beta (default is 8/3).
    rho : float, optional
        The value of the parameter rho (default is 28).

    Returns
    -------
    dx, dy, dz : float
        The derivatives of the Lorenz system at the current point.
    """
    x, y, z = X[0], X[1], X[2]
    dx = sigma * (y - x)
    dy = rho * x - y - x * z
    dz = x * y - beta * z
    return dx, dy, dz


def get_lorenz_trajectory(t_end=1000, start=[10, 10, 10], sigma=10, beta= 8 / 3, rho=28, dt=0.01):
    """Get the trajectory of the Lorenz system.
    Parameters
    ----------
    dt : float, optional
        The time step (default is 0.01).
    t_end : float, optional
        The final time (default is 1000).
    start : tuple, optional
        The initial condition (default is (10, 10, 10)).
    sigma : float, optional
        The value of the parameter sigma (default is 10).
    beta : float, optional
        The value of the parameter beta (default is 2.667).
    rho : float, optional
        The value of the parameter rho (default is 28).
    """
    # Set up parameters for the analysis
    iterations = int(t_end / dt)
    init_x, init_y, init_z = start[0], start[1], start[2]

    # Initialize lists to store the x, y, and z values
    x_list = [init_x]
    y_list = [init_y]
    z_list = [init_z]

    # Iterate the Lorenz system and store the x, y, and z values
    for i in range(iterations):
        dx, dy, dz = lorenz([x_list[i], y_list[i], z_list[i]], sigma, beta, rho)
        x_list.append(x_list[i] + (dx * dt))
        y_list.append(y_list[i] + (dy * dt))
        z_list.append(z_list[i] + (dz * dt))

    # Return the x, y, and z lists 
    return x_list, y_list, z_list


def plot_lorenz_trajectory(x_list, y_list, z_list, linewidth, save = False, filename = ''):
    """Plot the trajectory of the Lorenz system.
    Parameters
    ----------
    x_list : list
        A list of x values.
    y_list : list
        A list of y values.
    z_list : list
        A list of z values.
    linewidth : float
        The width of the line.
    save : bool, optional
        Whether to save the plot (default is False).
    filename : str, optional
        The name of the file to save (default is '').
    """
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(projection='3d')
    ax.plot(x_list, y_list, z_list, linewidth = linewidth )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Lorenz Attractor")
    
    if save:
        fig.savefig(filename + '.png')
    plt.show()


def plot_difference_between_trajectories(x1, y1, z1, x2, y2, z2, save = False, filename = ''):
    """Plot the difference between two trajectories of the Lorenz system.
    Parameters
    ----------
    x1 : list
        A list of x values for the first trajectory.
    y1 : list
        A list of y values for the first trajectory.
    z1 : list
        A list of z values for the first trajectory.
    x2 : list
        A list of x values for the second trajectory.
    y2 : list
        A list of y values for the second trajectory.
    z2 : list
        A list of z values for the second trajectory.
    save : bool, optional
        Whether to save the plot (default is False).
    filename : str, optional
        The name of the file to save (default is '').
    """
    p1 = np.array([x1, y1, z1])
    p2 = np.array([x2, y2, z2])

    # Calculate the squared distance between the two trajectories
    squared_dist = np.sum((p1-p2)**2, axis=0)

    # Generate the time step
    time_step = np.arange(0, len(squared_dist), 1)

    # Plot the difference between the two trajectories
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot()
    ax.plot(time_step, squared_dist, linewidth = 0.5)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Difference")
    ax.set_title("Difference between two trajectories of Lorenz system with initial condition (10, 10, 10) and (10+1e-8, 10, 10)")
    ax.set_yscale("log") #the log transformation
    if save:
        fig.savefig(filename + '.png')
    plt.show()

    return squared_dist