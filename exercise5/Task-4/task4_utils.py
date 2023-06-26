
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def get_manifold_data(data_path) -> np.ndarray:
    """
    Read data from file and return as numpy array

    Parameters
    ----------
    data_path : str
        Path to the data file

    Returns
    -------
    numpy.ndarray
        Data as numpy array
    """
    data = pd.read_csv(data_path,sep=" ")
    data = data.to_numpy()
    return data


def add_delay(data, delta_n):
    """
    Add delay to the data

    Parameters
    ----------
    data : numpy.ndarray
        Data to be delayed
    delta_n : int
        Delay to be added

    Returns
    -------
    numpy.ndarray
        Delayed data
    """

    # Add delay to the data
    x = data[:-delta_n*2]
    y = data[delta_n:-delta_n]
    z = data[2*delta_n:]

    return x, y, z


def plot_2d(x, y, save = False, figure_name = '', title = 'title', xlabel = 'x', ylabel = 'y'):
    """
    Plot 2D data

    Parameters
    ----------
    x : numpy.ndarray
        x-axis data
    y : numpy.ndarray
        y-axis data
    save : bool, optional
        Save figure or not, by default False
    figure_name : str, optional
        Figure name, by default ''
    """
    plt.figure()
    plt.plot(x, y)
    plt.xlabel(xlabel=xlabel)
    plt.ylabel(ylabel=ylabel)
    plt.title(label=title)

    if save:
        plt.savefig(figure_name)
    
    plt.show()



def plot_3d(x, y, z, save = False, figure_name = '', title = 'title', xlabel = 'x', ylabel = 'y', zlabel = 'z'):
    """
    Plot 3D data

    Parameters
    ----------
    x : numpy.ndarray
        x-axis data
    y : numpy.ndarray   
        y-axis data
    z : numpy.ndarray
        z-axis data
    save : bool, optional
        Save figure or not, by default False
    figure_name : str, optional
        Figure name, by default ''
    """
    fig = plt.figure()
    #figure size 
    fig = plt.figure(figsize=(6, 7))
    ax = fig.add_subplot(projection='3d')
    ax.plot(x, y, z)
    ax.set_xlabel(xlabel=xlabel)
    ax.set_ylabel(ylabel=ylabel)
    ax.set_zlabel(zlabel=zlabel)
    ax.set_title(label=title)
    if save:
        plt.savefig(figure_name)
    
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

    Returns
    -------
    x_list, y_list, z_list : list
        The x, y, and z values of the trajectory.
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

    Returns
    -------
    None
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

