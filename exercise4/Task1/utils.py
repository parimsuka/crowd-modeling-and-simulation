import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from numpy import linalg as LA

from scipy.integrate import solve_ivp

def plot_A_field(alpha, y0, grid_precision, stop_time):
    A = np.array([
        [alpha, alpha],
        [-0.25, 0]
    ])
    
    w = 1
    Y, X = np.mgrid[-w:w:100j, -w:w:100j]

    # Define the function for the differential equation system
    f_ode = lambda t, y: A@y

    # time range
    time = np.linspace(start=0, stop=stop_time, num=grid_precision)

    # Solve the system with solve_ivp
    sol = solve_ivp(f_ode, (time[0], time[-1]), y0, t_eval=time)

    # example linear vector field A*x
    ax0 = plot_phase_portrait(A, X, Y, f'$\\alpha={alpha}$')

    # then plot the trajectory over it
    ax0.plot(sol.y[0, :], sol.y[1, :], c='red', label='Trajectory')

    # mark the y0 in the plot in green
    ax0.scatter(*y0, color='green', label = 'Start Position', s=100)

    # prettify
    ax0.legend(loc='upper right')
    ax0.set_aspect(1)


def plot_phase_portrait(A, X, Y, title):
    """
    Plots a linear vector field in a streamplot, defined with X and Y coordinates and the matrix A.
    """
    UV = A@np.row_stack([X.ravel(), Y.ravel()])
    U = UV[0,:].reshape(X.shape)
    V = UV[1,:].reshape(X.shape)

    fig = plt.figure(figsize=(25, 25))
    gs = gridspec.GridSpec(nrows=3, ncols=2, height_ratios=[1, 1, 2])

    #  Varying density along a streamline
    ax0 = fig.add_subplot(gs[0, 0])
    # ax0.streamplot(X, Y, U, V, density=[0.5, 1])
    ax0.streamplot(X, Y, U, V, density=2)
    ax0.set_title(title);
    ax0.set_aspect(1)
    return ax0


def plot_minus_A_field(alpha, y0, grid_precision):
    A = np.array([
        [-alpha, -alpha],
        [0.25, 0]
    ])
    
    w = 1
    Y, X = np.mgrid[-w:w:100j, -w:w:100j]

    # Define the function for the differential equation system
    f_ode = lambda t, y: A@y

    # time range
    time = np.linspace(start=0, stop=100, num=grid_precision)

    # Solve the system with solve_ivp
    sol = solve_ivp(f_ode, (time[0], time[-1]), y0, t_eval=time)

    # example linear vector field A*x
    ax0 = plot_phase_portrait(A, X, Y, f'$\\alpha={alpha}$')

    # then plot the trajectory over it
    ax0.plot(sol.y[0, :], sol.y[1, :], c='red', label='Trajectory')

    # mark the y0 in the plot in green
    ax0.scatter(*y0, color='green', label = 'Start Position', s=100)

    # prettify
    ax0.legend(loc='upper right')
    ax0.set_aspect(1)


def plot_vectorfield(A, title):
    w = 1
    Y, X = np.mgrid[-w:w:100j, -w:w:100j]
    eigenvalues = LA.eigvals(A)

    UV = A@np.row_stack([X.ravel(), Y.ravel()])
    U = UV[0,:].reshape(X.shape)
    V = UV[1,:].reshape(X.shape)

    fig = plt.figure(figsize=(25, 25))
    gs = gridspec.GridSpec(nrows=3, ncols=2, height_ratios=[1, 1, 2])

    #  Varying density along a streamline
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.streamplot(X, Y, U, V, density=2)
    ax0.set_title(f'{title}, eigenvalues: {eigenvalues}')
    ax0.set_aspect(1)
    return ax0


def plot_eigenvalues(eigenvalues):
    for i, eigenvalue_set in enumerate(eigenvalues):
        # Convert the eigenvalue set into a numpy array
        eigenvalue_set = np.array(eigenvalue_set)

        fig, ax = plt.subplots(figsize=(6, 6))

        # Scatter plot with real part on the x-axis and imaginary part on the y-axis
        ax.scatter(eigenvalue_set.real, eigenvalue_set.imag, label=f'Set {i+1}')

        # Line plot to connect the points
        ax.plot(eigenvalue_set.real, eigenvalue_set.imag, 'r--')

        # Draw x and y axis
        ax.axhline(0, color='black',linewidth=0.5)
        ax.axvline(0, color='black',linewidth=0.5)
        ax.grid(color = 'gray', linestyle = '--', linewidth = 0.5)

        # Grid size and label
        ax.set_xlim([-1.8, 1.8])
        ax.set_ylim([-1.2, 1.2])
        ax.set_xlabel("Real Part")
        ax.set_ylabel("Imaginary Part")

        # Show plot
        ax.set_title(f"Eigenvalues on the Complex Plane for Set {i+1}")
        ax.legend()
        plt.show()

