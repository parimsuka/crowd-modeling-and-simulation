import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D


def andronov_hopf_bifurcation(t, x, alpha):
    x1, x2 = x
    return [alpha*x1 - x2 - x1*(x1**2 + x2**2), x1 + alpha*x2 - x2*(x1**2 + x2**2)]


def plot_bifurcation_field(alpha, y0, grid_precision, w=2):
    """
    Plots the Andronov-Hopf bifurcation field for a given value of alpha.
    Also solves and plots the trajectory given initial conditions y0 and a time grid precision.
    """
    # Define grid for phase space
    Y, X = np.mgrid[-w:w:100j, -w:w:100j]

    # Define the function for the differential equation system
    def f_ode(t, y): return andronov_hopf_bifurcation(t, y, alpha)

    # time range
    time = np.linspace(start=0, stop=100, num=grid_precision)

    # Solve the system with solve_ivp
    sol = solve_ivp(f_ode, (time[0], time[-1]), y0, t_eval=time)

    # Create UV field from X, Y grid
    UV = np.array([f_ode(0, [x, y]) for x, y in zip(X.ravel(), Y.ravel())]).T
    U = UV[0, :].reshape(X.shape)
    V = UV[1, :].reshape(X.shape)

    fig, ax0 = plt.subplots(figsize=(10, 10))

    # Plot the vector field
    ax0.streamplot(X, Y, U, V, density=2)
    ax0.set_title(f'Phase diagram for α={alpha}')

    # Then plot the trajectory over it
    ax0.plot(sol.y[0, :], sol.y[1, :], c='red', label='Trajectory')

    # Mark the y0 in the plot in green
    ax0.scatter(*y0, color='green', label='Start Position', s=100)

    # Prettify
    ax0.legend(loc='upper right')
    ax0.set_aspect(1)


def plot_bifurcation_field_multiple_starts(alpha, y0s, grid_precision, w=2):
    """
    Plots the Andronov-Hopf bifurcation field for a given value of alpha.
    Also solves and plots the trajectory given initial conditions y0s (a list of initial conditions) and a time grid precision.
    """
    # Define grid for phase space
    Y, X = np.mgrid[-w:w:100j, -w:w:100j]

    # Define the function for the differential equation system
    def f_ode(t, y): return andronov_hopf_bifurcation(t, y, alpha)

    # time range
    time = np.linspace(start=0, stop=100, num=grid_precision)

    # Create UV field from X, Y grid
    UV = np.array([f_ode(0, [x, y]) for x, y in zip(X.ravel(), Y.ravel())]).T
    U = UV[0, :].reshape(X.shape)
    V = UV[1, :].reshape(X.shape)

    fig, ax0 = plt.subplots(figsize=(10, 10))

    # Plot the vector field
    ax0.streamplot(X, Y, U, V, density=2)
    ax0.set_title(f'Phase diagram for α={alpha}')

    # Then solve the system for each initial condition and plot the trajectory over it
    for y0 in y0s:
        sol = solve_ivp(f_ode, (time[0], time[-1]), y0, t_eval=time)

        # For semi-transparent lines
        ax0.plot(sol.y[0, :], sol.y[1, :], label=f'Trajectory starting at {y0}', alpha=0.5)

        # For dashed lines
        # ax0.plot(sol.y[0, :], sol.y[1, :], label=f'Trajectory starting at {y0}', linestyle='--')

        # Plot the starting points
        ax0.scatter(*y0, color='purple', s=100)

    # Prettify
    ax0.legend(loc='upper right')
    ax0.set_aspect(1)


# elev and azim are the viewing angles of the 3D plot.
def plot_cusp_bifurcation(elev=0, azim=0, width=1):
    # Define the range of values for x and α2
    x = np.linspace(-width, width, 500)
    alpha_2 = np.linspace(-width, width, 500)

    # Create a grid of values for x and α2
    X, Alpha_2 = np.meshgrid(x, alpha_2)

    # Compute α1 for each point in the (X, Alpha_2) grid
    Alpha_1 = X**3 - Alpha_2 * X

    # Create a 3D plot with increased size
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Set the viewing angle
    ax.view_init(elev=elev, azim=azim)

    # Plot the surface
    ax.plot_surface(X, Alpha_1, Alpha_2, rstride=10, cstride=10)

    # Set the labels and their colors
    ax.set_xlabel('x', color='red', labelpad=10, fontsize=12)
    ax.set_ylabel('α1', color='green', labelpad=10, fontsize=12)
    ax.set_zlabel('α2', color='blue', labelpad=10, fontsize=12)

    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)
    plt.show()
