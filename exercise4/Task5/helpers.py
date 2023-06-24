import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def create_3d_subplot(fig, position=111, title="SIR trajectory"):
    """
    Creates a 3D subplot on a given figure.

    Parameters:
    - fig: the figure where the subplot will be added.
    - position: the position of the subplot on the figure.
    - title: the title of the subplot.
    """
    ax = fig.add_subplot(position, projection="3d")
    ax.set_xlabel("S")
    ax.set_ylabel("I")
    ax.set_zlabel("R")
    ax.set_title(title)
    return ax

def solve_and_plot(ax, model, initial_cond, time, parameters, color, cmap):
    """
    Solves the given model with the given initial condition and parameters, 
    and plots the result on the given axes.

    Parameters:
    - ax: the axes where the result will be plotted.
    - model: the model to be solved.
    - initial_cond: the initial conditions for the model.
    - time: the time span for the solution.
    - parameters: the parameters for the model.
    - color: the color for the plot.
    - cmap: the colormap for the scatter plot.
    """
    sol = solve_ivp(model, t_span=[time[0],time[-1]], y0=initial_cond, t_eval=time, args=parameters, method='DOP853', rtol=1e-8, atol=1e-8)
    ax.plot(sol.y[0], sol.y[1], sol.y[2], color+'-');
    ax.scatter(sol.y[0], sol.y[1], sol.y[2], s=1, c=time, cmap=cmap);

def plot_for_b_values(model, initial_conds, time, parameters, b_values, colors, cmap, size=(20,55)):
    """
    Creates a figure with a 3D plot for each value of b.

    Parameters:
    - model: the model to be solved.
    - initial_conds: the initial conditions for the model.
    - time: the time span for the solution.
    - parameters: the parameters for the model, with b being replaced for each value in b_values.
    - b_values: the values of b to use.
    - colors: the colors for the plots.
    - cmap: the colormap for the scatter plot.
    - size: the size of the figure.
    """
    fig = plt.figure(figsize=size)
    axes = fig.subplots(nrows=int(b_values.size/3), ncols=3, subplot_kw={'projection': '3d'})

    for i, b_val in enumerate(b_values):
        row = int(i/3)
        col = i % 3
        ax = axes[row, col]
        for ic, color in zip(initial_conds, colors):
            params = parameters[:-1] + (b_val,) # Replace the last parameter (b) with b_val
            solve_and_plot(ax, model, ic, time, params, color, cmap)
        ax.set_title(f"SIR trajectory b: {b_val:.3f}")
    fig.tight_layout()
    return fig


def plot_for_sim_points_and_b_values(model, sim_points, time, parameters, b_values, cmap, size=(20,55), specific_b_values=None):
    """
    Creates a figure with a 3D plot for each combination of simulation point and b value.

    Parameters:
    - model: the model to be solved.
    - sim_points: the simulation points for the model.
    - time: the time span for the solution.
    - parameters: the parameters for the model, with b being replaced for each value in b_values.
    - b_values: the values of b to use.
    - cmap: the colormap for the scatter plot.
    - size: the size of the figure.
    - specific_b_values: specific b_values for a specific plot.
    """
    if specific_b_values is None:
        specific_b_values = b_values
        nrows = b_values.size
    else:
        nrows = len(specific_b_values)

    fig = plt.figure(figsize=size)
    axes = fig.subplots(nrows=nrows, ncols=len(sim_points), subplot_kw={'projection': '3d'})

    for i, b_val in enumerate(specific_b_values):
        for j, sim in enumerate(sim_points):
            ax = axes[i, j]
            params = parameters[:-1] + (b_val,) # Replace the last parameter (b) with b_val
            solve_and_plot(ax, model, sim, time, params, 'r', cmap)
            S, I, R = sim[:]
            ax.set_title(f"S: {S}, I: {I}, R: {R}, b: {b_val:.3f}")
    fig.tight_layout()
    return fig
