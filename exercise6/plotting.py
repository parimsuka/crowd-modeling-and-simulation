"""
Plotting Methods

TODO: Finalize Docstring(s)

"""

import matplotlib.pyplot as plt
import numpy as np


def plot_ped_paths(ped_paths, title=None):
    """
    Plots list of paths of pedestrians.
    TODO: finalize Docstring

    :param ped_paths:
    :return:
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for ped_path in ped_paths:
        ax.plot(ped_path[:, 2], ped_path[:, 3], ped_path[:, 4])

    plt.title(title)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    plt.show()


def plot_ped_speeds(ped_speeds, title=None):
    """
    Plots list of speeds of pedestrians.
    TODO: finalize Docstring

    :param ped_speeds:
    :return:
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for ped_speed in ped_speeds:
        ax.plot(ped_speed[:, 2])

    plt.title(title)

    plt.show()


def plot_fd_curve(test_x, test_y, x_model, y_fit, title=None, save_fig=False, fig_name=None):
    """
    Plots a fundamental diagram curve.

    :param fd_curve: fundamental diagram curve
    :param title: title of plot
    :return: None
    """
    # Plot the data and the curve fit
    plt.figure(figsize=(20, 10))
    plt.scatter(test_x, test_y, marker='o', color = (0.2, 0.5, 0.5), label='Data', s=0.3)
    plt.plot(x_model, y_fit, label='Curve Fit', color= 'red', linewidth=2 )
    plt.xlabel("Mean Spacing")
    plt.ylabel("Speed")
    plt.title(title)

    plt.show()

    if save_fig:
        plt.savefig(fig_name)

    return None

