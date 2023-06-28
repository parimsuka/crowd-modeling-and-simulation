"""
Plotting Methods

TODO: Finalize Docstring(s)

"""

import matplotlib.pyplot as plt


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
