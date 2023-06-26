"""
Module implementing Utilityfunctions that are used for plotting in the linear_vectorfields_showcase Jupyter Notebook.

author: Simon Blöchinger
"""

import matplotlib.pyplot as plt


def point_plot(x, y, color=None, marker='.', title=None, label=None):
    """Plots a Scatterplot of points."""
    plt.scatter(x, y, color=color, marker=marker, label=label)
    plt.title(title)
    if label:
        plt.legend()


def arrow_plot(x, y, u, v, color_arrow=None, color_point=None, marker='.', title=None):
    """Plots a Scatterplot of points with vectors on top of the points."""
    point_plot(x, y, color_point, marker, title)
    plt.quiver(x, y, u, v, color=color_arrow)
    plt.title(title)


def stream_plot(x, y, u, v, sp_x, sp_y, title=None):
    """Plots a Streamplot with Trajectory. x,y,u,v are the streamplot parameters, sp_x,sp_y are the trajectory parameters."""
    plt.streamplot(x, y, u, v, color='cyan')
    plt.plot(sp_x, sp_y, color='red')
    plt.xlim([x.flatten()[0], x.flatten()[-1]])
    plt.ylim([y.flatten()[0], y.flatten()[-1]])
    plt.title(title)

