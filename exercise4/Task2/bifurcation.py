"""
Module implementing a bifurcation diagram plotting method

author: Simon Blöchinger
"""

import collections.abc as c_abc
import typing as t
import numpy.typing as npt

import matplotlib.pyplot as plt
import numpy as np


def plot_bifurcation_diagram(func_steady_state_stable: c_abc.Callable[[npt.ArrayLike], npt.ArrayLike],
                             func_steady_state_unstable: c_abc.Callable[[npt.ArrayLike], npt.ArrayLike],
                             alpha_range: (float, float),
                             alpha_stepsize: float = 0.001,
                             plot_title: t.Optional[str] = None) -> None:
    """
    Plots a bifurcation diagram of a system with a stable and an unstable steady state.

    :param func_steady_state_stable: A function returning the stable steady state of the system for a given alpha.
    :param func_steady_state_unstable: A function returning the unstable steady state of the system for a given alpha.
    :param alpha_range: A range of alpha to plot.
    :param alpha_stepsize: The stepsize used for plotting the alpha.
    :param plot_title: A title for the plot.
    """
    # generate alpha values in alpha_range with alpha_stepsize
    alphas = np.arange(alpha_range[0], alpha_range[1], alpha_stepsize)

    # initialize figure of any dimensions with one subplot
    fig = plt.figure()
    ax = plt.subplot(1, 1, 1)

    # Plot stable steady state
    ax.plot(alphas, func_steady_state_stable(alphas), 'k-', label='Stable steady state')

    # Plot unstable steady state
    ax.plot(alphas, func_steady_state_unstable(alphas), 'k-.', label='Unstable (saddle) steady state')

    # show legend
    ax.legend()

    # set plotted range in x-direction to the alpha_range given
    plt.xlim(alpha_range)

    # show plot_title if available
    if plot_title:
        plt.title(plot_title)

    plt.show()
