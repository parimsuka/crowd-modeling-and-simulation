"""
Module containing various plotting methods.

Authors: Simon Blöchinger, Sena Korkut
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def plot_ped_paths(ped_paths: list, title=None) -> None:
    """
    Plots list of paths of pedestrians.

    :param ped_paths: List of paths of pedestrians.
    :param title: The title of the plot.
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


def plot_ped_speeds(ped_speeds: list, title=None) -> None:
    """
    Plots list of speeds of pedestrians.

    :param ped_speeds: List of speeds of pedestrians.
    :param title: The title of the plot.
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

    :param test_x: test data for curve fitting, mean spacing
    :param test_y: test data for curve fitting, speed
    :param x_model: x values for the curve fit
    :param y_fit: y values for the curve fit
    :param title: title of plot
    :param save_fig: save figure
    :param fig_name: name of figure to save

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


def plot_histogram(data, xlabel = None, ylabel = None, title = None):
    """
    Plots histograms of the data.

    :param data: data to plot
    :param title: title of plot
    :param xlabel: x label of plot
    :param ylabel: y label of plot

    :return: None
    """
    #plot histogtam of the data
    plt.figure(figsize=(20, 10))
    plt.hist(data, bins=100, density=True, alpha=0.6, color='g', edgecolor='black')
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.title(title)

    plt.show()
    

def plot_losses(train_losses, test_losses, all_models):
    """
    Plots the losses for different models.
    
    Args:
        train_losses (list): List of loss values for the train_losses models.
        test_losses (list): List of loss values for the test models.
        all_models (list): List of model configurations.

    Returns:
        None
    """
    x_axis = [str(models) for models in all_models]
    
    plt.figure(figsize=(10, 6))
    
    # Plot train_losses as gray line with "Training" label
    plt.plot(x_axis, train_losses, marker='o', color='gray', label='Training')
    
    # Plot test_losses as red line with "Test" label
    plt.plot(x_axis, test_losses, marker='o', color='red', label='Test')
    
    plt.xlabel("Models")
    plt.ylabel("Loss")
    plt.title("Losses for Different Models")
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    
    
def plot_losses_FD_NN(x_labels, y_FD, y_NN):
    """
    Plot MSE losses for FD and NN models.

    Args:
        x_labels (list): List of x-axis labels.
        y_FD (list): List of MSE losses for FD model.
        y_NN (list): List of MSE losses for NN model.

    Returns:
        None
    """
    plt.figure(figsize=(10, 6))
    plt.plot(x_labels, y_FD, color='red', label='FD')
    plt.plot(x_labels, y_NN, color='grey', label='NN')
    plt.xlabel('Labels')
    plt.ylabel('MSE')

    # Modify y-axis major ticks
    plt.yticks([0.020, 0.030, 0.040, 0.050])

    # Define minor ticks and draw a grid for them
    ax = plt.gca()
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.010))
    ax.yaxis.grid(True, linestyle='--', which='both', alpha=0.5)

    # Place legend at the bottom right without a frame
    plt.legend(frameon=False, loc='lower right')

    plt.show()
