import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from plotting import plot_fd_curve


def weidmann_equation(x, v0, l, T):
    '''
    Weidmann's fundamental diagram equation

    :param x: mean spacing
    :param v0: desired speed
    :param l: pedestrian size
    :param T: time gap
    :return: speed
    '''

    return v0 * ( 1 - np.exp( (l - x) / (v0 * T) ) )


def calculate_mse(y, y_fit):
    '''
    Calculate the mean squared error between the data and the curve fit

    :param y: data
    :param y_fit: curve fit
    :return: mean squared error
    '''

    return np.mean((y - y_fit) ** 2)


def use_weidmann_method(train_x, train_y, test_x, test_y, initial_guess):
    '''
    Use Weidmann's method to calculate the fundamental diagram

    :param train_x: training data for optimizing the parameters, mean spacing 
    :param train_y: training data for optimizing the parameters, speed 
    :param test_x: test data for curve fitting, mean spacing
    :param test_y: test data for curve fitting, speed
    :param initial_guess: initial guess for the curve fit

    :return: optimized parameters, mean squared error, curve fit
    '''
    # Perform the curve fit
    optimized_parameters, _ = curve_fit(weidmann_equation, train_x, train_y, p0=initial_guess, maxfev=5000)

    # Generate curve fit based on optimized parameters
    y_fit = weidmann_equation(test_x, *optimized_parameters)

    # Calculate the mean squared error
    mse = calculate_mse(test_y, y_fit)

    return optimized_parameters, mse, y_fit 
