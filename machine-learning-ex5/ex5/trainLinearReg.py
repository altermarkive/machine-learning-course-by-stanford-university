#!/usr/bin/env python3

import numpy as np

from scipy import optimize

from linearRegCostFunction import linearRegCostFunction


def trainLinearReg(X, y, lambda_value):
    #TRAINLINEARREG Trains linear regression given a dataset (X, y) and a
    #regularization parameter lambda
    #   [theta] = TRAINLINEARREG (X, y, lambda) trains linear regression using
    #   the dataset (X, y) and regularization parameter lambda. Returns the
    #   trained parameters theta.
    #

    # Initialize Theta
    initial_theta = np.zeros((X.shape[1], 1))

    # Create "short hand" for the cost function to be minimized
    costFunction = lambda t: linearRegCostFunction(X, y, t, lambda_value)

    # Now, costFunction is a function that takes in only one argument
    options = {'maxiter': 200}

    # Minimize using fmincg
    theta = optimize.minimize(costFunction, initial_theta, jac=True, method='TNC', options=options).x

    return theta
    #end
