#!/usr/bin/env python3

import numpy as np


def computeCostMulti(X, y, theta):
    #COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
    #   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
    #   parameter for linear regression to fit the data points in X and y

    # Initialize some useful values
    m = X.shape[0] # number of training examples

    # You need to return the following variables correctly
    #J = 0;

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta
    #               You should set J to the cost.

    J = np.sum(np.power(np.subtract(np.dot(X, theta), y), 2.0)) / (2 * m)

    # =========================================================================

    return J

