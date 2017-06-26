#!/usr/bin/env python3

import numpy as np


def linearRegCostFunction(X, y, theta, lambda_value):
    #LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
    #regression with multiple variables
    #   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
    #   cost of using theta as the parameter for linear regression to fit the 
    #   data points in X and y. Returns the cost in J and the gradient in grad

    # Initialize some useful values
    m = y.size # number of training examples

    # You need to return the following variables correctly 
    #J = 0;
    #grad = zeros(size(theta));

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost and gradient of regularized linear 
    #               regression for a particular choice of theta.
    #
    #               You should set J to the cost and grad to the gradient.
    #

    n = theta.size
    h = np.dot(X, theta)
    J = np.sum((h - y) ** 2) / (2 * m)
    J = J + lambda_value * np.sum(theta[1:] ** 2) / (2 * m)
    extra = lambda_value * theta
    extra[0] = 0
    grad = (np.dot(X.T, (h - y)) + extra) / m











    # =========================================================================

    grad = grad.ravel()

    return (J, grad)
    #end
