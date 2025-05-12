#!/usr/bin/env python3

import numpy as np

from sigmoid import sigmoid


def costFunction(theta, X, y):
    #COSTFUNCTION Compute cost and gradient for logistic regression
    #   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
    #   parameter for logistic regression and the gradient of the cost
    #   w.r.t. to the parameters.

    # Initialize some useful values
    m = y.shape[0] # number of training examples

    # You need to return the following variables correctly
    #J = 0;
    #grad = zeros(size(theta));

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta.
    #               You should set J to the cost.
    #               Compute the partial derivatives and set grad to the partial
    #               derivatives of the cost w.r.t. each parameter in theta
    #
    # Note: grad should have the same dimensions as theta
    #

    h = sigmoid(np.dot(X, theta.reshape(-1, 1)))
    J = np.sum(np.dot(-y.T, np.log(h)) - np.dot((1 - y.T), np.log(1 - h))) / m
    grad = np.dot(X.T, h - y) / m

    # =============================================================

    return (J, grad)
