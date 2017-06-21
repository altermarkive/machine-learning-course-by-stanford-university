#!/usr/bin/env python3

import numpy as np

from sigmoid import sigmoid


def costFunctionReg(theta, X, y, lambda_value):
    #COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
    #   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
    #   theta as the parameter for regularized logistic regression and the
    #   gradient of the cost w.r.t. to the parameters.

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

    h = sigmoid(np.dot(X, theta))
    cost = np.sum(np.dot(-y.T, np.log(h)) - np.dot((1 - y.T), np.log(1 - h)))
    J = (cost / m) + (lambda_value / (2 * m)) * np.sum(theta[1:] ** 2)
    extra = lambda_value * theta
    extra[0] = 0
    grad = (np.dot(X.T, h - y) + np.reshape(extra, (extra.shape[0], 1))) / m

    # =============================================================

    return (J, grad)
