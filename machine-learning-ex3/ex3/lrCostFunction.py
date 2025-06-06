#!/usr/bin/env python3

import numpy as np

from sigmoid import sigmoid


def lrCostFunction(theta, X, y, lambda_value):
    #LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
    #regularization
    #   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
    #   theta as the parameter for regularized logistic regression and the
    #   gradient of the cost w.r.t. to the parameters. 

    # Initialize some useful values
    m = y.size # number of training examples
    if y.dtype == bool:
        y = y.astype(int)

    # You need to return the following variables correctly 
    J = 0
    grad = np.zeros(theta.shape)

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta.
    #               You should set J to the cost.
    #               Compute the partial derivatives and set grad to the partial
    #               derivatives of the cost w.r.t. each parameter in theta
    #
    # Hint: The computation of the cost function and gradients can be
    #       efficiently vectorized. For example, consider the computation
    #
    #           sigmoid(X * theta)
    #
    #       Each row of the resulting matrix will contain the value of the
    #       prediction for that example. You can make use of this to vectorize
    #       the cost function and gradient computations. 
    #
    # Hint: When computing the gradient of the regularized cost function, 
    #       there're many possible vectorized solutions, but one solution
    #       looks like:
    #           grad = (unregularized gradient for logistic regression)
    #           temp = theta; 
    #           temp(1) = 0;   % because we don't add anything for j = 0  
    #           grad = grad + YOUR_CODE_HERE (using the temp variable)
    #

    h = sigmoid(np.dot(X, theta))
    cost = np.sum(np.dot(-y.T, np.log(h)) - np.dot((1 - y.T), np.log(1 - h)))
    J = (cost / m) + (lambda_value / (2 * m)) * np.sum(theta[1:] ** 2)
    extra = lambda_value * theta
    extra[0] = 0
    grad = (np.dot(X.T, h - y) + extra.T) / m
    return (J, grad)










    # =============================================================

    return (J, grad)

    #grad = grad(:);

    #end
