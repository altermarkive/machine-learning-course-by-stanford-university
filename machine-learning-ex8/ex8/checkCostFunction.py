#!/usr/bin/env python3

import numpy as np

from computeNumericalGradient import computeNumericalGradient
from cofiCostFunc import cofiCostFunc


def checkCostFunction(lambda_value=0):
    #CHECKCOSTFUNCTION Creates a collaborative filering problem 
    #to check your cost function and gradients
    #   CHECKCOSTFUNCTION(lambda) Creates a collaborative filering problem 
    #   to check your cost function and gradients, it will output the 
    #   analytical gradients produced by your code and the numerical gradients 
    #   (computed using computeNumericalGradient). These two gradient 
    #   computations should result in very similar values.

    ## Create small problem
    X_t = np.random.rand(4, 3)
    Theta_t = np.random.rand(5, 3)

    # Zap out most entries
    Y = np.dot(X_t, Theta_t.T)
    Y[np.random.rand(*Y.shape) > 0.5] = 0
    R = np.zeros(Y.shape)
    R[Y != 0] = 1

    ## Run Gradient Checking
    X = np.random.randn(*X_t.shape)
    Theta = np.random.randn(*Theta_t.shape)
    num_movies, num_users = Y.shape
    num_features = Theta_t.shape[1]

    numgrad = computeNumericalGradient(
        lambda x: cofiCostFunc(x, Y, R, num_users, num_movies, num_features, lambda_value), np.concatenate([X.ravel(), Theta.ravel()]))

    cost, grad = cofiCostFunc(np.concatenate([X.ravel(), Theta.ravel()]), Y, R, num_users, num_movies, num_features, lambda_value)

    print(np.stack([numgrad, grad], axis=1))
    print('The above two columns you get should be very similar.\n(Left-Your Numerical Gradient, Right-Analytical Gradient)\n')

    diff = np.linalg.norm(numgrad - grad) / np.linalg.norm(numgrad + grad)
    print('If your cost function implementation is correct, then \nthe relative difference will be small (less than 1e-9).\nRelative Difference: %g' % diff)

    #end