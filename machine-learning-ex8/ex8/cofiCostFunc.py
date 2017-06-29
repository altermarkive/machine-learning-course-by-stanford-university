#!/usr/bin/env python3

import numpy as np


def cofiCostFunc(params, Y, R, num_users, num_movies, num_features, lambda_value):
    #COFICOSTFUNC Collaborative filtering cost function
    #   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
    #   num_features, lambda) returns the cost and gradient for the
    #   collaborative filtering problem.
    #

    # Unfold the U and W matrices from params
    X = params[:num_movies*num_features].reshape(num_movies, num_features)
    Theta = params[num_movies*num_features:].reshape(num_users, num_features)

            
    # You need to return the following values correctly
    #J = 0
    X_grad = np.zeros(X.shape)
    Theta_grad = np.zeros(Theta.shape)

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost function and gradient for collaborative
    #               filtering. Concretely, you should first implement the cost
    #               function (without regularization) and make sure it is
    #               matches our costs. After that, you should implement the 
    #               gradient and use the checkCostFunction routine to check
    #               that the gradient is correct. Finally, you should implement
    #               regularization.
    #
    # Notes: X - num_movies  x num_features matrix of movie features
    #        Theta - num_users  x num_features matrix of user features
    #        Y - num_movies x num_users matrix of user ratings of movies
    #        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
    #            i-th movie was rated by the j-th user
    #
    # You should set the following variables correctly:
    #
    #        X_grad - num_movies x num_features matrix, containing the 
    #                 partial derivatives w.r.t. to each element of X
    #        Theta_grad - num_users x num_features matrix, containing the 
    #                     partial derivatives w.r.t. to each element of Theta
    #

    term = (np.dot(X, Theta.T) - Y) * R
    J = (1/2) * np.sum(np.sum(term ** 2)) + (lambda_value / 2) * (np.sum(np.sum(Theta ** 2)) + np.sum(np.sum(X ** 2)))
    X_grad = np.dot(term, Theta) + lambda_value * X
    Theta_grad = np.dot(X.T, term).T + lambda_value * Theta















    # =============================================================

    grad = np.concatenate([X_grad.ravel(), Theta_grad.ravel()])

    return (J, grad)
    #end

