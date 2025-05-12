#!/usr/bin/env python3

import numpy as np

from scipy import optimize
from scipy.optimize import fmin_cg

from lrCostFunction import lrCostFunction


def oneVsAll(X, y, num_labels, lambda_value, maxiter=50):
    #ONEVSALL trains multiple logistic regression classifiers and returns all
    #the classifiers in a matrix all_theta, where the i-th row of all_theta 
    #corresponds to the classifier for label i
    #   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
    #   logisitc regression classifiers and returns each of these classifiers
    #   in a matrix all_theta, where the i-th row of all_theta corresponds 
    #   to the classifier for label i

    # Some useful variables
    m, n = X.shape

    # You need to return the following variables correctly 
    all_theta = np.zeros((num_labels, n + 1))

    # Add ones to the X data matrix
    X = np.concatenate([np.ones((m, 1)), X], axis=1)

    # ====================== YOUR CODE HERE ======================
    # Instructions: You should complete the following code to train num_labels
    #               logistic regression classifiers with regularization
    #               parameter lambda. 
    #
    # Hint: theta(:) will return a column vector.
    #
    # Hint: You can use y == c to obtain a vector of 1's and 0's that tell use 
    #       whether the ground truth is true/false for this class.
    #
    # Note: For this assignment, we recommend using fmincg to optimize the cost
    #       function. It is okay to use a for-loop (for c = 1:num_labels) to
    #       loop over the different classes.
    #
    #       fmincg works similarly to fminunc, but is more efficient when we
    #       are dealing with large number of parameters.
    #
    # Example Code for fmincg:
    #
    #     % Set Initial theta
    #     initial_theta = zeros(n + 1, 1);
    #     
    #     % Set options for fminunc
    #     options = optimset('GradObj', 'on', 'MaxIter', 50);
    # 
    #     % Run fmincg to obtain the optimal theta
    #     % This function will return theta and the cost 
    #     [theta] = ...
    #         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
    #                 initial_theta, options);
    #

    for c in np.arange(num_labels):
        initial_theta = all_theta[c, :]
        options = {'maxiter': maxiter}
        result = optimize.minimize(
            lrCostFunction, initial_theta,
            (X, (y == c), lambda_value),
            jac=True, method='TNC', options=options) 
        all_theta[c] = result.x
    return all_theta


    # =========================================================================


    #end
