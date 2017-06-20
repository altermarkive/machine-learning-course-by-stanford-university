#!/usr/bin/env python3

import numpy as np


def normalEqn(X, y):
    #NORMALEQN Computes the closed-form solution to linear regression
    #   NORMALEQN(X,y) computes the closed-form solution to linear
    #   regression using the normal equations.

    #theta = zeros(size(X, 2), 1);

    # ====================== YOUR CODE HERE ======================
    # Instructions: Complete the code to compute the closed form solution
    #               to linear regression and put the result in theta.
    #

    # ---------------------- Sample Solution ----------------------

    theta = np.dot(np.dot(np.linalg.pinv(np.dot(X.T, X)), X.T), y)

    # -------------------------------------------------------------

    # ============================================================

    return theta
