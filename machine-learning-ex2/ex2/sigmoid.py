#!/usr/bin/env python3

import numpy as np


def sigmoid(z):
    #SIGMOID Compute sigmoid functoon
    #   J = SIGMOID(z) computes the sigmoid of z.

    # You need to return the following variables correctly
    #g = zeros(size(z));

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the sigmoid of each value of z (z can be a matrix,
    #               vector or scalar).


    g = np.divide(1, (1 + np.power(np.exp(1), -z)))

    # =============================================================

    return g
