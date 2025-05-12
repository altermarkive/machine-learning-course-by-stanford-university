#!/usr/bin/env python3

import numpy as np


def gaussianKernel(x1, x2, sigma):
    #RBFKERNEL returns a radial basis function kernel between x1 and x2
    #   sim = gaussianKernel(x1, x2) returns a gaussian kernel between x1 and x2
    #   and returns the value in sim

    # Ensure that x1 and x2 are column vectors
    x1 = x1.ravel()
    x2 = x2.ravel()

    # You need to return the following variables correctly.
    #sim = 0;

    # ====================== YOUR CODE HERE ======================
    # Instructions: Fill in this function to return the similarity between x1
    #               and x2 computed using a Gaussian kernel with bandwidth
    #               sigma
    #
    #

    sim = np.exp(-np.sum((x1 - x2) ** 2.0) / (2.0 * (sigma ** 2.0)))
    




    # =============================================================

    return sim
    #end
