#!/usr/bin/env python3

import numpy as np


def linearKernel(x1, x2):
    #LINEARKERNEL returns a linear kernel between x1 and x2
    #   sim = linearKernel(x1, x2) returns a linear kernel between x1 and x2
    #   and returns the value in sim

    # Ensure that x1 and x2 are column vectors
    x1 = x1.ravel()
    x2 = x2.ravel()

    # Compute the kernel
    sim = np.dot(x1, x2)

    return sim
    #end