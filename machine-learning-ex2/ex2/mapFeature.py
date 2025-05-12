#!/usr/bin/env python3

import numpy as np
import sklearn.preprocessing as preprocessing


def mapFeature(X1, X2):
    # MAPFEATURE Feature mapping function to polynomial features
    #
    #   MAPFEATURE(X1, X2) maps the two input features
    #   to quadratic features used in the regularization exercise.
    #
    #   Returns a new feature array with more features, comprising of
    #   X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
    #
    #   Inputs X1, X2 must be the same size
    #

    # n = X1.shape[0]
    # degree = 6
    # out = np.ones((n, 1)).reshape((n, 1))
    # for i in range(1, degree + 1):
    #     for j in range(i + 1):
    #         term1 = X1 ** (i - j)
    #         term2 = X2 ** j
    #         out = np.hstack((out, (term1 * term2).reshape((n, 1))))

    data = np.c_[X1, X2]
    poly = preprocessing.PolynomialFeatures(6)
    out = poly.fit_transform(data)

    return out
