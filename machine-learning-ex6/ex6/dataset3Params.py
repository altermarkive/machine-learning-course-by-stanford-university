#!/usr/bin/env python3

import numpy as np

from gaussianKernel import gaussianKernel
from svmTrain import svmTrain
from svmPredict import svmPredict


def dataset3Params(X, y, Xval, yval):
    #EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
    #where you select the optimal (C, sigma) learning parameters to use for SVM
    #with RBF kernel
    #   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
    #   sigma. You should complete this function to return the optimal C and 
    #   sigma based on a cross-validation set.
    #

    # You need to return the following variables correctly.
    #C = 1;
    #sigma = 0.3;

    # ====================== YOUR CODE HERE ======================
    # Instructions: Fill in this function to return the optimal C and sigma
    #               learning parameters found using the cross validation set.
    #               You can use svmPredict to predict the labels on the cross
    #               validation set. For example, 
    #                   predictions = svmPredict(model, Xval);
    #               will return the predictions on the cross validation set.
    #
    #  Note: You can compute the prediction error using 
    #        mean(double(predictions ~= yval))
    #

    values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    error = 10000
    for i in range(len(values)):
        C_val = values[i]
        for j in range(len(values)):
            sigma_val = values[j]
            model = svmTrain(X, y, C_val, gaussianKernel, args=(sigma_val,))
            predictions = svmPredict(model, Xval)
            error_val = np.mean(predictions != yval)
            if error_val < error:
                error, C, sigma = error_val, C_val, sigma_val\






    # =========================================================================

    return (C, sigma)
    #end
