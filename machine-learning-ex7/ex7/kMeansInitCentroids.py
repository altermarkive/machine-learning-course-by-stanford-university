#!/usr/bin/env python3

import numpy as np


def kMeansInitCentroids(X, K):
    #KMEANSINITCENTROIDS This function initializes K centroids that are to be 
    #used in K-Means on the dataset X
    #   centroids = KMEANSINITCENTROIDS(X, K) returns K initial centroids to be
    #   used with the K-Means on the dataset X
    #

    # You should return this values correctly
    #centroids = zeros(K, size(X, 2))

    # ====================== YOUR CODE HERE ======================
    # Instructions: You should set centroids to randomly chosen examples from
    #               the dataset X
    #

    randidx = np.random.permutation(X.shape[0])
    centroids = X[randidx[:K], :]







    # =============================================================

    return centroids
    #end

