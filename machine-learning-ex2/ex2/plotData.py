#!/usr/bin/env python3

import numpy as np
import matplotlib
# Force matplotlib to not use any X Windows backend (must be called befor importing pyplot)
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plotData(X, y, labels):
    #PLOTDATA Plots the data points X and y into a new figure
    #   PLOTDATA(x,y) plots the data points with + for the positive examples
    #   and o for the negative examples. X is assumed to be a Mx2 matrix.

    # Create New Figure
    #figure; hold on;

    # ====================== YOUR CODE HERE ======================
    # Instructions: Plot the positive and negative examples on a
    #               2D plot, using the option 'k+' for the positive
    #               examples and 'ko' for the negative examples.
    #

    pos = np.nonzero(y == 1)
    neg = np.nonzero(y == 0)

    pos_handle = plt.plot(X[pos, 0], X[pos, 1], 'k+', linewidth=2, markersize=7, label=labels[0])[0]
    neg_handle = plt.plot(X[neg, 0], X[neg, 1], 'ko', markerfacecolor='y', markersize=7, label=labels[1])[0]

    # =========================================================================

    #hold off;
    return (pos_handle, neg_handle)
