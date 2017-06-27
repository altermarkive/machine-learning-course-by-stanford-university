#!/usr/bin/env python3

import numpy as np
import matplotlib
# Force matplotlib to not use any X Windows backend (must be called befor importing pyplot)
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plotData(X, y):
    #PLOTDATA Plots the data points X and y into a new figure 
    #   PLOTDATA(x,y) plots the data points with + for the positive examples
    #   and o for the negative examples. X is assumed to be a Mx2 matrix.
    #
    # Note: This was slightly modified such that it expects y = 1 or y = 0

    # Find Indices of Positive and Negative Examples
    pos = y == 1
    neg = y == 0

    # Plot Examples
    plt.plot(X[pos, 0], X[pos, 1], marker='+', linestyle='None', lw=1, ms=7)
    #hold on;
    plt.plot(X[neg, 0], X[neg, 1], marker='o', linestyle='None', ms=7)
    #hold off;

    #end
