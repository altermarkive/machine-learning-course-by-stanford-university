#!/usr/bin/env python3

import numpy as np
import matplotlib
# Force matplotlib to not use any X Windows backend (must be called befor importing pyplot)
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plotDataPoints(X, idx, K, i):
    #PLOTDATAPOINTS plots data points in X, coloring them so that those with the same
    #index assignments in idx have the same color
    #   PLOTDATAPOINTS(X, idx, K) plots data points in X, coloring them so that those 
    #   with the same index assignments in idx have the same color

    # Create palette
    cmap = plt.cm.rainbow

    # Plot the data
    plt.scatter(X[:, 0], X[:, 1], c=np.array(idx[i]), cmap=cmap, marker='o', s=8**2, lw=1)

    #end
