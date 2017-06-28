#!/usr/bin/env python3

import numpy as np
import matplotlib
# Force matplotlib to not use any X Windows backend (must be called befor importing pyplot)
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from plotDataPoints import plotDataPoints
from drawLine import drawLine


def plotProgresskMeans(X, centroids, previous, idx, K, i):
    #PLOTPROGRESSKMEANS is a helper function that displays the progress of 
    #k-Means as it is running. It is intended for use only with 2D data.
    #   PLOTPROGRESSKMEANS(X, centroids, previous, idx, K, i) plots the data
    #   points with colors assigned to each centroid. With the previous
    #   centroids, it also plots a line between the previous locations and
    #   current locations of the centroids.
    #

    # Plot the examples
    plotDataPoints(X, idx, K, i)

    current = centroids
    for last in previous[::-1]:
        # Plot the centroids as black x's
        plt.plot(current[:,0], current[:,1], linestyle='None', marker='x', markeredgecolor='k', ms=10, lw=3)

        # Plot the history of the centroids with lines
        for j in range(current.shape[0]):
            drawLine(current[j,:], last[j,:])
        current = last
    #end

    # Title
    plt.title('Iteration number %d' % i)

    #end

