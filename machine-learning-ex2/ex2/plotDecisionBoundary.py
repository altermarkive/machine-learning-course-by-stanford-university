#!/usr/bin/env python3

import numpy as np
import matplotlib
# Force matplotlib to not use any X Windows backend (must be called befor importing pyplot)
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mapFeature import mapFeature
from plotData import plotData


def plotDecisionBoundary(theta, X, y, labels):
    #PLOTDECISIONBOUNDARY Plots the data points X and y into a new figure with
    #the decision boundary defined by theta
    #   PLOTDECISIONBOUNDARY(theta, X,y) plots the data points with + for the
    #   positive examples and o for the negative examples. X is assumed to be
    #   a either
    #   1) Mx3 matrix, where the first column is an all-ones column for the
    #      intercept.
    #   2) MxN, N>3 matrix, where the first column is all-ones

    # Plot Data
    pos_handle, neg_handle = plotData(X[:, 1:3], y, labels)
    #hold on

    if X.shape[1] <= 3:
        # Only need 2 points to define a line, so choose two endpoints
        plot_x = [np.min(X[:, 1]) - 2, np.max(X[:, 1]) + 2]

        # Calculate the decision boundary line
        plot_y = np.dot((-1.0 / theta[2]), (np.dot(theta[1], plot_x) + theta[0]))

        # Plot, and adjust axes for better viewing
        boundary_handle = plt.plot(plot_x, plot_y, label='Decision Boundary')[0]

        # Legend, specific for the exercise
        #axis([30, 100, 30, 100])
    else:
        # Here is the grid range
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)

        z = np.zeros((u.size, v.size))
        # Evaluate z = theta*x over the grid
        for i in range(u.size):
            for j in range(v.size):
                z[i, j] = np.dot(mapFeature(np.array([u[i]]), np.array([v[j]])), theta)
        z = z.T # important to transpose z before calling contour

        # Plot z = 0
        # Notice you need to specify the range [0, 0]
        u, v = np.meshgrid(u, v)
        boundary_handle = plt.contour(u, v, z, [0], linewidth=2).collections[0]
        boundary_handle.set_label('Decision Boundary')
    #hold off
    return (pos_handle, neg_handle, boundary_handle)
