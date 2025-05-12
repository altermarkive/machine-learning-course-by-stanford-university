#!/usr/bin/env python3

import numpy as np
import matplotlib
# Force matplotlib to not use any X Windows backend (must be called befor importing pyplot)
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from polyFeatures import polyFeatures


def plotFit(min_x, max_x, mu, sigma, theta, p):
    #PLOTFIT Plots a learned polynomial regression fit over an existing figure.
    #Also works with linear regression.
    #   PLOTFIT(min_x, max_x, mu, sigma, theta, p) plots the learned polynomial
    #   fit with power p and feature normalization (mu, sigma).

    # Hold on to the current figure
    #hold on;

    # We plot a range slightly bigger than the min and max values to get
    # an idea of how the fit will vary outside the range of the data points
    x = np.arange(min_x - 15, max_x + 25, 0.05)[None].T

    # Map the X values 
    X_poly = polyFeatures(x, p)
    X_poly -= mu
    X_poly /= sigma

    # Add ones
    X_poly = np.concatenate([np.ones((x.shape[0], 1)), X_poly], axis=1)

    # Plot
    plt.plot(x, np.dot(X_poly, theta), '--', lw=2)

    # Hold off to the current figure
    #hold off

    #end
