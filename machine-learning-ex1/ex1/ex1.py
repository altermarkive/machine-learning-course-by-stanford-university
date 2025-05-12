#!/usr/bin/env python3

import numpy as np
import matplotlib
import matplotlib.cm as cm
# Force matplotlib to not use any X Windows backend (must be called befor importing pyplot)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import sklearn.linear_model as linear_model

from warmUpExercise import warmUpExercise
from plotData import plotData
from computeCost import computeCost
from gradientDescent import gradientDescent

# Machine Learning Online Class - Exercise 1: Linear Regression
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions
#  in this exericse:
#
#     warmUpExercise.m
#     plotData.m
#     gradientDescent.m
#     computeCost.m
#     gradientDescentMulti.m
#     computeCostMulti.m
#     featureNormalize.m
#     normalEqn.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#
#  x refers to the population size in 10,000s
#  y refers to the profit in $10,000s

def ex1():
    # Initialization

    # ==================== Part 1: Basic Function ====================
    # Complete warmUpExercise.m
    print('Running warmUpExercise ...')
    print('5x5 Identity Matrix:')
    print(warmUpExercise())

    print('Program paused. Press enter to continue.\n')
    #input()

    # ======================= Part 2: Plotting =======================
    print('Plotting Data ...')
    data = np.loadtxt('ex1data1.txt', delimiter=',')
    X = np.reshape(data[:, 0], (data.shape[0], 1))
    y = np.reshape(data[:, 1], (data.shape[0], 1))
    m = y.shape[0] # number of training examples

    # Plot Data
    # Note: You have to complete the code in plotData.m
    plotData(X, y)

    print('Program paused. Press enter to continue.')
    #input()

    # =================== Part 3: Gradient descent ===================
    print('Running Gradient Descent ...')

    X = np.c_[np.ones((m, 1)), X] # Add a column of ones to x
    theta = np.reshape(np.zeros((2, 1)), (2, 1)) # initialize fitting parameters

    # Some gradient descent settings
    iterations = 1500
    alpha = 0.01

    # compute and display initial cost
    print(computeCost(X, y, theta))

    # run gradient descent
    theta, _ = gradientDescent(X, y, theta, alpha, iterations)

    # print theta to screen
    print('Theta found by gradient descent: ')
    print('%f %f ' % (theta[0, 0], theta[1, 0]))

    # Plot the linear fit
    #hold on; % keep previous plot visible
    plt.plot(X[:, 1], np.dot(X, theta), '-')
    plt.legend(['Training data', 'Linear regression'])
    plt.savefig('figure2.png') #hold off % don't overlay any more plots on this figure

    # Predict values for population sizes of 35,000 and 70,000
    predict1 = np.dot([1, 3.5], theta)[0]
    print('For population = 35,000, we predict a profit of %f' % (predict1 * 10000))
    predict2 = np.dot([1, 7], theta)[0]
    print('For population = 70,000, we predict a profit of %f' % (predict2 * 10000))

    print('Program paused. Press enter to continue.')
    #input()

    # ============= Part 4: Visualizing J(theta_0, theta_1) =============
    print('Visualizing J(theta_0, theta_1) ...')

    # Grid over which we will calculate J
    theta0_vals = np.linspace(-10, 10, 100)
    theta1_vals = np.linspace(-1, 4, 100)

    # initialize J_vals to a matrix of 0's
    J_vals = np.zeros((theta0_vals.shape[0], theta1_vals.shape[0]))

    # Fill out J_vals
    for i in range(theta0_vals.shape[0]):
        for j in range(theta1_vals.shape[0]):
            t = [[theta0_vals[i]], [theta1_vals[j]]]
            J_vals[i, j] = computeCost(X, y, t)

    # Because of the way meshgrids work in the surf command, we need to
    # transpose J_vals before calling surf, or else the axes will be flipped
    #J_vals = J_vals
    # Surface plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.view_init(25, -130)
    ax.plot_surface(np.tile(theta0_vals, (theta0_vals.shape[0], 1)).T, np.tile(theta1_vals, (theta1_vals.shape[0], 1)), J_vals, cmap=cm.jet, antialiased=True)
    plt.xlabel(r'$\theta_0$')
    plt.ylabel(r'$\theta_1$')
    plt.savefig('figure3.png')

    # Contour plot
    fig = plt.figure()
    # Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
    plt.contour(theta0_vals, theta1_vals, J_vals.T, np.logspace(-2, 3, 20), cmap=cm.jet)
    plt.xlabel(r'$\theta_0$')
    plt.ylabel(r'$\theta_1$')
    #hold on;
    plt.plot(theta[0, 0], theta[1, 0], 'rx', markersize=10, linewidth=2)
    plt.savefig('figure4.png')

    # Using sklearn
    X = np.reshape(data[:, 0], (data.shape[0], 1))
    #model = linear_model.LinearRegression()
    model = linear_model.Ridge(alpha=alpha, max_iter=iterations, solver='lsqr')
    model.fit(X, y)
    print('Theta found: ')
    print('%f %f ' % (model.intercept_[0], model.coef_[0, 0]))
    fig = plt.figure()
    plt.plot(X, y, 'rx', markersize=10)
    plt.ylabel('Profit in $10,000s')
    plt.xlabel('Population of City in 10,000s')
    plt.plot(X, model.predict(X), '-')
    plt.legend(['Training data', 'Linear regression'])
    plt.savefig('figure2.sklearn.png')
