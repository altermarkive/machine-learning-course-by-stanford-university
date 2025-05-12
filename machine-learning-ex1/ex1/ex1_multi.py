#!/usr/bin/env python3

import numpy as np
import matplotlib
# Force matplotlib to not use any X Windows backend (must be called befor importing pyplot)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn.linear_model as linear_model
import sklearn.metrics as metrics

from featureNormalize import featureNormalize
from gradientDescentMulti import gradientDescentMulti
from normalEqn import normalEqn


# Machine Learning Online Class
#  Exercise 1: Linear regression with multiple variables
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  linear regression exercise.
#
#  You will need to complete the following functions in this
#  exericse:
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
#  For this part of the exercise, you will need to change some
#  parts of the code below for various experiments (e.g., changing
#  learning rates).
#

def ex1_multi():
    # Initialization

    # ================ Part 1: Feature Normalization ================

    # Clear and Close Figures
    #clear ; close all; clc

    print('Loading data ...')

    # Load Data
    data = np.loadtxt('ex1data2.txt', delimiter=',')
    X = np.reshape(data[:, 0:2], (data.shape[0], 2))
    y = np.reshape(data[:, 2], (data.shape[0], 1))
    m = y.shape[0]

    # Print out some data points
    print('First 10 examples from the dataset: ')
    print(np.c_[X[0:10, :], y[0:10, :]].T)

    print('Program paused. Press enter to continue.')
    #input()

    # Scale features and set them to zero mean
    print('Normalizing Features ...')

    X, mu, sigma = featureNormalize(X)

    # Add intercept term to X
    X = np.c_[np.ones((m, 1)), X]


    # ================ Part 2: Gradient Descent ================

    # ====================== YOUR CODE HERE ======================
    # Instructions: We have provided you with the following starter
    #               code that runs gradient descent with a particular
    #               learning rate (alpha).
    #
    #               Your task is to first make sure that your functions -
    #               computeCost and gradientDescent already work with
    #               this starter code and support multiple variables.
    #
    #               After that, try running gradient descent with
    #               different values of alpha and see which one gives
    #               you the best result.
    #
    #               Finally, you should complete the code at the end
    #               to predict the price of a 1650 sq-ft, 3 br house.
    #
    # Hint: By using the 'hold on' command, you can plot multiple
    #       graphs on the same figure.

    # Hint: At prediction, make sure you do the same feature normalization.


    # Begin: My code plotting for different learning rates
    alphas = [0.3, 0.1, 0.03, 0.01]
    colors = ['r', 'g', 'b', 'k']
    short_iters = 50
    fig = plt.figure()
    #hold on;
    plt.xlabel('Number of iterations')
    plt.ylabel('Cost J')
    for i in range(len(alphas)):
        _, J = gradientDescentMulti(X, y, np.reshape(np.zeros((3, 1)), (3, 1)), alphas[i], short_iters)
        plt.plot(range(len(J)), J, colors[i], markersize=2)
    plt.savefig('figure1.multi.png')
    # End: My code plotting for different learning rates

    print('Running gradient descent ...')

    # Choose some alpha value
    alpha = 0.01
    num_iters = 400

    # Init Theta and Run Gradient Descent
    theta = np.reshape(np.zeros((3, 1)), (3, 1))
    theta, J_history = gradientDescentMulti(X, y, theta, alpha, num_iters)

    # Plot the convergence graph
    fig = plt.figure()
    plt.plot(range(len(J_history)), J_history, '-b', markersize=2)
    plt.xlabel('Number of iterations')
    plt.ylabel('Cost J')
    plt.savefig('figure2.multi.png')

    # Display gradient descent's result
    print('Theta computed from gradient descent: ')
    print(theta)
    print()

    # Estimate the price of a 1650 sq-ft, 3 br house
    # ====================== YOUR CODE HERE ======================
    # Recall that the first column of X is all-ones. Thus, it does
    # not need to be normalized.
    #price = 0; % You should change this

    price = np.dot(np.r_[1, np.divide(np.subtract([1650, 3], mu), sigma)], theta)

    # ============================================================

    print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):\n $%f' % price)

    print('Program paused. Press enter to continue.')
    #input()

    # ================ Part 3: Normal Equations ================

    print('Solving with normal equations...')

    # ====================== YOUR CODE HERE ======================
    # Instructions: The following code computes the closed form
    #               solution for linear regression using the normal
    #               equations. You should complete the code in
    #               normalEqn.m
    #
    #               After doing so, you should complete this code
    #               to predict the price of a 1650 sq-ft, 3 br house.
    #

    # Load Data
    data = np.loadtxt('ex1data2.txt', delimiter=',')
    X = np.reshape(data[:, 0:2], (data.shape[0], 2))
    y = np.reshape(data[:, 2], (data.shape[0], 1))
    m = y.shape[0]

    # Add intercept term to X
    X = np.c_[np.ones((m, 1)), X]

    # Calculate the parameters from the normal equation
    theta = normalEqn(X, y)

    # Display normal equation's result
    print('Theta computed from the normal equations: ')
    print(theta)
    print('')


    # Estimate the price of a 1650 sq-ft, 3 br house
    # ====================== YOUR CODE HERE ======================
    price = np.dot([1, 1650, 3], theta) # You should change this


    # ============================================================

    print('Predicted price of a 1650 sq-ft, 3 br house (using normal equations):\n $%f' % price)

    # http://scikit-learn.org/stable/auto_examples/linear_model/plot_ridge_coeffs.html
    # Using sklearn
    X = np.reshape(data[:, 0:2], (data.shape[0], 2))
    y = np.reshape(data[:, 2], (data.shape[0], 1))
    model = linear_model.Ridge(max_iter=num_iters, solver='lsqr')
    count = 200
    alphas = np.logspace(-3, 1, count)
    coefs = np.zeros((count, 2))
    errors = np.zeros((count, 1))
    for i, alpha in enumerate(alphas):
        model.set_params(alpha=alpha)
        model.fit(X, y)
        coefs[i, :] = model.coef_
        errors[i, 0] = metrics.mean_squared_error(model.predict(X), y)
    results = [(r'$\theta_1$', coefs[:, 0]), (r'$\theta_2$', coefs[:, 1]), ('MSE', errors)]
    for i, result in enumerate(results):
        label, values = result
        plt.figure()
        ax = plt.gca()
        ax.set_xscale('log')
        ax.plot(alphas, values)
        plt.xlabel(r'$\alpha$')
        plt.ylabel(label)
        plt.savefig('figure%d.multi.sklearn.png' % (i + 1))
    #model = linear_model.LinearRegression()
    model = linear_model.Ridge(alpha=alpha, max_iter=num_iters, solver='lsqr')
    model.fit(X, y)
    print('Theta found: ')
    print('%f %f %f' % (model.intercept_[0], model.coef_[0, 0], model.coef_[0, 1]))
    print('Predicted price of a 1650 sq-ft, 3 br house (using sklearn):\n $%f' % model.predict([[1650, 3]]))
