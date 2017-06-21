#!/usr/bin/env python3

import numpy as np
import matplotlib
# Force matplotlib to not use any X Windows backend (must be called befor importing pyplot)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.optimize as optimize

from sigmoid import sigmoid
from plotData import plotData
from mapFeature import mapFeature
from costFunctionReg import costFunctionReg
from plotDecisionBoundary import plotDecisionBoundary
from predict import predict


def costRegOnly(theta, X, y, lambda_value):
    m = y.shape[0]
    h = sigmoid(np.dot(X, theta))
    if np.any(h <= 0) or np.any(h >= 1):
        return np.nan
    cost = np.sum(np.dot(-y.T, np.log(h)) - np.dot((1 - y.T), np.log(1 - h)))
    J = (cost / m) + (lambda_value / (2 * m)) * np.sum(np.power(theta[1:], 2))
    return J


def gradientRegOnly(theta, X, y, lambda_value):
    m = y.shape[0]
    h = sigmoid(np.dot(X, theta.reshape(-1, 1)))
    extra = lambda_value * theta
    extra[0] = 0
    extra = extra.reshape(-1, 1)
    grad = (np.dot(X.T, h - y) + extra) / m
    return grad.flatten()


def ex2_reg():
    # Machine Learning Online Class - Exercise 2: Logistic Regression
    #
    #  Instructions
    #  ------------
    #
    #  This file contains code that helps you get started on the second part
    #  of the exercise which covers regularization with logistic regression.
    #
    #  You will need to complete the following functions in this exericse:
    #
    #     sigmoid.m
    #     costFunction.m
    #     predict.m
    #     costFunctionReg.m
    #
    #  For this exercise, you will not need to change any code in this file,
    #  or any other files other than those mentioned above.
    #

    # Initialization
    #clear ; close all; clc

    # Load Data
    #  The first two columns contains the X values and the third column
    #  contains the label (y).

    data = np.loadtxt('ex2data2.txt', delimiter=',')
    X = np.reshape(data[:, 0:2], (data.shape[0], 2))
    y = np.reshape(data[:, 2], (data.shape[0], 1))

    pos_handle, neg_handle = plotData(X, y, ['y = 1', 'y = 0'])

    # Put some labels
    #hold on;

    # Labels and Legend
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')

    # Specified in plot order
    plt.legend(handles=[pos_handle, neg_handle])
    plt.savefig('figure1.reg.png')


    # =========== Part 1: Regularized Logistic Regression ============
    #  In this part, you are given a dataset with data points that are not
    #  linearly separable. However, you would still like to use logistic
    #  regression to classify the data points.
    #
    #  To do so, you introduce more features to use -- in particular, you add
    #  polynomial features to our data matrix (similar to polynomial
    #  regression).
    #

    # Add Polynomial Features

    # Note that mapFeature also adds a column of ones for us, so the intercept
    # term is handled
    X = mapFeature(X[:, 0], X[:, 1])

    # Initialize fitting parameters
    initial_theta = np.zeros((X.shape[1], 1))

    # Set regularization parameter lambda to 1
    lambda_value = 1

    # Compute and display initial cost and gradient for regularized logistic
    # regression
    cost, grad = costFunctionReg(initial_theta, X, y, lambda_value)

    print('Cost at initial theta (zeros): %f' % cost)

    print('Program paused. Press enter to continue.')
    #input()

    # ============= Part 2: Regularization and Accuracies =============
    #  Optional Exercise:
    #  In this part, you will get to try different values of lambda and
    #  see how regularization affects the decision coundart
    #
    #  Try the following values of lambda (0, 1, 10, 100).
    #
    #  How does the decision boundary change when you vary lambda? How does
    #  the training set accuracy vary?
    #

    # Initialize fitting parameters
    initial_theta = np.zeros((X.shape[1], 1))

    # Set regularization parameter lambda to 1 (you should vary this)
    lambda_value = 1

    # Set Options
    #options = optimset('GradObj', 'on', 'MaxIter', 400);

    # Optimize
    result = optimize.minimize(costRegOnly, initial_theta, args=(X, y, lambda_value), method=None, jac=gradientRegOnly, options={"maxiter":400})
    theta = result.x

    # Plot Boundary
    pos_handle, neg_handle, boundary_handle = plotDecisionBoundary(theta, X, y, ['y = 1', 'y = 0'])
    plt.title('lambda = %g' % lambda_value)

    # Labels and Legend
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')

    plt.legend(handles=[pos_handle, neg_handle, boundary_handle])
    plt.savefig('figure2.reg.png')

    # Compute accuracy on our training set
    p = predict(theta, X)

    print('Train Accuracy: %f' % (np.mean((p == y.flatten()).astype(int)) * 100))
