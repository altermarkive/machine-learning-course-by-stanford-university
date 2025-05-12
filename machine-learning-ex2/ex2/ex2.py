#!/usr/bin/env python3

import numpy as np
import matplotlib
# Force matplotlib to not use any X Windows backend (must be called befor importing pyplot)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.optimize as optimize

from sigmoid import sigmoid
from plotData import plotData
from costFunction import costFunction
from plotDecisionBoundary import plotDecisionBoundary
from predict import predict


def costOnly(theta, X, y):
    m = y.shape[0]
    h = sigmoid(np.dot(X, theta))
    if np.any(h <= 0) or np.any(h >= 1):
        return np.nan
    J = np.sum(np.dot(-y.T, np.log(h)) - np.dot((1 - y.T), np.log(1 - h))) / m
    return J


def gradientOnly(theta, X, y):
    m = y.shape[0]
    h = sigmoid(np.dot(X, theta.reshape(-1, 1)))
    grad = np.dot(X.T, h - y) / m
    return grad.flatten()


def ex2():
    # Machine Learning Online Class - Exercise 2: Logistic Regression
    #
    #  Instructions
    #  ------------
    #
    #  This file contains code that helps you get started on the logistic
    #  regression exercise. You will need to complete the following functions
    #  in this exericse:
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
    #  The first two columns contains the exam scores and the third column
    #  contains the label.

    data = np.loadtxt('ex2data1.txt', delimiter=',')
    X = np.reshape(data[:, 0:2], (data.shape[0], 2))
    y = np.reshape(data[:, 2], (data.shape[0], 1))

    # ==================== Part 1: Plotting ====================
    #  We start the exercise by first plotting the data to understand the
    #  the problem we are working with.

    print('Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.')

    pos_handle, neg_handle = plotData(X, y, ['Admitted', 'Not admitted'])

    # Put some labels
    #hold on;
    # Labels and Legend
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')

    # Specified in plot order
    plt.legend(handles=[pos_handle, neg_handle])
    plt.savefig('figure1.png')

    print('Program paused. Press enter to continue.')
    #input()


    # ============ Part 2: Compute Cost and Gradient ============
    #  In this part of the exercise, you will implement the cost and gradient
    #  for logistic regression. You neeed to complete the code in
    #  costFunction.m

    #  Setup the data matrix appropriately, and add ones for the intercept term
    m, n = X.shape

    # Add intercept term to x and X_test
    X = np.c_[np.ones((m, 1)), X]

    # Initialize fitting parameters
    initial_theta = np.zeros((n + 1, 1))

    # Compute and display initial cost and gradient
    cost, grad = costFunction(initial_theta, X, y)

    print('Cost at initial theta (zeros): %f' % cost)
    print('Gradient at initial theta (zeros):')
    print(grad)

    print('Program paused. Press enter to continue.')
    #input()


    # ============= Part 3: Optimizing using fminunc  =============
    #  In this exercise, you will use a built-in function (fminunc) to find the
    #  optimal parameters theta.

    #  Set options for fminunc
    #options = optimset('GradObj', 'on', 'MaxIter', 400);

    #  Run fminunc to obtain the optimal theta
    #  This function will return theta and the cost
    result = optimize.minimize(costOnly, initial_theta, args=(X, y), method=None, jac=gradientOnly, options={'maxiter':400})
    theta = result.x
    cost = costOnly(theta, X, y)

    # Print theta to screen
    print('Cost at theta found by fminunc: %f' % cost)
    print('theta:')
    print(theta)

    # Plot Boundary
    pos_handle, neg_handle, boundary_handle = plotDecisionBoundary(theta, X, y, ['Admitted', 'Not admitted'])

    # Put some labels
    #hold on;
    # Labels and Legend
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')

    # Specified in plot order
    plt.legend(handles=[pos_handle, neg_handle, boundary_handle])
    plt.savefig('figure2.png')

    print('Program paused. Press enter to continue.')
    #input()

    # ============== Part 4: Predict and Accuracies ==============
    #  After learning the parameters, you'll like to use it to predict the outcomes
    #  on unseen data. In this part, you will use the logistic regression model
    #  to predict the probability that a student with score 45 on exam 1 and
    #  score 85 on exam 2 will be admitted.
    #
    #  Furthermore, you will compute the training and test set accuracies of
    #  our model.
    #
    #  Your task is to complete the code in predict.m

    #  Predict probability for a student with score 45 on exam 1
    #  and score 85 on exam 2

    prob = sigmoid(np.dot([1, 45, 85], theta))
    print('For a student with scores 45 and 85, we predict an admission probability of %f' % prob)

    # Compute accuracy on our training set
    p = predict(theta, X)

    print('Train Accuracy: %f' % (np.mean((p == y.flatten()).astype(int)) * 100))

    print('Program paused. Press enter to continue.')
    #input()
