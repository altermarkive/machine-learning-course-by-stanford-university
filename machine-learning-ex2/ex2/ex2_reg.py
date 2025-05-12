#!/usr/bin/env python3

import numpy as np
import matplotlib
import matplotlib.cm as cm
# Force matplotlib to not use any X Windows backend (must be called befor importing pyplot)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import scipy.optimize as optimize
import sklearn.linear_model as linear_model
import sklearn.preprocessing as preprocessing

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

def mapFeature(X1, X2):
    # MAPFEATURE Feature mapping function to polynomial features
    #
    #   MAPFEATURE(X1, X2) maps the two input features
    #   to quadratic features used in the regularization exercise.
    #
    #   Returns a new feature array with more features, comprising of
    #   X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
    #
    #   Inputs X1, X2 must be the same size
    #

    # n = X1.shape[0]
    # degree = 6
    # out = np.ones((n, 1)).reshape((n, 1))
    # for i in range(1, degree + 1):
    #     for j in range(i + 1):
    #         term1 = X1 ** (i - j)
    #         term2 = X2 ** j
    #         out = np.hstack((out, (term1 * term2).reshape((n, 1))))

    data = np.c_[X1, X2]
    poly = preprocessing.PolynomialFeatures(6)
    out = poly.fit_transform(data)

    return out

def sigmoid(z):
    #SIGMOID Compute sigmoid functoon
    #   J = SIGMOID(z) computes the sigmoid of z.

    # You need to return the following variables correctly
    #g = zeros(size(z));

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the sigmoid of each value of z (z can be a matrix,
    #               vector or scalar).


    g = np.divide(1, (1 + np.power(np.exp(1), -z)))

    # =============================================================

    return g


def costFunctionReg(theta, X, y, lambda_value):
    #COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
    #   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
    #   theta as the parameter for regularized logistic regression and the
    #   gradient of the cost w.r.t. to the parameters.

    # Initialize some useful values
    m = y.shape[0] # number of training examples

    # You need to return the following variables correctly
    #J = 0;
    #grad = zeros(size(theta));

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta.
    #               You should set J to the cost.
    #               Compute the partial derivatives and set grad to the partial
    #               derivatives of the cost w.r.t. each parameter in theta

    h = sigmoid(np.dot(X, theta))
    cost = np.sum(np.dot(-y.T, np.log(h)) - np.dot((1 - y.T), np.log(1 - h)))
    J = (cost / m) + (lambda_value / (2 * m)) * np.sum(theta[1:] ** 2)
    extra = lambda_value * theta
    extra[0] = 0
    grad = (np.dot(X.T, h - y) + extra) / m

    # =============================================================

    return (J, grad)

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

def predict(theta, X):
    #PREDICT Predict whether the label is 0 or 1 using learned logistic
    #regression parameters theta
    #   p = PREDICT(theta, X) computes the predictions for X using a
    #   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

    m = X.shape[0] # Number of training examples

    # You need to return the following variables correctly
    #p = zeros(m, 1);

    # ====================== YOUR CODE HERE ======================
    # Instructions: Complete the following code to make predictions using
    #               your learned logistic regression parameters.
    #               You should set p to a vector of 0's and 1's
    #

    p = sigmoid(np.dot(X, theta)) >= 0.5

    return p.astype(int)


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

if __name__ == "__main__":
    ex2_reg()
