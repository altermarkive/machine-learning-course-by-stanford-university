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


def costFunction(theta, X, y):
    #COSTFUNCTION Compute cost and gradient for logistic regression
    #   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
    #   parameter for logistic regression and the gradient of the cost
    #   w.r.t. to the parameters.

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
    #
    # Note: grad should have the same dimensions as theta
    #

    h = sigmoid(np.dot(X, theta.reshape(-1, 1)))
    J = np.sum(np.dot(-y.T, np.log(h)) - np.dot((1 - y.T), np.log(1 - h))) / m
    grad = np.dot(X.T, h - y) / m

    # =============================================================

    return (J, grad)

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


if __name__ == "__main__":
    ex2()
