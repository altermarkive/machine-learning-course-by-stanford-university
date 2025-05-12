#!/usr/bin/env python3

import numpy as np
import matplotlib
import matplotlib.cm as cm
# Force matplotlib to not use any X Windows backend (must be called befor importing pyplot)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import sklearn.linear_model as linear_model


def warmUpExercise():
    #WARMUPEXERCISE Example function in octave
    #   A = WARMUPEXERCISE() is an example function that returns the 5x5 identity matrix

    #A = [];
    # ============= YOUR CODE HERE ==============
    # Instructions: Return the 5x5 identity matrix
    #               In octave, we return values by defining which variables
    #               represent the return values (at the top of the file)
    #               and then set them accordingly.

    A = np.eye(5)

    # ===========================================

    return A

def plotData(x, y):
    #PLOTDATA Plots the data points x and y into a new figure
    #   PLOTDATA(x,y) plots the data points and gives the figure axes labels of
    #   population and profit.

    # ====================== YOUR CODE HERE ======================
    # Instructions: Plot the training data into a figure using the
    #               "figure" and "plot" commands. Set the axes labels using
    #               the "xlabel" and "ylabel" commands. Assume the
    #               population and revenue data have been passed in
    #               as the x and y arguments of this function.
    #
    # Hint: You can use the 'rx' option with plot to have the markers
    #       appear as red crosses. Furthermore, you can make the
    #       markers larger by using plot(..., 'rx', 'MarkerSize', 10);

    #figure; % open a new figure window

    plt.plot(x, y, 'rx', markersize=10)
    plt.ylabel('Profit in $10,000s')
    plt.xlabel('Population of City in 10,000s')
    plt.savefig('figure1.png')

    # ============================================================

def computeCost(X, y, theta):
    #COMPUTECOST Compute cost for linear regression
    #   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
    #   parameter for linear regression to fit the data points in X and y

    # Initialize some useful values
    #m = y.shape[1] # number of training examples

    # You need to return the following variables correctly
    #J = 0;

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta
    #               You should set J to the cost.

    J = np.sum(np.power(np.subtract(np.dot(X, theta), y), 2.0)) / (2 * X.shape[0])

    # =========================================================================

    return J

def gradientDescent(X, y, theta, alpha, num_iters):
    #GRADIENTDESCENT Performs gradient descent to learn theta
    #   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by
    #   taking num_iters gradient steps with learning rate alpha

    # Initialize some useful values
    m = y.shape[0] # number of training examples
    J_history = np.reshape(np.zeros((num_iters, 1)), (num_iters, 1))

    for i in range(num_iters):

        # ====================== YOUR CODE HERE ======================
        # Instructions: Perform a single gradient step on the parameter vector
        #               theta.
        #
        # Hint: While debugging, it can be useful to print out the values
        #       of the cost function (computeCost) and gradient here.
        #

        theta = np.subtract(theta, (alpha / m) * np.dot(np.subtract(np.dot(X, theta), y).T, X).T)

        # ============================================================

        # Save the cost J in every iteration
        J_history[i, 0] = computeCost(X, y, theta)

    return (theta, J_history)


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
    regr = linear_model.LinearRegression()
    regr.fit(X, y)
    fig = plt.figure()
    plt.plot(X, y, 'rx', markersize=10)
    plt.ylabel('Profit in $10,000s')
    plt.xlabel('Population of City in 10,000s')
    plt.plot(X, regr.predict(X), '-')
    plt.legend(['Training data', 'Linear regression'])
    plt.savefig('figure2.sklearn.png')

if __name__ == "__main__":
    ex1()
