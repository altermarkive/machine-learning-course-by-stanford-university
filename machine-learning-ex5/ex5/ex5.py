#!/usr/bin/env python3

import numpy as np
import scipy.io
import matplotlib
# Force matplotlib to not use any X Windows backend (must be called befor importing pyplot)
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from linearRegCostFunction import linearRegCostFunction
from trainLinearReg import trainLinearReg
from learningCurve import learningCurve
from polyFeatures import polyFeatures
from featureNormalize import featureNormalize
from plotFit import plotFit
from validationCurve import validationCurve
from lib.submitWithConfiguration import formatter


def ex5():
    ## Machine Learning Online Class
    #  Exercise 5 | Regularized Linear Regression and Bias-Variance
    #
    #  Instructions
    #  ------------
    # 
    #  This file contains code that helps you get started on the
    #  exercise. You will need to complete the following functions:
    #
    #     linearRegCostFunction.m
    #     learningCurve.m
    #     validationCurve.m
    #
    #  For this exercise, you will not need to change any code in this file,
    #  or any other files other than those mentioned above.
    #

    ## Initialization
    #clear ; close all; clc

    ## =========== Part 1: Loading and Visualizing Data =============
    #  We start the exercise by first loading and visualizing the dataset. 
    #  The following code will load the dataset into your environment and plot
    #  the data.
    #

    # Load Training Data
    print('Loading and Visualizing Data ...')

    # Load from ex5data1: 
    # You will have X, y, Xval, yval, Xtest, ytest in your environment
    mat = scipy.io.loadmat('ex5data1.mat')
    X = mat['X']
    y = mat['y'].ravel()
    Xval = mat['Xval']
    yval = mat['yval'].ravel()
    Xtest = mat['Xtest']
    ytest = mat['ytest'].ravel()

    # m = Number of examples
    m = X.shape[0]

    # Plot training data
    plt.plot(X, y, marker='x', linestyle='None', ms=10, lw=1.5)
    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')
    plt.savefig('figure1.png')

    print('Program paused. Press enter to continue.')
    #pause;

    ## =========== Part 2: Regularized Linear Regression Cost =============
    #  You should now implement the cost function for regularized linear 
    #  regression. 
    #

    theta = np.array([1, 1])
    J, _ = linearRegCostFunction(np.concatenate([np.ones((m, 1)), X], axis=1), y, theta, 1)

    print('Cost at theta = [1 ; 1]: %f \n(this value should be about 303.993192)' % J)

    print('Program paused. Press enter to continue.')
    #pause;

    ## =========== Part 3: Regularized Linear Regression Gradient =============
    #  You should now implement the gradient for regularized linear 
    #  regression.
    #

    theta = np.array([1, 1])
    J, grad = linearRegCostFunction(np.concatenate([np.ones((m, 1)), X], axis=1), y, theta, 1)

    print('Gradient at theta = [1 ; 1]:  [%f; %f] \n(this value should be about [-15.303016; 598.250744])' % (grad[0], grad[1]))

    print('Program paused. Press enter to continue.')
    #pause;


    ## =========== Part 4: Train Linear Regression =============
    #  Once you have implemented the cost and gradient correctly, the
    #  trainLinearReg function will use your cost function to train 
    #  regularized linear regression.
    # 
    #  Write Up Note: The data is non-linear, so this will not give a great 
    #                 fit.
    #

    fig = plt.figure()

    #  Train linear regression with lambda = 0
    lambda_value = 0
    theta = trainLinearReg(np.concatenate([np.ones((m, 1)), X], axis=1), y, lambda_value)

    #  Plot fit over the data
    plt.plot(X, y, marker='x', linestyle='None', ms=10, lw=1.5)
    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')
    plt.plot(X, np.dot(np.concatenate([np.ones((m, 1)), X], axis=1), theta), '--', lw=2)
    plt.savefig('figure2.png')

    print('Program paused. Press enter to continue.')
    #pause;


    ## =========== Part 5: Learning Curve for Linear Regression =============
    #  Next, you should implement the learningCurve function. 
    #
    #  Write Up Note: Since the model is underfitting the data, we expect to
    #                 see a graph with "high bias" -- slide 8 in ML-advice.pdf 
    #

    fig = plt.figure()

    lambda_value = 0
    error_train, error_val = learningCurve(np.concatenate([np.ones((m, 1)), X], axis=1), y, np.concatenate([np.ones((yval.size, 1)), Xval], axis=1), yval, lambda_value)

    plt.plot(np.arange(1, m + 1), error_train, np.arange(1, m + 1), error_val)
    plt.title('Learning curve for linear regression')
    plt.legend(['Train', 'Cross Validation'])
    plt.xlabel('Number of training examples')
    plt.ylabel('Error')
    plt.axis([0, 13, 0, 150])

    print('# Training Examples\tTrain Error\tCross Validation Error')
    for i in range(m):
        print('  \t%d\t\t%f\t%f' % (i, error_train[i], error_val[i]))
    plt.savefig('figure3.png')

    print('Program paused. Press enter to continue.')
    #pause;

    ## =========== Part 6: Feature Mapping for Polynomial Regression =============
    #  One solution to this is to use polynomial regression. You should now
    #  complete polyFeatures to map each example into its powers
    #

    p = 8

    # Map X onto Polynomial Features and Normalize
    X_poly = polyFeatures(X, p)
    X_poly, mu, sigma = featureNormalize(X_poly)                                 # Normalize
    X_poly = np.concatenate([np.ones((m, 1)), X_poly], axis=1)                   # Add Ones

    # Map X_poly_test and normalize (using mu and sigma)
    X_poly_test = polyFeatures(Xtest, p)
    X_poly_test -= mu
    X_poly_test /= sigma
    X_poly_test = np.concatenate([np.ones((X_poly_test.shape[0], 1)), X_poly_test], axis=1)                   # Add Ones

    # Map X_poly_val and normalize (using mu and sigma)
    X_poly_val = polyFeatures(Xval, p)
    X_poly_val -= mu
    X_poly_val /= sigma
    X_poly_val = np.concatenate([np.ones((X_poly_val.shape[0], 1)), X_poly_val], axis=1)                   # Add Ones

    print('Normalized Training Example 1:')
    print(formatter('  %f  \n', X_poly[0, :]))

    print('\nProgram paused. Press enter to continue.')
    #pause;



    ## =========== Part 7: Learning Curve for Polynomial Regression =============
    #  Now, you will get to experiment with polynomial regression with multiple
    #  values of lambda. The code below runs polynomial regression with 
    #  lambda = 0. You should try running the code with different values of
    #  lambda to see how the fit and learning curve change.
    #

    fig = plt.figure()

    lambda_value = 0
    theta = trainLinearReg(X_poly, y, lambda_value)

    # Plot training data and fit
    plt.plot(X, y, marker='x', ms=10, lw=1.5)
    plotFit(np.min(X), np.max(X), mu, sigma, theta, p)
    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')
    plt.title('Polynomial Regression Fit (lambda = %f)' % lambda_value)

    plt.figure()
    error_train, error_val = learningCurve(X_poly, y, X_poly_val, yval, lambda_value)
    plt.plot(np.arange(1, 1 + m), error_train, np.arange(1, 1 + m), error_val)

    plt.title('Polynomial Regression Learning Curve (lambda = %f)' % lambda_value)
    plt.xlabel('Number of training examples')
    plt.ylabel('Error')
    plt.axis([0, 13, 0, 100])
    plt.legend(['Train', 'Cross Validation'])

    print('Polynomial Regression (lambda = %f)\n' % lambda_value)
    print('# Training Examples\tTrain Error\tCross Validation Error')
    for i in range(m):
        print('  \t%d\t\t%f\t%f' % (i, error_train[i], error_val[i]))
    plt.savefig('figure4.png')

    print('Program paused. Press enter to continue.')
    #pause;

    ## =========== Part 8: Validation for Selecting Lambda =============
    #  You will now implement validationCurve to test various values of 
    #  lambda on a validation set. You will then use this to select the
    #  "best" lambda value.
    #

    fig = plt.figure()

    lambda_vec, error_train, error_val = validationCurve(X_poly, y, X_poly_val, yval)

    plt.plot(lambda_vec, error_train, lambda_vec, error_val)
    plt.legend(['Train', 'Cross Validation'])
    plt.xlabel('lambda');
    plt.ylabel('Error');

    print('lambda\t\tTrain Error\tValidation Error')
    for i in range(lambda_vec.size):
	    print(' %f\t%f\t%f' % (lambda_vec[i], error_train[i], error_val[i]))
    plt.savefig('figure5.png')

    print('Program paused. Press enter to continue.')
    #pause;
