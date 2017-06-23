#!/usr/bin/env python3

import numpy as np
import scipy.io
import matplotlib
# Force matplotlib to not use any X Windows backend (must be called befor importing pyplot)
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from displayData import displayData
from oneVsAll import oneVsAll
from predictOneVsAll import predictOneVsAll


def ex3():
    ## Machine Learning Online Class - Exercise 3 | Part 1: One-vs-all

    #  Instructions
    #  ------------
    # 
    #  This file contains code that helps you get started on the
    #  linear exercise. You will need to complete the following functions 
    #  in this exericse:
    #
    #     lrCostFunction.m (logistic regression cost function)
    #     oneVsAll.m
    #     predictOneVsAll.m
    #     predict.m
    #
    #  For this exercise, you will not need to change any code in this file,
    #  or any other files other than those mentioned above.
    #

    ## Initialization
    #clear ; close all; clc

    ## Setup the parameters you will use for this part of the exercise
    input_layer_size  = 400  # 20x20 Input Images of Digits
    num_labels = 10          # 10 labels, from 1 to 10   
                             # (note that we have mapped "0" to label 10)

    ## =========== Part 1: Loading and Visualizing Data =============
    #  We start the exercise by first loading and visualizing the dataset. 
    #  You will be working with a dataset that contains handwritten digits.
    #

    # Load Training Data
    print('Loading and Visualizing Data ...')

    mat = scipy.io.loadmat('ex3data1.mat') # training data stored in arrays X, y
    X = mat['X']
    y = mat['y'].ravel() % 10
    m = y.size


    # Randomly select 100 data points to display
    rand_indices = np.random.choice(m, 100, replace=False)
    sel = X[rand_indices, :]

    displayData(sel)
    plt.savefig('figure1.png')


    print('Program paused. Press enter to continue.')
    #pause;

    ## ============ Part 2: Vectorize Logistic Regression ============
    #  In this part of the exercise, you will reuse your logistic regression
    #  code from the last exercise. You task here is to make sure that your
    #  regularized logistic regression implementation is vectorized. After
    #  that, you will implement one-vs-all classification for the handwritten
    #  digit dataset.
    #

    print('\nTraining One-vs-All Logistic Regression...')

    lambda_value = 0.1
    all_theta = oneVsAll(X, y, num_labels, lambda_value)

    print('Program paused. Press enter to continue.')
    #pause;


    ## ================ Part 3: Predict for One-Vs-All ================
    #  After ...
    pred = predictOneVsAll(all_theta, X)

    print('\nTraining Set Accuracy: %f' % (np.mean((pred - 1 == y).astype(int)) * 100))
