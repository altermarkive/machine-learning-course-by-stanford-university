#!/usr/bin/env python3

import numpy as np
import scipy.io
import matplotlib
# Force matplotlib to not use any X Windows backend (must be called befor importing pyplot)
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from displayData import displayData
from predict import predict


def ex3_nn():
    ## Machine Learning Online Class - Exercise 3 | Part 2: Neural Networks

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

    ## Setup the parameters you will use for this exercise
    input_layer_size  = 400  # 20x20 Input Images of Digits
    hidden_layer_size = 25   # 25 hidden units
    num_labels = 10          # 10 labels, from 1 to 10   
                             # (note that we have mapped "0" to label 10)

    ## =========== Part 1: Loading and Visualizing Data =============
    #  We start the exercise by first loading and visualizing the dataset. 
    #  You will be working with a dataset that contains handwritten digits.
    #

    # Load Training Data
    print('Loading and Visualizing Data ...')

    mat = scipy.io.loadmat('ex3data1.mat')
    X = mat['X']
    y = mat['y'].ravel()
    m = X.shape[0]

    # Randomly select 100 data points to display
    sel = np.random.choice(m, 100, replace=False)

    displayData(X[sel, :])
    plt.savefig('figure2.png')

    print('Program paused. Press enter to continue.')
    #pause;

    ## ================ Part 2: Loading Pameters ================
    # In this part of the exercise, we load some pre-initialized 
    # neural network parameters.

    print('\nLoading Saved Neural Network Parameters ...')

    # Load the weights into variables Theta1 and Theta2
    mat = scipy.io.loadmat('ex3weights.mat')
    Theta1 = mat['Theta1']
    Theta2 = mat['Theta2']

    ## ================= Part 3: Implement Predict =================
    #  After training the neural network, we would like to use it to predict
    #  the labels. You will now implement the "predict" function to use the
    #  neural network to predict the labels of the training set. This lets
    #  you compute the training set accuracy.

    pred = predict(Theta1, Theta2, X)

    print('\nTraining Set Accuracy: %f' % (np.mean((pred == y).astype(int)) * 100))

    print('Program paused. Press enter to continue.\n')
    #pause;

    #  To give you an idea of the network's output, you can also run
    #  through the examples one at the a time to see what it is predicting.

    #  Randomly permute examples
    rp = np.random.choice(m, 100, replace=False)

    for i in range(m):
        # Display 
        print('\nDisplaying Example Image')
        displayData(X[rp[i], :][None])
        plt.savefig('figure3_%d.png' % i)

        pred = predict(Theta1, Theta2, X[rp[i],:])
        print('\nNeural Network Prediction: %d (digit %d)' % (pred, pred % 10))
    
        # Pause
        print('Program paused. Press enter to continue.')
        input()
    #end
