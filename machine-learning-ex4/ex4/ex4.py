#!/usr/bin/env python3

import numpy as np
import scipy.io
import matplotlib
# Force matplotlib to not use any X Windows backend (must be called befor importing pyplot)
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy import optimize

from displayData import displayData
from randInitializeWeights import randInitializeWeights
from checkNNGradients import checkNNGradients
from nnCostFunction import nnCostFunction
from sigmoidGradient import sigmoidGradient
from predict import predict
from lib.submitWithConfiguration import formatter


def ex4():
    ## Machine Learning Online Class - Exercise 4 Neural Network Learning

    #  Instructions
    #  ------------
    # 
    #  This file contains code that helps you get started on the
    #  linear exercise. You will need to complete the following functions 
    #  in this exericse:
    #
    #     sigmoidGradient.m
    #     randInitializeWeights.m
    #     nnCostFunction.m
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

    mat = scipy.io.loadmat('ex4data1.mat')
    X = mat['X']
    y = mat['y'].ravel()
    m = X.shape[0]

    # Randomly select 100 data points to display
    sel = np.random.choice(m, 100, replace=False)

    displayData(X[sel, :])
    plt.savefig('figure1.png')

    print('Program paused. Press enter to continue.')
    #pause;


    ## ================ Part 2: Loading Parameters ================
    # In this part of the exercise, we load some pre-initialized 
    # neural network parameters.

    print('\nLoading Saved Neural Network Parameters ...')

    # Load the weights into variables Theta1 and Theta2
    mat = scipy.io.loadmat('ex4weights.mat')
    Theta1 = mat['Theta1']
    Theta2 = mat['Theta2']

    # Unroll parameters 
    nn_params = np.concatenate([Theta1.ravel(), Theta2.ravel()])

    ## ================ Part 3: Compute Cost (Feedforward) ================
    #  To the neural network, you should first start by implementing the
    #  feedforward part of the neural network that returns the cost only. You
    #  should complete the code in nnCostFunction.m to return cost. After
    #  implementing the feedforward to compute the cost, you can verify that
    #  your implementation is correct by verifying that you get the same cost
    #  as us for the fixed debugging parameters.
    #
    #  We suggest implementing the feedforward cost *without* regularization
    #  first so that it will be easier for you to debug. Later, in part 4, you
    #  will get to implement the regularized cost.
    #
    print('\nFeedforward Using Neural Network ...')

    # Weight regularization parameter (we set this to 0 here).
    lambda_value = 0

    J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_value)[0]

    print('Cost at parameters (loaded from ex4weights): %f \n(this value should be about 0.287629)' % J)

    print('\nProgram paused. Press enter to continue.')
    #pause;

    ## =============== Part 4: Implement Regularization ===============
    #  Once your cost function implementation is correct, you should now
    #  continue to implement the regularization with the cost.
    #

    print('\nChecking Cost Function (w/ Regularization) ... ')

    # Weight regularization parameter (we set this to 1 here).
    lambda_value = 1

    J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_value)[0]

    print('Cost at parameters (loaded from ex4weights): %f \n(this value should be about 0.383770)' % J)

    print('Program paused. Press enter to continue.')
    #pause;


    ## ================ Part 5: Sigmoid Gradient  ================
    #  Before you start implementing the neural network, you will first
    #  implement the gradient for the sigmoid function. You should complete the
    #  code in the sigmoidGradient.m file.
    #

    print('\nEvaluating sigmoid gradient...')

    g = sigmoidGradient(np.array([1, -0.5, 0, 0.5, 1]))
    print('Sigmoid gradient evaluated at [1 -0.5 0 0.5 1]:')
    print(formatter('%f ', g))
    print('\n')

    print('Program paused. Press enter to continue.')
    #pause;


    ## ================ Part 6: Initializing Pameters ================
    #  In this part of the exercise, you will be starting to implment a two
    #  layer neural network that classifies digits. You will start by
    #  implementing a function to initialize the weights of the neural network
    #  (randInitializeWeights.m)

    print('\nInitializing Neural Network Parameters ...')

    initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
    initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)

    # Unroll parameters
    initial_nn_params = np.concatenate([initial_Theta1.ravel(), initial_Theta2.ravel()])


    ## =============== Part 7: Implement Backpropagation ===============
    #  Once your cost matches up with ours, you should proceed to implement the
    #  backpropagation algorithm for the neural network. You should add to the
    #  code you've written in nnCostFunction.m to return the partial
    #  derivatives of the parameters.
    #
    print('\nChecking Backpropagation... ')

    #  Check gradients by running checkNNGradients
    checkNNGradients()

    print('\nProgram paused. Press enter to continue.')
    #pause;


    ## =============== Part 8: Implement Regularization ===============
    #  Once your backpropagation implementation is correct, you should now
    #  continue to implement the regularization with the cost and gradient.
    #

    print('\nChecking Backpropagation (w/ Regularization) ... ')

    #  Check gradients by running checkNNGradients
    lambda_value = 3
    checkNNGradients(lambda_value)

    # Also output the costFunction debugging values
    debug_J  = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_value)[0]

    print('\n\nCost at (fixed) debugging parameters (w/ lambda = 10): %f \n(this value should be about 0.576051)\n\n' % debug_J)

    print('Program paused. Press enter to continue.')
    #pause;


    ## =================== Part 8: Training NN ===================
    #  You have now implemented all the code necessary to train a neural 
    #  network. To train your neural network, we will now use "fmincg", which
    #  is a function which works similarly to "fminunc". Recall that these
    #  advanced optimizers are able to train our cost functions efficiently as
    #  long as we provide them with the gradient computations.
    #
    print('\nTraining Neural Network... ')

    #  After you have completed the assignment, change the MaxIter to a larger
    #  value to see how more training helps.
    options = {'maxiter': 50}

    #  You should also try different values of lambda
    lambda_value = 1

    # Create "short hand" for the cost function to be minimized
    costFunction = lambda p: nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_value)

    # Now, costFunction is a function that takes in only one argument (the
    # neural network parameters)
    res = optimize.minimize(costFunction, initial_nn_params, jac=True, method='TNC', options=options)
    nn_params = res.x

    # Obtain Theta1 and Theta2 back from nn_params
    Theta1 = nn_params[0:hidden_layer_size*(input_layer_size+1)].reshape(
        hidden_layer_size, input_layer_size+1)

    Theta2 = nn_params[hidden_layer_size*(input_layer_size+1):].reshape(
        num_labels, hidden_layer_size+1)


    print('Program paused. Press enter to continue.')
    #pause;


    ## ================= Part 9: Visualize Weights =================
    #  You can now "visualize" what the neural network is learning by 
    #  displaying the hidden units to see what features they are capturing in 
    #  the data.

    print('\nVisualizing Neural Network... ')

    displayData(Theta1[:, 1:])
    plt.savefig('figure2.png')

    print('\nProgram paused. Press enter to continue.')
    #pause;

    ## ================= Part 10: Implement Predict =================
    #  After training the neural network, we would like to use it to predict
    #  the labels. You will now implement the "predict" function to use the
    #  neural network to predict the labels of the training set. This lets
    #  you compute the training set accuracy.

    pred = predict(Theta1, Theta2, X)

    print('\nTraining Set Accuracy: %f' % (np.mean((pred == y).astype(int)) * 100))
