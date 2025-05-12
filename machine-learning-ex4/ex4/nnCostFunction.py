#!/usr/bin/env python3

import numpy as np

from sigmoid import sigmoid
from sigmoidGradient import sigmoidGradient


def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_value):
    #NNCOSTFUNCTION Implements the neural network cost function for a two layer
    #neural network which performs classification
    #   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
    #   X, y, lambda) computes the cost and gradient of the neural network. The
    #   parameters for the neural network are "unrolled" into the vector
    #   nn_params and need to be converted back into the weight matrices. 
    # 
    #   The returned parameter grad should be a "unrolled" vector of the
    #   partial derivatives of the neural network.
    #

    # Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
    # for our 2 layer neural network
    Theta1 = nn_params[0:hidden_layer_size*(input_layer_size+1)].reshape(
        hidden_layer_size, input_layer_size+1)
    Theta2 = nn_params[hidden_layer_size*(input_layer_size+1):].reshape(
        num_labels, hidden_layer_size+1)

    # Setup some useful variables
    m, n = X.shape
         
    # You need to return the following variables correctly 
    J = 0
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)

    # ====================== YOUR CODE HERE ======================
    # Instructions: You should complete the code by working through the
    #               following parts.
    #
    # Part 1: Feedforward the neural network and return the cost in the
    #         variable J. After implementing Part 1, you can verify that your
    #         cost function computation is correct by verifying the cost
    #         computed in ex4.m
    #
    # Part 2: Implement the backpropagation algorithm to compute the gradients
    #         Theta1_grad and Theta2_grad. You should return the partial derivatives of
    #         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
    #         Theta2_grad, respectively. After implementing Part 2, you can check
    #         that your implementation is correct by running checkNNGradients
    #
    #         Note: The vector y passed into the function is a vector of labels
    #               containing values from 1..K. You need to map this vector into a 
    #               binary vector of 1's and 0's to be used with the neural network
    #               cost function.
    #
    #         Hint: We recommend implementing backpropagation using a for-loop
    #               over the training examples if you are implementing it for the 
    #               first time.
    #
    # Part 3: Implement regularization with the cost function and gradients.
    #
    #         Hint: You can implement this around the code for
    #               backpropagation. That is, you can compute the gradients for
    #               the regularization separately and then add them to Theta1_grad
    #               and Theta2_grad from Part 2.
    #

    # Feed forward
    a1 = np.column_stack([np.ones(m), X])
    z2 = np.matmul(a1, Theta1.T)
    a2 = sigmoid(z2)
    a2 = np.column_stack([np.ones(m), a2])
    z3 = np.matmul(a2, Theta2.T)
    h = sigmoid(z3)

    # Main term of the cost function
    for k in range(1, num_labels + 1):
        yk = (y == k).astype(int)
        hk = h[:, k - 1]
        Jk = np.sum(-yk * np.log(hk) - (1 - yk) * np.log(1 - hk)) / m
        J = J + Jk

    # Regularization term of the cost function
    J = J + lambda_value * (np.sum(np.sum(Theta1[:, 1:] ** 2)) + np.sum(np.sum(Theta2[:, 1:] ** 2))) / (2 * m)

    # Backpropagation
    for t in range(1, m + 1):
        # For each training sample
        d3 = np.zeros((1, num_labels))
        for k in range(1, num_labels + 1):
            yk = (y[t - 1] == k).astype(int)
            d3[0, k - 1] = h[t - 1, k - 1] - yk
        d2 = np.multiply(np.dot(Theta2.T, d3.T), sigmoidGradient(np.r_[1, z2[t - 1, :]])[None].T)
        d2 = d2[1:]
        Theta1_grad = Theta1_grad + np.dot(d2, a1[t - 1, :][None])
        Theta2_grad = Theta2_grad + np.dot(d3.T, a2[t - 1, :][None])
    # Main term of the gradient
    Theta1_grad = Theta1_grad / m
    Theta2_grad = Theta2_grad / m
    # Regularization term of the gradient
    Theta1_grad[:, 1:] = Theta1_grad[:, 1:] + lambda_value * Theta1[:, 1:] / m
    Theta2_grad[:, 1:] = Theta2_grad[:, 1:] + lambda_value * Theta2[:, 1:] / m















    # -------------------------------------------------------------

    # =========================================================================

    # Unroll gradients
    grad = np.concatenate([Theta1_grad.ravel(), Theta2_grad.ravel()])


    return (J, grad)
    #end
