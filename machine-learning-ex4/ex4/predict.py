#!/usr/bin/env python3

import numpy as np

from sigmoid import sigmoid


def predict(Theta1, Theta2, X):
    #PREDICT Predict the label of an input given a trained neural network
    #   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
    #   trained weights of a neural network (Theta1, Theta2)

    # Useful values
    m = X.shape[0]
    num_labels = Theta2.shape[0]

    # You need to return the following variables correctly 
    p = np.zeros((X.shape[0], 1))

    h1 = sigmoid(np.dot(np.column_stack([np.ones((m, 1)), X]), Theta1.T))
    h2 = sigmoid(np.dot(np.column_stack([np.ones((m, 1)), h1]), Theta2.T))
    p = np.argmax(h2, axis = 1) + 1

    # =========================================================================


    return p
    #end
