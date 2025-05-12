#!/usr/bin/env python3

import numpy as np

from lib.submitWithConfiguration import submitWithConfiguration, formatter
from sigmoid import sigmoid
from costFunction import costFunction
from costFunctionReg import costFunctionReg
from predict import predict


def submit():
    conf = {}
    conf['assignmentSlug'] = 'logistic-regression'
    conf['itemName'] = 'Logistic Regression'
    conf['partArrays'] = [
        [
            '1',
            ['sigmoid.m'],
            'Sigmoid Function',
        ],
        [
            '2',
            ['costFunction.m'],
            'Logistic Regression Cost',
        ],
        [
            '3',
            ['costFunction.m'],
            'Logistic Regression Gradient',
        ],
        [
            '4',
            ['predict.m'],
            'Predict',
        ],
        [
            '5',
            ['costFunctionReg.m'],
            'Regularized Logistic Regression Cost',
        ],
        [
            '6',
            ['costFunctionReg.m'],
            'Regularized Logistic Regression Gradient',
        ],
    ]
    conf['output'] = output

    submitWithConfiguration(conf)


def output(partId):
    # Random Test Cases
    X = np.stack([np.ones(20), np.exp(1) * np.sin(np.arange(1, 21)), np.exp(0.5) * np.cos(np.arange(1, 21))], axis=1)
    y = np.reshape((np.sin(X[:, 0] + X[:, 1]) > 0).astype(float), (20, 1))
    if partId == '1':
        out = formatter('%0.5f ', sigmoid(X))
    elif partId == '2':
        out = formatter('%0.5f ', costFunction(np.reshape(np.array([0.25, 0.5, -0.5]), (3, 1)), X, y))
    elif partId == '3':
        cost, grad = costFunction(np.reshape(np.array([0.25, 0.5, -0.5]), (3, 1)), X, y)
        out = formatter('%0.5f ', grad)
    elif partId == '4':
        out = formatter('%0.5f ', predict(np.reshape(np.array([0.25, 0.5, -0.5]), (3, 1)), X))
    elif partId == '5':
        out = formatter('%0.5f ', costFunctionReg(np.reshape(np.array([0.25, 0.5, -0.5]), (3, 1)), X, y, 0.1))
    elif partId == '6':
        cost, grad = costFunctionReg(np.reshape(np.array([0.25, 0.5, -0.5]), (3, 1)), X, y, 0.1)
        out = formatter('%0.5f ', grad)
    return out