#!/usr/bin/env python3

import numpy as np

from lib.submitWithConfiguration import submitWithConfiguration, formatter
from lrCostFunction import lrCostFunction
from oneVsAll import oneVsAll
from predictOneVsAll import predictOneVsAll
from predict import predict


def submit():
    conf = {}
    conf['assignmentSlug'] = 'multi-class-classification-and-neural-networks'
    conf['itemName'] = 'Multi-class Classification and Neural Networks'
    conf['partArrays'] = [
        [
            '1',
            ['lrCostFunction.m'],
            'Regularized Logistic Regression',
        ],
        [
            '2',
            ['oneVsAll.m'],
            'One-vs-All Classifier Training',
        ],
        [
            '3',
            ['predictOneVsAll.m'],
            'One-vs-All Classifier Prediction',
        ],
        [
            '4',
            ['predict.m'],
            'Neural Network Prediction Function'
        ],
    ]
    conf['output'] = output

    submitWithConfiguration(conf)


def output(partId):
    # Random Test Cases
    X = np.stack([np.ones(20), np.exp(1) * np.sin(np.arange(1, 21)), np.exp(0.5) * np.cos(np.arange(1, 21))], axis=1)
    y = (np.sin(X[:, 0] + X[:, 1]) > 0).astype(float)
    Xm = np.array([[-1, -1], [-1, -2], [-2, -1], [-2, -2],
                   [1, 1], [1, 2], [2, 1], [2, 2],
                   [-1, 1], [-1, 2], [-2, 1], [-2, 2],
                   [1, -1], [1, -2], [-2, -1], [-2, -2]])
    ym = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
    t1 = np.sin(np.reshape(np.arange(1, 25, 2), (4, 3), order='F'))
    t2 = np.cos(np.reshape(np.arange(1, 41, 2), (4, 5), order='F'))
    if partId == '1':
        J, grad = lrCostFunction(np.array([0.25, 0.5, -0.5]), X, y, 0.1)
        out = formatter('%0.5f ', J)
        out += formatter('%0.5f ', grad)
    elif partId == '2':
        out = formatter('%0.5f ', oneVsAll(Xm, ym, 4, 0.1))
    elif partId == '3':
        out = formatter('%0.5f ', predictOneVsAll(t1, Xm))
    elif partId == '4':
        out = formatter('%0.5f ', predict(t1, t2, Xm))
    return out
