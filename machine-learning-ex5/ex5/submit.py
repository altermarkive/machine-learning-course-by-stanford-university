#!/usr/bin/env python3

import numpy as np

from lib.submitWithConfiguration import submitWithConfiguration, formatter
from linearRegCostFunction import linearRegCostFunction
from learningCurve import learningCurve
from polyFeatures import polyFeatures
from validationCurve import validationCurve


def submit():
    conf = {}
    conf['assignmentSlug'] = 'regularized-linear-regression-and-bias-variance'
    conf['itemName'] = 'Regularized Linear Regression and Bias/Variance'
    conf['partArrays'] = [
        [
            '1',
            ['linearRegCostFunction.m'],
            'Regularized Linear Regression Cost Function',
        ],
        [
            '2',
            ['linearRegCostFunction.m'],
            'Regularized Linear Regression Gradient',
        ],
        [
            '3',
            ['learningCurve.m'],
            'Learning Curve',
        ],
        [
            '4',
            ['polyFeatures.m'],
            'Polynomial Feature Mapping',
        ],
        [
            '5',
            ['validationCurve.m'],
            'Validation Curve',
        ],
    ]
    conf['output'] = output

    submitWithConfiguration(conf)


def output(partId):
    # Random Test Cases
    X = np.vstack([np.ones(10), np.sin(np.arange(1, 15, 1.5)), np.cos(np.arange(1, 15, 1.5))]).T
    y = np.sin(np.arange(1, 31, 3))
    Xval = np.vstack([np.ones(10), np.sin(np.arange(0, 14, 1.5)), np.cos(np.arange(0, 14, 1.5))]).T
    yval = np.sin(np.arange(1, 11))
    if partId == '1':
        J, _ = linearRegCostFunction(X, y, np.array([0.1, 0.2, 0.3]), 0.5)
        out = formatter('%0.5f ', J)
    elif partId == '2':
        J, grad = linearRegCostFunction(X, y, np.array([0.1, 0.2, 0.3]), 0.5)
        out = formatter('%0.5f ', grad)
    elif partId == '3':
        error_train, error_val = learningCurve(X, y, Xval, yval, 1)
        out = formatter('%0.5f ', np.concatenate([error_train.ravel(), error_val.ravel()]))
    elif partId == '4':
        X_poly = polyFeatures(X[1,:].T, 8)
        out = formatter('%0.5f ', X_poly)
    elif partId == '5':
        lambda_vec, error_train, error_val = validationCurve(X, y, Xval, yval)
        out = formatter('%0.5f ', np.concatenate([lambda_vec.ravel(), error_train.ravel(), error_val.ravel()]))
    return out
