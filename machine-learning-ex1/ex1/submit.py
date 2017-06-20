#!/usr/bin/env python3

import numpy as np

from lib.submitWithConfiguration import submitWithConfiguration, formatter
from warmUpExercise import warmUpExercise
from computeCost import computeCost
from gradientDescent import gradientDescent
from featureNormalize import featureNormalize
from computeCostMulti import computeCostMulti
from gradientDescentMulti import gradientDescentMulti
from normalEqn import normalEqn


def submit():
    conf = {}
    conf['assignmentSlug'] = 'linear-regression'
    conf['itemName'] = 'Linear Regression with Multiple Variables'
    conf['partArrays'] = [
        [
            '1',
            ['warmUpExercise.m'],
            'Warm-up Exercise',
        ],
        [
            '2',
            ['computeCost.m'],
            'Computing Cost (for One Variable)',
        ],
        [
            '3',
            ['gradientDescent.m'],
            'Gradient Descent (for One Variable)',
        ],
        [
            '4',
            ['featureNormalize.m'],
            'Feature Normalization',
        ],
        [
            '5',
            ['computeCostMulti.m'],
            'Computing Cost (for Multiple Variables)',
        ],
        [
            '6',
            ['gradientDescentMulti.m'],
            'Gradient Descent (for Multiple Variables)',
        ],
        [
            '7',
            ['normalEqn.m'],
            'Normal Equations',
        ],
    ]
    conf['output'] = output

    submitWithConfiguration(conf)


def output(partId):
    # Random Test Cases
    X1 = np.column_stack((np.ones(20), np.exp(1) + np.exp(2) * np.linspace(0.1, 2, 20)))
    Y1 = X1[:, 1] + np.sin(X1[:, 0]) + np.cos(X1[:, 1])
    X2 = np.column_stack((X1, X1[:, 1]**0.5, X1[:, 1]**0.25))
    Y2 = np.power(Y1, 0.5) + Y1
    if partId == '1':
        out = formatter('%0.5f ', warmUpExercise())
    elif partId == '2':
        out = formatter('%0.5f ', computeCost(X1, Y1, np.array([0.5, -0.5])))
    elif partId == '3':
        out = formatter('%0.5f ', gradientDescent(X1, Y1, np.array([0.5, -0.5]), 0.01, 10))
    elif partId == '4':
        out = formatter('%0.5f ', featureNormalize(X2[:, 1:4]))
    elif partId == '5':
        out = formatter('%0.5f ', computeCostMulti(X2, Y2, np.array([0.1, 0.2, 0.3, 0.4])))
    elif partId == '6':
        out = formatter('%0.5f ', gradientDescentMulti(X2, Y2, np.array([-0.1, -0.2, -0.3, -0.4]), 0.01, 10))
    elif partId == '7':
        out = formatter('%0.5f ', normalEqn(X2, Y2))
    return out