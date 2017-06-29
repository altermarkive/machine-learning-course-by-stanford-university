#!/usr/bin/env python3

import numpy as np
import scipy.io

from lib.submitWithConfiguration import submitWithConfiguration, formatter
from estimateGaussian import estimateGaussian
from selectThreshold import selectThreshold
from cofiCostFunc import cofiCostFunc


def submit():
    conf = {}
    conf['assignmentSlug'] = 'anomaly-detection-and-recommender-systems'
    conf['itemName'] = 'Anomaly Detection and Recommender Systems'
    conf['partArrays'] = [
        [
            '1',
            ['estimateGaussian.m'],
            'Estimate Gaussian Parameters',
        ],
        [
            '2',
            ['selectThreshold.m'],
            'Select Threshold',
        ],
        [
            '3',
            ['cofiCostFunc.m'],
            'Collaborative Filtering Cost',
        ],
        [
            '4',
            ['cofiCostFunc.m'],
            'Collaborative Filtering Gradient',
        ],
        [
            '5',
            ['cofiCostFunc.m'],
            'Regularized Cost',
        ],
        [
            '6',
            ['cofiCostFunc.m'],
            'Regularized Gradient',
        ],
    ]
    conf['output'] = output

    submitWithConfiguration(conf)


def output(partId):
    # Random Test Cases
    n_u = 3
    n_m = 4
    n = 5
    X = np.sin(np.arange(1, 1 + n_m * n)).reshape(n_m, n, order='F')
    Theta = np.cos(np.arange(1, 1 + n_u * n)).reshape(n_u, n, order='F')
    Y = np.sin(np.arange(1, 1 + 2 * n_m * n_u, 2)).reshape(n_m, n_u, order='F')
    R = Y > 0.5
    pval = np.concatenate([abs(Y.ravel('F')), [0.001], [1]])
    Y = Y * R
    yval = np.concatenate([R.ravel('F'), [1], [0]])
    params = np.concatenate([X.ravel(), Theta.ravel()])
    if partId == '1':
        mu, sigma2 = estimateGaussian(X)
        out = formatter('%0.5f ', mu.ravel())
        out += formatter('%0.5f ', sigma2.ravel())
    elif partId == '2':
        bestEpsilon, bestF1 = selectThreshold(yval, pval)
        out = formatter('%0.5f ', bestEpsilon.ravel())
        out += formatter('%0.5f ', bestF1.ravel())
    elif partId == '3':
        J, _ = cofiCostFunc(params, Y, R, n_u, n_m, n, 0)
        out = formatter('%0.5f ', J.ravel())
    elif partId == '4':
        J, grad = cofiCostFunc(params, Y, R, n_u, n_m, n, 0)
        X_grad = grad[:n_m * n].reshape(n_m, n)
        Theta_grad = grad[n_m * n:].reshape(n_u, n)
        out = formatter('%0.5f ', np.concatenate([X_grad.ravel('F'), Theta_grad.ravel('F')]))
    elif partId == '5':
        J, _ = cofiCostFunc(params, Y, R, n_u, n_m, n, 1.5)
        out = formatter('%0.5f ', J.ravel())
    elif partId == '6':
        J, grad = cofiCostFunc(params, Y, R, n_u, n_m, n, 1.5)
        X_grad = grad[:n_m * n].reshape(n_m, n)
        Theta_grad = grad[n_m * n:].reshape(n_u, n)
        out = formatter('%0.5f ', np.concatenate([X_grad.ravel('F'), Theta_grad.ravel('F')]))
    return out
