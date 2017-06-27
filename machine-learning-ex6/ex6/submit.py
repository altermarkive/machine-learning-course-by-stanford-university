#!/usr/bin/env python3

import numpy as np
import scipy.io

from lib.submitWithConfiguration import submitWithConfiguration, formatter
from gaussianKernel import gaussianKernel
from dataset3Params import dataset3Params
from processEmail import processEmail
from emailFeatures import emailFeatures


def submit():
    conf = {}
    conf['assignmentSlug'] = 'support-vector-machines'
    conf['itemName'] = 'Support Vector Machines'
    conf['partArrays'] = [
        [
            '1',
            ['gaussianKernel.m'],
            'Gaussian Kernel',
        ],
        [
            '2',
            ['dataset3Params.m'],
            'Parameters (C, sigma) for Dataset 3',
        ],
        [
            '3',
            ['processEmail.m'],
            'Email Preprocessing',
        ],
        [
            '4',
            ['emailFeatures.m'],
            'Email Feature Extraction',
        ],
    ]
    conf['output'] = output

    submitWithConfiguration(conf)


def output(partId):
    # Random Test Cases
    x1 = np.sin(np.arange(1, 11))
    x2 = np.cos(np.arange(1, 11))
    ec = 'the quick brown fox jumped over the lazy dog'
    wi = np.abs(np.round(x1 * 1863)).astype(int)
    wi = np.concatenate([wi, wi])
    if partId == '1':
        sim = gaussianKernel(x1, x2, 2)
        out = formatter('%0.5f ', sim)
    elif partId == '2':
        mat = scipy.io.loadmat('ex6data3.mat')
        X = mat['X']
        y = mat['y'].ravel()
        Xval = mat['Xval']
        yval = mat['yval'].ravel()
        C, sigma = dataset3Params(X, y, Xval, yval)
        out = formatter('%0.5f ', C)
        out += formatter('%0.5f ', sigma)
    elif partId == '3':
        word_indices = processEmail(ec) + 1
        out = formatter('%d ', word_indices)
    elif partId == '4':
        x = emailFeatures(wi)
        out = formatter('%d ', x)
    return out
