#!/usr/bin/env python3

import numpy as np

from lib.submitWithConfiguration import submitWithConfiguration, formatter
from findClosestCentroids import findClosestCentroids
from computeCentroids import computeCentroids
from pca import pca
from projectData import projectData
from recoverData import recoverData


def submit():
    conf = {}
    conf['assignmentSlug'] = 'k-means-clustering-and-pca'
    conf['itemName'] = 'K-Means Clustering and PCA'
    conf['partArrays'] = [
        [
            '1',
            ['findClosestCentroids.m'],
            'Find Closest Centroids (k-Means)',
        ],
        [
            '2',
            ['computeCentroids.m'],
            'Compute Centroid Means (k-Means)',
        ],
        [
            '3',
            ['pca.m'],
            'PCA',
        ],
        [
            '4',
            ['projectData.m'],
            'Project Data (PCA)',
        ],
        [
            '5',
            ['recoverData.m'],
            'Recover Data (PCA)',
        ],
    ]
    conf['output'] = output

    submitWithConfiguration(conf)


def output(partId):
    # Random Test Cases
    X = np.sin(np.arange(1, 166)).reshape(15, 11, order='F')
    Z = np.cos(np.arange(1, 122)).reshape(11, 11, order='F')
    C = Z[:5, :]
    idx = np.arange(1, 16) % 3
    if partId == '1':
        idx = findClosestCentroids(X, C) + 1
        out = formatter('%0.5f ', idx.ravel('F'))
    elif partId == '2':
        centroids = computeCentroids(X, idx, 3)
        out = formatter('%0.5f ', centroids.ravel('F'))
    elif partId == '3':
        U, S = pca(X)
        out = formatter('%0.5f ', np.abs(np.hstack([U.ravel('F'), np.diag(S).ravel('F')]))) 
    elif partId == '4':
        X_proj = projectData(X, Z, 5)
        out = formatter('%0.5f ', X_proj.ravel('F'))
    elif partId == '5':
        X_rec = recoverData(X[:,:5], Z, 5)
        out = formatter('%0.5f ', X_rec.ravel('F'))
    return out
