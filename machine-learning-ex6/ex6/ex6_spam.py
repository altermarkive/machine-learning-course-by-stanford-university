#!/usr/bin/env python3

import numpy as np
import scipy.io
import matplotlib
# Force matplotlib to not use any X Windows backend (must be called befor importing pyplot)
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from plotData import plotData
from readFile import readFile
from processEmail import processEmail
from emailFeatures import emailFeatures
from linearKernel import linearKernel
from svmTrain import svmTrain
from svmPredict import svmPredict
from getVocabList import getVocabList
from lib.submitWithConfiguration import formatter


def ex6_spam():
    ## Machine Learning Online Class
    #  Exercise 6 | Spam Classification with SVMs
    #
    #  Instructions
    #  ------------
    # 
    #  This file contains code that helps you get started on the
    #  exercise. You will need to complete the following functions:
    #
    #     gaussianKernel.m
    #     dataset3Params.m
    #     processEmail.m
    #     emailFeatures.m
    #
    #  For this exercise, you will not need to change any code in this file,
    #  or any other files other than those mentioned above.
    #

    ## Initialization
    #clear ; close all; clc

    ## ==================== Part 1: Email Preprocessing ====================
    #  To use an SVM to classify emails into Spam v.s. Non-Spam, you first need
    #  to convert each email into a vector of features. In this part, you will
    #  implement the preprocessing steps for each email. You should
    #  complete the code in processEmail.m to produce a word indices vector
    #  for a given email.

    print('\nPreprocessing sample email (emailSample1.txt)')

    # Extract Features
    file_contents = readFile('emailSample1.txt')
    word_indices  = processEmail(file_contents)

    # Print Stats
    print('Word Indices: ')
    print(formatter(' %d', np.array(word_indices) + 1))
    print('\n')

    print('Program paused. Press enter to continue.')
    #pause;

    ## ==================== Part 2: Feature Extraction ====================
    #  Now, you will convert each email into a vector of features in R^n. 
    #  You should complete the code in emailFeatures.m to produce a feature
    #  vector for a given email.

    print('\nExtracting features from sample email (emailSample1.txt)')

    # Extract Features
    file_contents = readFile('emailSample1.txt')
    word_indices  = processEmail(file_contents)
    features      = emailFeatures(word_indices)

    # Print Stats
    print('Length of feature vector: %d' % features.size)
    print('Number of non-zero entries: %d' % np.sum(features > 0))

    print('Program paused. Press enter to continue.')
    #pause;

    ## =========== Part 3: Train Linear SVM for Spam Classification ========
    #  In this section, you will train a linear classifier to determine if an
    #  email is Spam or Not-Spam.

    # Load the Spam Email dataset
    # You will have X, y in your environment
    mat = scipy.io.loadmat('spamTrain.mat')
    X = mat['X'].astype(float)
    y = mat['y'][:, 0]

    print('\nTraining Linear SVM (Spam Classification)\n')
    print('(this may take 1 to 2 minutes) ...\n')

    C = 0.1
    model = svmTrain(X, y, C, linearKernel)

    p = svmPredict(model, X)

    print('Training Accuracy: %f' % (np.mean(p == y) * 100))

    ## =================== Part 4: Test Spam Classification ================
    #  After training the classifier, we can evaluate it on a test set. We have
    #  included a test set in spamTest.mat

    # Load the test dataset
    # You will have Xtest, ytest in your environment
    mat = scipy.io.loadmat('spamTest.mat')
    Xtest = mat['Xtest'].astype(float)
    ytest = mat['ytest'][:, 0]

    print('\nEvaluating the trained Linear SVM on a test set ...\n')

    p = svmPredict(model, Xtest)

    print('Test Accuracy: %f\n' % (np.mean(p == ytest) * 100))
    #pause;


    ## ================= Part 5: Top Predictors of Spam ====================
    #  Since the model we are training is a linear SVM, we can inspect the
    #  weights learned by the model to understand better how it is determining
    #  whether an email is spam or not. The following code finds the words with
    #  the highest weights in the classifier. Informally, the classifier
    #  'thinks' that these words are the most likely indicators of spam.
    #

    # Sort the weights and obtin the vocabulary list
    idx = np.argsort(model['w'])
    top_idx = idx[-15:][::-1]
    vocabList = getVocabList()

    print('\nTop predictors of spam: ')
    for word, w in zip(np.array(vocabList)[top_idx], model['w'][top_idx]):
        print(' %-15s (%f)' % (word, w))
    #end

    print('\n')
    print('\nProgram paused. Press enter to continue.')
    #pause;

    ## =================== Part 6: Try Your Own Emails =====================
    #  Now that you've trained the spam classifier, you can use it on your own
    #  emails! In the starter code, we have included spamSample1.txt,
    #  spamSample2.txt, emailSample1.txt and emailSample2.txt as examples. 
    #  The following code reads in one of these emails and then uses your 
    #  learned SVM classifier to determine whether the email is Spam or 
    #  Not Spam

    # Set the file to be read in (change this to spamSample2.txt,
    # emailSample1.txt or emailSample2.txt to see different predictions on
    # different emails types). Try your own emails as well!
    filename = 'spamSample1.txt'

    # Read and predict
    file_contents = readFile(filename)
    word_indices  = processEmail(file_contents)
    x             = emailFeatures(word_indices)
    p = svmPredict(model, x.ravel())

    print('\nProcessed %s\n\nSpam Classification: %d' % (filename, p))
    print('(1 indicates spam, 0 indicates not spam)\n')

