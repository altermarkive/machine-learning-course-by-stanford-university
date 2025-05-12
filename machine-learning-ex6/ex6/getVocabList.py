#!/usr/bin/env python3

import numpy as np


def getVocabList():
    #GETVOCABLIST reads the fixed vocabulary list in vocab.txt and returns a
    #cell array of the words
    #   vocabList = GETVOCABLIST() reads the fixed vocabulary list in vocab.txt 
    #   and returns a cell array of the words in vocabList.
    vocabList = np.genfromtxt('vocab.txt', dtype=object)
    return list(vocabList[:, 1].astype(str))
