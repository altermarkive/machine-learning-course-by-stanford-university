#!/usr/bin/env python3

import re
import numpy as np

from getVocabList import getVocabList
from porterStemmer import PorterStemmer


def processEmail(email_contents):
    #PROCESSEMAIL preprocesses a the body of an email and
    #returns a list of word_indices 
    #   word_indices = PROCESSEMAIL(email_contents) preprocesses 
    #   the body of an email and returns a list of indices of the 
    #   words contained in the email. 
    #

    # Load Vocabulary
    vocabList = getVocabList()

    # Init return value
    word_indices = []

    # ========================== Preprocess Email ===========================

    # Find the Headers ( \n\n and remove )
    # Uncomment the following lines if you are working with raw emails with the
    # full headers

    # hdrstart = strfind(email_contents, ([char(10) char(10)]));
    # email_contents = email_contents(hdrstart(1):end);

    # Lower case
    email_contents = email_contents.lower()

    # Strip all HTML
    # Looks for any expression that starts with < and ends with > and replace
    # and does not have any < or > in the tag it with a space
    email_contents = re.compile('<[^<>]+>').sub(' ', email_contents)

    # Handle Numbers
    # Look for one or more characters between 0-9
    email_contents = re.compile('[0-9]+').sub(' number ', email_contents)

    # Handle URLS
    # Look for strings starting with http:// or https://
    email_contents = re.compile('(http|https)://[^\\s]*').sub(' httpaddr ', email_contents)

    # Handle Email Addresses
    # Look for strings with @ in the middle
    email_contents = re.compile('[^\\s]+@[^\\s]+').sub(' emailaddr ', email_contents)

    # Handle $ sign
    email_contents = re.compile('[$]+').sub(' dollar ', email_contents)

    # Other
    email_contents = re.split('[ @$/#.-:&*+=\\[\\]?!(){},''\">_<;%\\n\\r]', email_contents)
    email_contents = [word for word in email_contents if len(word) > 0]


    # ========================== Tokenize Email ===========================

    # Output the email to screen as well
    print('\n==== Processed Email ====\n')

    # Process file
    stemmer = PorterStemmer()
    processed_email = []
    for word in email_contents:
        word = re.compile('[^a-zA-Z0-9]').sub('', word).strip()
        word = stemmer.stem(word)
        processed_email.append(word)
        # Skip the word if it is too short
        if len(word) < 1:
            continue
        # Look up the word in the dictionary and add to word_indices if
        # found
        # ====================== YOUR CODE HERE ======================
        # Instructions: Fill in this function to add the index of str to
        #               word_indices if it is in the vocabulary. At this point
        #               of the code, you have a stemmed word from the email in
        #               the variable str. You should look up str in the
        #               vocabulary list (vocabList). If a match exists, you
        #               should add the index of the word to the word_indices
        #               vector. Concretely, if str = 'action', then you should
        #               look up the vocabulary list to find where in vocabList
        #               'action' appears. For example, if vocabList{18} =
        #               'action', then, you should add 18 to the word_indices 
        #               vector (e.g., word_indices = [word_indices ; 18]; ).
        # 
        # Note: vocabList{idx} returns a the word with index idx in the
        #       vocabulary list.
        # 
        # Note: You can use strcmp(str1, str2) to compare two strings (str1 and
        #       str2). It will return 1 only if the two strings are equivalent.
        #
        try:
            index = vocabList.index(word)
        except ValueError:
            pass
        else:
            word_indices.append(index)
        # ============================================================"
    print(' '.join(processed_email))
    # Print footer
    print('\n\n=========================')
    return word_indices