#!/usr/bin/env python3

import json
import os
import os.path
import urllib.request
import numpy as np
import scipy.io

from .makeValidFieldName import makeValidFieldName


def submitWithConfiguration(conf):
    parts_ = parts(conf)

    print('== Submitting solutions | %s...' % conf['itemName'])

    tokenFile = 'token.mat'
    if os.path.isfile(tokenFile):
        mat = scipy.io.loadmat(tokenFile)
        email = mat['email'][0]
        token = mat['token'][0]
        email, token = promptToken(email, token, tokenFile)
    else:
        email, token = promptToken('', '', tokenFile)

    if isempty(token):
        print('!! Submission Cancelled')
        return

    try:
        response = submitParts(conf, email, token, parts_)
    except Exception as e:
        print('!! Submission failed: unexpected error: %s' % str(e))
        print('!! Please try again later.')
        return

    if 'errorMessage' in response:
        print('!! Submission failed: %s', response['errorMessage'])
    else:
        showFeedback(parts_, response)
        scipy.io.savemat(tokenFile, {'email': np.array([email]), 'token': np.array([token])})


def isempty(text):
    return text is None or text == ''

def promptToken(email, existingToken, tokenFile):
    if not isempty(email) and  not isempty(existingToken):
        prompt = 'Use token from last successful submission (%s)? (Y/n): ' % email
        reenter = input(prompt)

        if isempty(reenter) or reenter[0] == 'Y' or reenter[0] == 'y':
            token = existingToken
            return (email, token)
        else:
            os.remove(tokenFile)
    email = input('Login (email address): ')
    token = input('Token: ')
    return (email, token)


def submitParts(conf, email, token, parts_):
    body = makePostBody(conf, email, token, parts_)
    submissionUrl_ = submissionUrl()
    params = {'jsonBody': body}
    print(json.dumps(params).encode('utf-8'))
    responseBody = urllib.request.urlopen(urllib.request.Request(submissionUrl_, method='POST', data=json.dumps(params).encode('utf-8')))
    responseBody = responseBody.read().decode('utf-8')
    response = json.loads(responseBody)
    return response


def makePostBody(conf, email, token, parts_):
    bodyStruct = {}
    bodyStruct['assignmentSlug'] = conf['assignmentSlug']
    bodyStruct['submitterEmail'] = email
    bodyStruct['secret'] = token
    bodyStruct['parts'] = makePartsStruct(conf, parts_)

    body = json.dumps(bodyStruct)
    return body


def makePartsStruct(conf, parts_):
    partsStruct = {}
    for part in parts_:
        partId = part['id']
        fieldName = makeValidFieldName(partId)
        outputStruct = {}
        outputStruct['output'] = conf['output'](partId)
        partsStruct[fieldName] = outputStruct
    return partsStruct


def parts(conf):
    parts_ = []
    for partArray in conf['partArrays']:
        part = {}
        part['id'] = partArray[0]
        part['sourceFiles'] = partArray[1]
        part['name'] = partArray[2]
        parts_.append(part)
    return parts_


def showFeedback(parts_, response):
    print('== ')
    print('== %43s | %9s | %-s' % ('Part Name', 'Score', 'Feedback'))
    print('== %43s | %9s | %-s' % ('---------', '-----', '--------'))
    for part in parts_:
        score = ''
        partFeedback = ''
        partFeedback = response['partFeedbacks'][makeValidFieldName(part['id'])]
        partEvaluation = response['partEvaluations'][makeValidFieldName(part['id'])]
        score = '%d / %3d' % (partEvaluation['score'], partEvaluation['maxScore'])
        print('== %43s | %9s | %-s' % (part['name'], score, partFeedback))
    evaluation = response['evaluation']
    totalScore = '%d / %d' % (evaluation['score'], evaluation['maxScore'])
    print('==                                   --------------------------------')
    print('== %43s | %9s | %-s' % ('', totalScore, ''))
    print('== ')


###############################################################################
#
# Service configuration
#
###############################################################################
def submissionUrl():
    return 'https://www-origin.coursera.org/api/onDemandProgrammingImmediateFormSubmissions.v1'


def formatter(template, value):
    if isinstance(value, tuple):
        value = value[0]
    if isinstance(value, (np.ndarray, list)):
        return ''.join(template % item for item in np.asarray(value).ravel('F'))
    return template % value