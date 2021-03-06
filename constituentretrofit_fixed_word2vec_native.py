#!/usr/bin/env python
# encoding: utf-8
"""
Copyright (C) 2015 Sujay Kumar Jauhar <sjauhar@cs.cmu.edu>
Modified to include constituent by Han-Chin Shing
Licenced under the Apache Licence, v2.0 - http://www.apache.org/licenses/LICENSE-2.0
"""

import sys
import getopt
import numpy
# import gzip
import json

from scipy.sparse import lil_matrix
# from copy import deepcopy
from itertools import izip
from gensim.models import word2vec
import re

help_message = '''
$ python constituentretrofit_fixed_word2vec_native.py -v <vectorsFile> -t <testPhrases> [-o outputFile] [-n numIters] [-e epsilon] [-h]
-v or --vectors to specify path to the word vectors input file (gzip or txt files are acceptable)
-o or --output to optionally set path to output word sense vectors file (<vectorsFile>.sense is used by default)
-n or --numiters to optionally set the number of retrofitting iterations (10 is the default)
-e or --epsilon to optionally set the convergence threshold (0.001 is the default)
-h or --help (this message is displayed)
'''

phraseSeparator = '|'


class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg


# Read command line arguments
def readCommandLineInput(argv):
    try:
        try:
            # specify the possible option switches
            opts, _ = getopt.getopt(argv[1:], "hv:t:o:n:e:", ["help", "vectors=", "testPhrases="
                                                              "output=", "numiters=", "epsilon="])
        except getopt.error, msg:
            raise Usage(msg)

        # default values
        vectorsFile = None
        testPhraseFile = None
        outputFile = None
        numIters = 6
        epsilon = 0.001

        setOutput = False
        # option processing
        for option, value in opts:
            if option in ("-h", "--help"):
                raise Usage(help_message)
            elif option in ("-v", "--vectors"):
                vectorsFile = value
            elif option in ("-o", "--output"):
                outputFile = value
                setOutput = True
            elif option in ("-n", "--numiters"):
                try:
                    numIters = int(value)
                except:
                    raise Usage(help_message)
            elif option in ("-e", "--epsilon"):
                try:
                    epsilon = float(value)
                except:
                    raise Usage(help_message)
            elif option in ("-t", "--testPhrases"):
                testPhraseFile = value
            else:
                raise Usage(help_message)

        if (vectorsFile is None) or (testPhraseFile is None):
            raise Usage(help_message)
        else:
            if not setOutput:
                outputFile = vectorsFile + '.cons'
            return (vectorsFile, testPhraseFile, outputFile, numIters, epsilon)

    except Usage, err:
        print str(err.msg)
        return 2


# Read all the word vectors from file.
def readWordVectors(filename):
    sys.stderr.write('Reading vectors from file...\n')

    model = word2vec.Word2Vec.load(filename)

    vectorDim = len(model[model.vocab.iterkeys().next()])
    wordVectors = model
    sys.stderr.write('Loaded vectors from file...\n')

    vocab = {word: model.vocab[word].index for word in model.vocab}

    sys.stderr.write('Finished reading vectors.\n')

    return vocab, wordVectors, vectorDim


# Write word vectors to file
def writeWordVectors(wordVectors, vectorDim, filename):
    sys.stderr.write('Writing vectors to file...\n')

    wordVectors.save_word2vec_format(filename, binary=True)

    sys.stderr.write('Finished writing vectors.\n')


def selectTestVocab(vocab, testPhraseFile):
    sys.stderr.write('generating test phrases...\n')
    phrase = set([word for word in vocab if phraseSeparator in word and
                  sum([1 for token in word.split(phraseSeparator) if token in vocab]) ==
                  len(word.split(phraseSeparator))])
    sys.stderr.write('possible phrases count is ' + str(len(phrase)) + '.\n')
    with open(testPhraseFile, 'r') as fp:
        testph = json.load(fp)
        testPhrase = set([re.sub(' ', '|', ph) for ph in testph]) & phrase
    sys.stderr.write('test phrases count is ' + str(len(testPhrase)) + '.\n')

    return testPhrase


def lowercase(s):
    if len(s) == 0:
        return s
    else:
        return s[0].lower() + s[1:]


# link constituent
def linkConstituent(vocab, testVocab, vocabLength):
    missWordCount = 0
    linksCount = 0
    sys.stderr.write('Building linkage between phrase and tokens...\n')
    constituentMatrix = lil_matrix((vocabLength, vocabLength))
    for word in vocab:
        if phraseSeparator in word and word not in testVocab:
            buildLink = True
            phraseIndex = vocab[word]
            tokenList = word.split(phraseSeparator)
            tokenIndexList = []
            for token in tokenList:
                if token in vocab:
                    tokenIndexList.append(vocab[token])
                # elif lowercase(token) in vocab:
                #     tokenIndexList.append(vocab[lowercase(token)])
                else:
                    missWordCount += 1
                    buildLink = False

            if buildLink:
                linksCount += 1
                weightOfConstituent = 1.0/float(len(tokenIndexList)+1.0)
                weightOfIdentity = 1

                # constituentMatrix[phraseIndex, phraseIndex] = weightOfIdentity
                for tokenIndex in tokenIndexList:
                    # constituentMatrix[phraseIndex, tokenIndex] = weightOfConstituent/2.0
                    constituentMatrix[tokenIndex, tokenIndex] = weightOfIdentity
                    for tokenIndex2 in tokenIndexList:
                        if not tokenIndex == tokenIndex2:
                            constituentMatrix[tokenIndex, tokenIndex2] = -1*weightOfConstituent/4.0
                    constituentMatrix[tokenIndex, phraseIndex] = weightOfConstituent/2.0
    sys.stderr.write('Finished building linkage.\n')
    sys.stderr.write('missing ' + str(missWordCount) + ' words\n')
    sys.stderr.write('built ' + str(linksCount) + ' links\n')
    return constituentMatrix.tocoo()


# Return the maximum differential between old and new vectors to check for convergence.
def maxVectorDiff(newVecs, oldVecs):
    maxDiff = 0.0
    for k in newVecs:
        diff = numpy.linalg.norm(newVecs[k] - oldVecs[k])
        if diff > maxDiff:
            maxDiff = diff
    return maxDiff


def update(wordVectors, newSenseVectors):
    for word in newSenseVectors:
        numpy.put(wordVectors[word], range(len(newSenseVectors[word])), newSenseVectors[word])


# Run the retrofitting procedure.
def retrofit(wordVectors, vectorDim, vocab, constituentMatrix, numIters, epsilon):
    sys.stderr.write('Starting the retrofitting procedure...\n')

    # map index to word/phrase
    senseVocab = {vocab[k]: k for k in vocab}
    # initialize sense vectors
    newSenseVectors = {}

    # old sense vectors to check for convergence
    # oldSenseVectors = wordVectors

    # run for a maximum number of iterations
    for it in range(numIters):
        newVector = None
        normalizer = None
        prevRow = None
        isPhrase = None
        sys.stderr.write('Running retrofitting iter '+str(it+1)+'... ')
        # loop through all the non-zero weights in the adjacency matrix
        for row, col, val in izip(constituentMatrix.row, constituentMatrix.col, constituentMatrix.data):
            # a new sense has started
            if row != prevRow:
                if prevRow and phraseSeparator not in senseVocab[prevRow]:
                    newSenseVectors[senseVocab[prevRow]] = newVector/normalizer

                newVector = numpy.zeros(vectorDim, dtype=float)
                normalizer = 0.0
                prevRow = row
                isPhrase = phraseSeparator in senseVocab[row]

            # in the case that senseVocab[row] is not a phrase
            if not isPhrase:
                # add the identity vector
                if row == col:
                    newVector += val * wordVectors[senseVocab[row]]
                    normalizer += val
                # add the constituent vector
                else:
                    if senseVocab[col] not in newSenseVectors:
                        newSenseVectors[senseVocab[col]] = wordVectors[senseVocab[col]]
                    newVector += val * newSenseVectors[senseVocab[col]]
                    if val >= 0:
                        normalizer += val/2
            # in the case that senseVocab[row] is a phrase
            # else:
            #     # add the identity vector
            #     if row == col:
            #         newVector += val * wordVectors[senseVocab[row]]
            #         normalizer += val
            #     # add the constituent vector
            #     else:
            #         newVector += val * newSenseVectors[senseVocab[col]]
            #         normalizer += val

        # diffScore = maxVectorDiff(newSenseVectors, oldSenseVectors)
        # sys.stderr.write('Max vector differential is '+str(diffScore)+'\n')
        sys.stderr.write('Done!\n')
        # if diffScore <= epsilon:
        #    break
        # oldSenseVectors = deepcopy(newSenseVectors)

    update(wordVectors, newSenseVectors)
    sys.stderr.write('Finished running retrofitting.\n')

    return wordVectors


if __name__ == "__main__":
    # parse command line input
    commandParse = readCommandLineInput(sys.argv)
    # failed command line input
    if commandParse == 2:
        sys.exit(2)

    # try opening the specified files
    vocab, vectors, vectorDim = readWordVectors(commandParse[0])
    sys.stderr.write('vocab length is '+str(len(vocab.keys()))+'\n')
    testVocab = selectTestVocab(vocab, commandParse[1])
    # senseVocab, ontologyAdjacency = readOntology(commandParse[1], vectors)
    constituentMatrix = linkConstituent(vocab, testVocab, len(vocab.keys()))
    numIters = commandParse[3]
    epsilon = commandParse[4]

    # run retrofitting and write to output file
    vectors = retrofit(vectors, vectorDim, vocab, constituentMatrix, numIters, epsilon)
    writeWordVectors(vectors, vectorDim, commandParse[2])

    sys.stderr.write('All done!\n')
