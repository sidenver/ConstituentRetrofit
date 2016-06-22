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
import gzip

from scipy.sparse import lil_matrix
from copy import deepcopy
from itertools import izip


help_message = '''
$ python constituentretrofit.py -v <vectorsFile> [-o outputFile] [-n numIters] [-e epsilon] [-h]
-v or --vectors to specify path to the word vectors input file (gzip or txt files are acceptable)
-o or --output to optionally set path to output word sense vectors file (<vectorsFile>.sense is used by default)
-n or --numiters to optionally set the number of retrofitting iterations (10 is the default)
-e or --epsilon to optionally set the convergence threshold (0.001 is the default)
-h or --help (this message is displayed)
'''

senseSeparator = '%'
valueSeparator = '#'


class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg


# Read command line arguments
def readCommandLineInput(argv):
    try:
        try:
            # specify the possible option switches
            opts, _ = getopt.getopt(argv[1:], "hv:o:n:e:", ["help", "vectors=",
                                                            "output=", "numiters=", "epsilon="])
        except getopt.error, msg:
            raise Usage(msg)

        # default values
        vectorsFile = None
        ontologyFile = None  # not used
        outputFile = None
        numIters = 10
        epsilon = 0.001

        setOutput = False
        # option processing
        for option, value in opts:
            if option in ("-h", "--help"):
                raise Usage(help_message)
            elif option in ("-v", "--vectors"):
                vectorsFile = value
            elif option in ("-q", "--ontology"):
                # not used
                ontologyFile = value
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
            else:
                raise Usage(help_message)

        if (vectorsFile is None):
            raise Usage(help_message)
        else:
            if not setOutput:
                outputFile = vectorsFile + '.cons'
            return (vectorsFile, ontologyFile, outputFile, numIters, epsilon)

    except Usage, err:
        print str(err.msg)
        return 2


# Read all the word vectors from file.
def readWordVectors(filename):
    sys.stderr.write('Reading vectors from file...\n')

    if filename.endswith('.gz'):
        fileObject = gzip.open(filename, 'r')
    else:
        fileObject = open(filename, 'r')

    vectorDim = int(fileObject.readline().strip().split()[1])
    vectors = numpy.loadtxt(filename, dtype=float, comments=None, skiprows=1, usecols=range(1, vectorDim+1))
    sys.stderr.write('Loaded vectors from file...\n')

    wordVectors = {}
    vocab = {}
    lineNum = 0
    for line in fileObject:
        word = line.strip().split()[0]
        vocab[word] = lineNum
        wordVectors[word] = vectors[lineNum]
        lineNum += 1

    sys.stderr.write('Finished reading vectors.\n')

    fileObject.close()
    return vocab, wordVectors, vectorDim


# Write word vectors to file
def writeWordVectors(wordVectors, vectorDim, filename):
    sys.stderr.write('Writing vectors to file...\n')

    if filename.endswith('.gz'):
        fileObject = gzip.open(filename, 'w')
    else:
        fileObject = open(filename, 'w')

    fileObject.write(str(len(wordVectors.keys())) + ' ' + str(vectorDim) + '\n')
    for word in wordVectors:
        fileObject.write(word + ' ' + ' '.join(map(str, wordVectors[word])) + '\n')
    fileObject.close()

    sys.stderr.write('Finished writing vectors.\n')


def lowercase(s):
    if len(s) == 0:
        return s
    else:
        return s[0].lower() + s[1:]


# link constituent
def linkConstituent(vocab, vocabLength):
    missWordCount = 0
    sys.stderr.write('Building linkage between phrase and tokens...\n')
    constituentMatrix = lil_matrix((vocabLength, vocabLength))
    for word in vocab:
        if '_' in word:
            buildLink = True
            phraseIndex = vocab[word]
            tokenList = word.split('_')
            tokenIndexList = []
            for token in tokenList:
                if token in vocab:
                    tokenIndexList.append(vocab[token])
                # elif lowercase(token) in vocab:
                #     tokenIndexList.append(vocab[lowercase(token)])
                else:
                    sys.stderr.write('Missing ' + token + '\n')
                    missWordCount += 1
                    buildLink = False

            if buildLink:
                weightOfConstituent = 1.0/float(len(tokenIndexList)+1.0)
                weightOfIdentity = 1

                constituentMatrix[phraseIndex, phraseIndex] = weightOfIdentity
                for tokenIndex in tokenIndexList:
                    constituentMatrix[phraseIndex, tokenIndex] = weightOfConstituent/2.0
                    constituentMatrix[tokenIndex, tokenIndex] = weightOfIdentity
                    for tokenIndex2 in tokenIndexList:
                        if not tokenIndex == tokenIndex2:
                            constituentMatrix[tokenIndex, tokenIndex2] = -1*weightOfConstituent/4.0
                    constituentMatrix[tokenIndex, phraseIndex] = weightOfConstituent/2.0
    sys.stderr.write('Finished building linkage.\n')
    sys.stderr.write('missing ' + str(missWordCount) + ' words\n')
    return constituentMatrix.tocoo()


# Return the maximum differential between old and new vectors to check for convergence.
def maxVectorDiff(newVecs, oldVecs):
    maxDiff = 0.0
    for k in newVecs:
        diff = numpy.linalg.norm(newVecs[k] - oldVecs[k])
        if diff > maxDiff:
            maxDiff = diff
    return maxDiff


# Run the retrofitting procedure.
def retrofit(wordVectors, vectorDim, vocab, constituentMatrix, numIters, epsilon):
    sys.stderr.write('Starting the retrofitting procedure...\n')

    # map index to word/phrase
    senseVocab = {vocab[k]: k for k in vocab}
    # initialize sense vectors
    newSenseVectors = {senseVocab[k]: wordVectors[senseVocab[k]]
                       for k in senseVocab}

    # create a copy of the sense vectors to check for convergence
    oldSenseVectors = deepcopy(newSenseVectors)

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
                if prevRow and '_' not in senseVocab[prevRow]:
                    newSenseVectors[senseVocab[prevRow]] = newVector/normalizer

                newVector = numpy.zeros(vectorDim, dtype=float)
                normalizer = 0.0
                prevRow = row
                isPhrase = '_' in senseVocab[row]

            # in the case that senseVocab[row] is not a phrase
            if not isPhrase:
                # add the identity vector
                if row == col:
                    newVector += val * wordVectors[senseVocab[row]]
                    normalizer += val
                # add the constituent vector
                else:
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

        diffScore = maxVectorDiff(newSenseVectors, oldSenseVectors)
        sys.stderr.write('Max vector differential is '+str(diffScore)+'\n')
        if diffScore <= epsilon:
            break
        oldSenseVectors = deepcopy(newSenseVectors)

    sys.stderr.write('Finished running retrofitting.\n')

    return newSenseVectors


if __name__ == "__main__":
    # parse command line input
    commandParse = readCommandLineInput(sys.argv)
    # failed command line input
    if commandParse == 2:
        sys.exit(2)

    # try opening the specified files

    vocab, vectors, vectorDim = readWordVectors(commandParse[0])
    sys.stderr.write('vocab length is '+str(len(vocab.keys()))+'\n')
    sys.stderr.write('vector length is '+str(len(vectors.keys()))+'\n')
    # senseVocab, ontologyAdjacency = readOntology(commandParse[1], vectors)
    constituentMatrix = linkConstituent(vocab, len(vocab.keys()))
    numIters = commandParse[3]
    epsilon = commandParse[4]

    # run retrofitting and write to output file
    writeWordVectors(retrofit(vectors, vectorDim, vocab, constituentMatrix, numIters, epsilon),
                     vectorDim, commandParse[2])

    sys.stderr.write('All done!\n')
