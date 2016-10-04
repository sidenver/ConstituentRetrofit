"""Usage:
    constituentretrofit_word2vec.py -v <vectorsFile> [options]
    constituentretrofit_word2vec.py (-h | --help)

Arguments:
-v <vectorsFile>             to specify word2vec input file

Options:
-t <testPhrase>              specify test phrases to leave out
-o <outputFile>              set output word sense vectors file (<vectorsFile>.sense)
-n <numiters>                set the number of retrofitting iterations [default: 8]
-e <epsilon>                 set the convergence threshold [default: 0.001]
--phrase <phraseSeparator>   phrase separator [default: |]
-f <inputFormat>             can be set to gensim, binary, or txt [default: gensim]
-h --help                    (this message is displayed)

Copyright (C) 2015 Sujay Kumar Jauhar <sjauhar@cs.cmu.edu>
Modified to include constituent by Han-Chin Shing <shing@cs.umd.edu>
Licenced under the Apache Licence, v2.0 - http://www.apache.org/licenses/LICENSE-2.0

"""

import sys
import numpy
# import gzip
import json

from scipy.sparse import lil_matrix
# from copy import deepcopy
from itertools import izip
import re
from docopt import docopt


phraseSeparator = '|'


def writeWordVectors(wordVectors, vectorDim, filename):
    sys.stderr.write('Writing vectors to file...\n')

    wordVectors.save(filename)

    sys.stderr.write('Finished writing vectors.\n')


def selectTestVocab(vocab, testPhraseFile):
    phrase = set([word for word in vocab if phraseSeparator in word and
                  sum([1 for token in word.split(phraseSeparator) if token in vocab]) ==
                  len(word.split(phraseSeparator))])
    sys.stderr.write('possible phrases count is ' + str(len(phrase)) + '.\n')
    if testPhraseFile:
        sys.stderr.write('generating test phrases...\n')
        with open(testPhraseFile, 'r') as fp:
            testph = json.load(fp)
            testPhrase = set([re.sub(' ', '|', ph) for ph in testph]) & phrase
        sys.stderr.write('test phrases count is ' + str(len(testPhrase)) + '.\n')

        return testPhrase
    else:
        return set()


def lowercase(s):
    if len(s) == 0:
        return s
    else:
        return s[0].lower() + s[1:]


# link constituent
def linkConstituent(vocab, testVocab, vocabLength, fixPhrase=False):
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
                weightOfConstituent = 1.0 / float(len(tokenIndexList) + 1.0)
                weightOfIdentity = 1

                if not fixPhrase:
                    constituentMatrix[phraseIndex, phraseIndex] = weightOfIdentity
                for tokenIndex in tokenIndexList:
                    if not fixPhrase:
                        constituentMatrix[phraseIndex, tokenIndex] = weightOfConstituent / 2.0
                    constituentMatrix[tokenIndex, tokenIndex] = weightOfIdentity
                    for tokenIndex2 in tokenIndexList:
                        if not tokenIndex == tokenIndex2:
                            constituentMatrix[tokenIndex, tokenIndex2] = -1 * weightOfConstituent / 4.0
                    constituentMatrix[tokenIndex, phraseIndex] = weightOfConstituent / 2.0
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
def retrofit(wordVectors, vectorDim, vocab, constituentMatrix, numIters, epsilon, fixPhrase=False):
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
        sys.stderr.write('Running retrofitting iter ' + str(it + 1) + '... ')
        # loop through all the non-zero weights in the adjacency matrix
        for row, col, val in izip(constituentMatrix.row, constituentMatrix.col, constituentMatrix.data):
            # a new sense has started
            if row != prevRow:
                if prevRow and (not fixPhrase or phraseSeparator not in senseVocab[prevRow]):
                    newSenseVectors[senseVocab[prevRow]] = newVector / normalizer

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
                        normalizer += val / 2
            # in the case that senseVocab[row] is a phrase
            elif not fixPhrase:
                # add the identity vector
                if row == col:
                    newVector += val * wordVectors[senseVocab[row]]
                    normalizer += val
                # add the constituent vector
                else:
                    if senseVocab[col] not in newSenseVectors:
                        newSenseVectors[senseVocab[col]] = wordVectors[senseVocab[col]]
                    newVector += val * newSenseVectors[senseVocab[col]]
                    normalizer += val

        # diffScore = maxVectorDiff(newSenseVectors, oldSenseVectors)
        # sys.stderr.write('Max vector differential is '+str(diffScore)+'\n')
        sys.stderr.write('Done!\n')
        # if diffScore <= epsilon:
        #    break
        # oldSenseVectors = deepcopy(newSenseVectors)

    update(wordVectors, newSenseVectors)
    sys.stderr.write('Finished running retrofitting.\n')

    return wordVectors


def setup(commandParse):
    outputFile = commandParse["-o"] if commandParse["-o"] else commandParse["-v"] + ".cons"
    global phraseSeparator
    phraseSeparator = commandParse["--phrase"]

    if commandParse["-f"] == "txt":
        import constituentretrofit_fixed as consfit
    elif commandParse["-f"] == "binary":
        import constituentretrofit_fixed_word2vec as consfit
    elif commandParse["-f"] == "gensim":
        import constituentretrofit_fixed_word2vec_native as consfit
    else:
        sys.stderr.write('unknown format specify, use gensim format instead\n')
        import constituentretrofit_fixed_word2vec_native as consfit

    return outputFile, consfit


if __name__ == "__main__":
    # parse command line input
    commandParse = docopt(__doc__)
    print commandParse
    outputFile, consfit = setup(commandParse)
    print outputFile
    print consfit

    # try opening the specified files
    vocab, vectors, vectorDim = consfit.readWordVectors(commandParse["-v"])
    # vocab is {word: frequency rank}
    # vectors is {word: vector}
    sys.stderr.write('vocab length is ' + str(len(vocab.keys())) + '\n')
    testVocab = selectTestVocab(vocab, commandParse["-t"])

    constituentMatrix = linkConstituent(vocab, testVocab, len(vocab.keys()))

    numIters = int(commandParse["-n"])
    epsilon = float(commandParse["-e"])
    # run retrofitting and write to output file
    vectors = retrofit(vectors, vectorDim, vocab, constituentMatrix, numIters, epsilon)
    writeWordVectors(vectors, vectorDim, outputFile)

    sys.stderr.write('All done!\n')
