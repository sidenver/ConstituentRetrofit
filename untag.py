"""Usage:
    untag.py -v <vectorsFile> [options]
    untag.py -h | --help

take a POS tagged word2vec format file and turn it into a txt file

Options:
-h --help                  show this message
-f, --filter=FILTERFILE    optionally set path to filter word vectors txt file
-o, --output=OUTPUTFILE    optionally set path to output untagged word vectors txt file
--phrase=SEPARATOR         phrase separator, default '_'
--tag=SEPARATOR            tag separator, default '/'
--gensim                   the default, read in gensim word2vec file
--binary                   read in native bainary word2vec file
--txt                      read in txt format word2vec file

"""

import sys
import numpy as np
import re
from docopt import docopt


phraseSeparator = '_'
tagSeparator = '/'


def readInFilterFile(filterFile):
    filterSet = set()
    with open(filterFile, 'r') as f:
        for line in f:
            filterSet.add(line.rstrip('\n'))
    return filterSet


def untag(vocabs):
    sys.stderr.write('Untagging vec\n')
    untagVocab = {}
    for tagVocab in vocabs:
        if phraseSeparator not in tagVocab:
            vocab = tagVocab.split(tagSeparator)[0]
            if vocab in untagVocab:
                untagVocab[vocab].append(tagVocab)
            else:
                untagVocab[vocab] = [tagVocab]
    return untagVocab


def untagWithVectors(untagDict, vocab, vectors, filterSet=None):
    untagVectors = {}
    for untagVocab in untagDict:
        if filterSet is None or untagVocab in filterSet:
            if len(untagDict[untagVocab]) > 1:
                def getFreqOfVocab(x):
                    vocab[x]
                mostFreqVocab = min(untagDict[untagVocab], key=getFreqOfVocab)
                untagVectors[untagVocab] = vectors[mostFreqVocab]
            else:
                untagVectors[untagVocab] = vectors[untagDict[untagVocab][0]]
    sys.stderr.write('Untagged!\n')
    return untagVectors


def writeWordVectors(vectors, vectorDim, outputFile):
    sys.stderr.write('Writing untagged vec to file\n')
    with open(outputFile, 'w') as output:
        for vocab in vectors:
            output.write(vocab)
            npVec = vectors[vocab]
            vecStr = np.array2string(npVec, max_line_width='infty', precision=6, suppress_small=True)
            vecStr = vecStr.replace('[', ' ')
            vecStr = re.sub(r' +', ' ', vecStr)
            output.write(vecStr[:-1])
            output.write('\n')


def setUp(commandParse):
    if not commandParse['--output']:
        outputFile = commandParse['<vectorsFile>'] + '.txt'
    else:
        outputFile = commandParse['--output']

    if commandParse['--txt']:
        import constituentretrofit_fixed as consfit
    elif commandParse['--binary']:
        import constituentretrofit_fixed_word2vec as consfit
    else:
        # the default --gensim format
        import constituentretrofit_fixed_word2vec_native as consfit

    if commandParse['--phrase']:
        global phraseSeparator
        phraseSeparator = commandParse['--phrase']
    if commandParse['--tag']:
        global tagSeparator
        tagSeparator = commandParse['--tag']
    return outputFile, consfit

if __name__ == '__main__':
    commandParse = docopt(__doc__)
    print commandParse
    outputFile, consfit = setUp(commandParse)

    vocab, vectors, vectorDim = consfit.readWordVectors(commandParse['<vectorsFile>'])
    # vocab is {word: frequency rank}
    # vectors is {word: vector}
    sys.stderr.write('vocab length is '+str(len(vocab.keys()))+'\n')
    filterSet = None
    if commandParse['--filter']:
        filterSet = readInFilterFile(commandParse['--filter'])
    untagDict = untag(vocab)
    # untag and write to output file
    untagVectors = untagWithVectors(untagDict, vocab, vectors, filterSet)
    writeWordVectors(untagVectors, vectorDim, outputFile)

    sys.stderr.write('All done!\n')
