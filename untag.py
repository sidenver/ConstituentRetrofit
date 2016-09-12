import constituentretrofit_fixed_word2vec_native as consfit
import sys
import getopt
import numpy as np
import re

# take a POS tagged word2vec format file and turn it into a txt file
help_message = '''
$ python untag.py -v <vectorsFile> [-f filterFile] [-o outputFile] [-h]
-v or --vectors to specify path to the word2vectors input file
-f or --filter to optionally set path to filter word vectors txt file
-o or --output to optionally set path to output untagged word vectors txt file
-h or --help (this message is displayed)
'''

phraseSeparator = '|'
tagSeparator = '_'


class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg


# Read command line arguments
def readCommandLineInput(argv):
    try:
        try:
            # specify the possible option switches
            opts, _ = getopt.getopt(argv[1:], "hv:f:o:", ["help", "vectors=", "filter=", "output="])
        except getopt.error, msg:
            raise Usage(msg)

        # default values
        vectorsFile = None
        outputFile = None
        filterFile = None

        setOutput = False
        # option processing
        for option, value in opts:
            if option in ("-h", "--help"):
                raise Usage(help_message)
            elif option in ("-v", "--vectors"):
                vectorsFile = value
            elif option in ("-f", "--filter"):
                filterFile = value
            elif option in ("-o", "--output"):
                outputFile = value
                setOutput = True
            else:
                raise Usage(help_message)

        if (vectorsFile is None):
            raise Usage(help_message)
        else:
            if not setOutput:
                outputFile = vectorsFile + '.txt'
            return (vectorsFile, outputFile, filterFile)

    except Usage, err:
        print str(err.msg)
        return 2


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
            vocab = tagVocab.split('_')[0]
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

if __name__ == '__main__':
    commandParse = readCommandLineInput(sys.argv)
    if commandParse == 2:
        sys.exit(2)

    vocab, vectors, vectorDim = consfit.readWordVectors(commandParse[0])
    # vocab is {word: frequency rank}
    # vectors is {word: vector}
    sys.stderr.write('vocab length is '+str(len(vocab.keys()))+'\n')
    filterSet = None
    if commandParse[2]:
        filterSet = readInFilterFile(commandParse[2])
    untagDict = untag(vocab)
    # untag and write to output file
    untagVectors = untagWithVectors(untagDict, vocab, vectors, filterSet)
    writeWordVectors(untagVectors, vectorDim, commandParse[1])

    sys.stderr.write('All done!\n')
