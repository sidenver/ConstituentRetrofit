import constituentretrofit_fixed_word2vec_native as consfit
import sys
import getopt
import numpy as np

# take a POS tagged word2vec format file and turn it into a txt file
help_message = '''
$ python untag.py -v <vectorsFile> [-o outputFile] [-h]
-v or --vectors to specify path to the word vectors input file (gzip or txt files are acceptable)
-o or --output to optionally set path to output word sense vectors file (<vectorsFile>.sense is used by default)
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
            opts, _ = getopt.getopt(argv[1:], "hv:o:", ["help", "vectors=", "output="])
        except getopt.error, msg:
            raise Usage(msg)

        # default values
        vectorsFile = None
        outputFile = None

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
            else:
                raise Usage(help_message)

        if (vectorsFile is None):
            raise Usage(help_message)
        else:
            if not setOutput:
                outputFile = vectorsFile + '.txt'
            return (vectorsFile, outputFile)

    except Usage, err:
        print str(err.msg)
        return 2


def untag(vocabs):
    untagVocab = {}
    for tagVocab in vocabs:
        if phraseSeparator not in tagVocab:
            vocab = tagVocab.split('_')[0]
            if vocab in untagVocab:
                untagVocab[vocab].append(tagVocab)
            else:
                untagVocab[vocab] = [tagVocab]
    return untagVocab


def untagWithVectors(untagDict, vocab, vectors):
    untagVectors = {}
    for untagVocab in untagDict:
        if len(untagDict[untagVocab]) > 1:
            def getFreqOfVocab(x):
                vocab[x]
            mostFreqVocab = min(untagDict[untagVocab], key=getFreqOfVocab)
            untagVectors[untagVocab] = vectors[mostFreqVocab]
        else:
            untagVectors[untagVocab] = vectors[untagDict[untagVocab][0]]
    return untagVectors


def writeWordVectors(vectors, vectorDim, outputFile):
    sys.stderr.write('Writing untagged vec to file\n')
    with open(outputFile, 'w') as output:
        for vocab in vectors:
            output.write(vocab)
            npVec = vectors[vocab]
            vecStr = np.array2string(npVec, max_line_width='infty', precision=6)
            vecStr = vecStr.replace('  ', ' ')
            output.write(vecStr[1:-1])
            output.write('\n')

if __name__ == '__main__':
    commandParse = readCommandLineInput(sys.argv)
    if commandParse == 2:
        sys.exit(2)

    vocab, vectors, vectorDim = consfit.readWordVectors(commandParse[0])
    # vocab is {word: frequency rank}
    # vectors is {word: vector}
    sys.stderr.write('vocab length is '+str(len(vocab.keys()))+'\n')
    untagDict = untag(vocab)
    # untag and write to output file
    untagVectors = untagWithVectors(untagDict, vocab, vectors)
    writeWordVectors(untagVectors, vectorDim, commandParse[1])

    sys.stderr.write('All done!\n')
