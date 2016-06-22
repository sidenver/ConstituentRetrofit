import evaluate_rank as evaRank
import constituentretrofit as consfit
import sys
import getopt
import numpy as np

help_message = '''
$ python splitTestTrain.py -v <vectorsFile> [-h]
-v or --vectors to specify path to the word vectors input file (gzip or txt files are acceptable)
-h or --help (this message is displayed)
'''


class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg


# Read command line arguments
def readCommandLineInput(argv):
    try:
        try:
            # specify the possible option switches
            opts, _ = getopt.getopt(argv[1:], "hv:", ["help", "vectors="])
        except getopt.error, msg:
            raise Usage(msg)

        # default values
        vectorsFile = None

        setOutput = False
        # option processing
        for option, value in opts:
            if option in ("-h", "--help"):
                raise Usage(help_message)
            elif option in ("-v", "--vectors"):
                vectorsFile = value
            else:
                raise Usage(help_message)

        if (vectorsFile is None):
            raise Usage(help_message)
        else:
            if not setOutput:
                outputFileTest = vectorsFile + '.test'
                outputFileTrain = vectorsFile + '.train'
            return (vectorsFile, outputFileTest, outputFileTrain)

    except Usage, err:
        print str(err.msg)
        return 2


if __name__ == '__main__':
    # parse command line input
    commandParse = readCommandLineInput(sys.argv)
    # failed command line input
    if commandParse == 2:
        sys.exit(2)

    vocab, vectors, vectorDim = consfit.readWordVectors(commandParse[0])
    testVocab = evaRank.selectTestVocab(vocab, 4600)
    consfit.writeWordVectors({word: vectors[word] for word in vectors if word in testVocab}, vectorDim, commandParse[1])
    consfit.writeWordVectors({word: vectors[word] for word in vectors if word not in testVocab}, vectorDim, commandParse[2])

    sys.stderr.write('All done!\n')
