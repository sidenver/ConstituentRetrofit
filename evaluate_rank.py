import constituentretrofit as consfit
import sys
import getopt
import numpy as np

help_message = '''
$ python evaluate_rank.py -v <vectorsFile> [-o outputFile] [-h]
-v or --vectors to specify path to the word vectors input file (gzip or txt files are acceptable)
-o or --output to optionally set path to output evaluation result (<vectorsFile>.result is used by default)
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
            opts, _ = getopt.getopt(argv[1:], "hv:o:", ["help", "vectors=",
                                                        "output="])
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
                outputFile = vectorsFile + '.result'
            return (vectorsFile, outputFile)

    except Usage, err:
        print str(err.msg)
        return 2


def selectTestVocab(vocab, testNum=4600):
    sys.stderr.write('generating test phrases...\n')
    phrase = [word for word in vocab if '_' in word and
              sum([1 for token in word.split('_') if token in vocab]) == len(word.split('_'))]
    sys.stderr.write('possible test phrases count is ' + str(len(phrase)) + '.\n')
    return np.random.choice(phrase, testNum, replace=False)


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v/norm


def calMedRank(ranks):
    P = float(len(ranks))
    medRanks = [(1.0 - float(r)/P) for r in ranks]
    return sum(medRanks)/float(len(medRanks))


def calMRR(ranks):
    mrrs = [(1.0/float(r)) for r in ranks]
    return sum(mrrs)/float(len(mrrs))


def evaluate(testVocab, vocab, vectors):
    sys.stderr.write('start evaluation...\n')
    sys.stderr.write('normalizing test phrase vectors...\n')
    normalizedTestVec = {word: normalize(vectors[word]) for word in testVocab}
    sys.stderr.write('finished normalized.\n')
    ranks = []
    sys.stderr.write('calculating ranks...\n')
    for trueWord in testVocab:
        tokens = trueWord.split('_')
        # what if token is not found in vectors?
        estimateVec = normalize(sum([vectors[token] for token in tokens])/float(len(tokens)))
        cosineSimlarities = []
        trueValue = None
        for testWord in testVocab:
            cosineSimlarity = np.dot(estimateVec, normalizedTestVec[testWord])
            cosineSimlarities.append(cosineSimlarity)
            if testWord == trueWord:
                trueValue = cosineSimlarity

        rank = sum([1 for cosine in cosineSimlarities if trueValue <= cosine])
        ranks.append(rank)
    sys.stderr.write('ranks calculated.\n')

    medRank = calMedRank(ranks)
    mrr = calMRR(ranks)
    perfect = float(sum([1 for oneRank in ranks if oneRank == 1]))/float(len(ranks))
    return medRank, mrr, perfect


if __name__ == "__main__":
    # parse command line input
    commandParse = readCommandLineInput(sys.argv)
    # failed command line input
    if commandParse == 2:
        sys.exit(2)

    # try opening the specified files

    vocab, vectors, vectorDim = consfit.readWordVectors(commandParse[0])
    testVocab = selectTestVocab(vocab, 4600)
    medRank, mrr, perfect = evaluate(testVocab, vocab, vectors)

    sys.stderr.write('vocab length is '+str(len(vocab.keys()))+'\n')
    sys.stderr.write('vector length is '+str(len(vectors.keys()))+'\n')
    sys.stderr.write('phrase count is '+str(sum([1 for token in vocab if '_' in token]))+'\n')
    sys.stderr.write('MedRank is '+str(100*medRank)+'\n')
    sys.stderr.write('MRR is '+str(100*mrr)+'\n')
    sys.stderr.write('Perfect is '+str(100*perfect)+'\n')
    sys.stderr.write('All done!\n')
