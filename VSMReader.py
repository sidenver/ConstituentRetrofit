import sys
import numpy as np
import gzip
from gensim.models import word2vec


class VSMReader(object):
    def __init__(self, mode=None):
        if not mode:
            mode = 'gensim'

        if mode == "txt":
            self.reader = self.txtReadWordVectors
        elif mode == "binary":
            self.reader = self.binaryReadWordVectors
        elif mode == "gensim":
            self.reader = self.gensimReadWordVectors
        else:
            sys.stderr.write('unknown format specify, use gensim format instead\n')
            self.reader = self.gensimReadWordVectors

    def read(self, vectorFile):
        vocab, vectors, vectorDim = self.reader(vectorFile)
        return vocab, vectors, vectorDim

    def txtReadWordVectors(self, filename):
        sys.stderr.write('Reading vectors from file...\n')

        if filename.endswith('.gz'):
            fileObject = gzip.open(filename, 'r')
        else:
            fileObject = open(filename, 'r')

        vectorDim = int(fileObject.readline().strip().split()[1])
        vectors = np.loadtxt(filename, dtype=float, comments=None, skiprows=1, usecols=range(1, vectorDim+1))
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

    def binaryReadWordVectors(self, filename):
        sys.stderr.write('Reading vectors from file...\n')

        model = word2vec.Word2Vec.load_word2vec_format(filename, binary=True)

        vectorDim = len(model[model.vocab.iterkeys().next()])
        wordVectors = model
        sys.stderr.write('Loaded vectors from file...\n')

        vocab = {word: model.vocab[word].index for word in model.vocab}

        sys.stderr.write('Finished reading vectors.\n')

        return vocab, wordVectors, vectorDim

    def gensimReadWordVectors(self, filename):
        sys.stderr.write('Reading vectors from file...\n')

        model = word2vec.Word2Vec.load(filename)

        vectorDim = len(model[model.vocab.iterkeys().next()])
        wordVectors = model
        sys.stderr.write('Loaded vectors from file...\n')

        vocab = {word: model.vocab[word].index for word in model.vocab}

        sys.stderr.write('Finished reading vectors.\n')

        return vocab, wordVectors, vectorDim
