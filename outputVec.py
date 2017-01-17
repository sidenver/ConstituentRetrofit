# load required libraries
import numpy as np
import os
from nltk.tokenize import RegexpTokenizer
import re
from collections import Counter
from gensim.models import word2vec
import sys
import pandas as pd

w2vDir = '/fs/clip-scratch/shing/output/sgWordPhrase'


class SentimentRetrofit(object):
    def __init__(self, vectors=None, vocab=None, dim=50, lambDa=0.05, vocabSize=16000):
        self.vectors = vectors
        self.vocab = vocab
        self.dim = dim
        self.lambDa = lambDa
        # {name: (pos or neg, {word_index: freq)}
        self.documentDictPos = Counter()
        self.documentDictNeg = Counter()
        self.documentDictTol = Counter()
        self.wordSet = set()
        self.vocabSize = vocabSize
        self.word2indx = {}
        self.wordNum = 0
        self.tokenizer = RegexpTokenizer(r"[\w'-]+")

    def loadVocab(self, fname):
        """
        load vocab in imdb
        """
        print 'loading imdb vocab...'
        with open(fname, 'r') as vocabFile:
            for line in vocabFile:
                token = line.strip(' \n')
                if len(token) > 0 and token not in self.wordSet and (self.vocab is None or token in self.vocab):
                    self.wordSet.add(token)

    def buildVocab(self):

        if self.vectors is None:
            # self.originalVec = np.zeros((len(self.word2indx), self.dim))
            self.originalVec = []
            for indx in range(self.vocabSize):
                vec = self.makeRandVector(self.dim)
                self.originalVec = np.append(self.originalVec, vec)
            self.originalVec = self.originalVec.reshape((self.vocabSize, self.dim))
        else:
            self.documentDictTol.update(self.documentDictPos)
            self.documentDictTol.update(self.documentDictNeg)
            self.wordList = [word for word, freq in self.documentDictTol.most_common(self.vocabSize)]
            self.posWeight = [self.documentDictPos[word] for word in self.wordList]
            self.negWeight = [self.documentDictNeg[word] for word in self.wordList]
            self.originalVec = np.zeros(0)
            for word in self.wordList:
                vec = self.vectors[word]
                self.originalVec = np.append(self.originalVec, vec)
            self.originalVec = self.originalVec.reshape((self.vocabSize, self.dim))

        print 'original vec is of dimension:', self.originalVec.shape

    def loadDocument(self, directory, polarity):
        print 'loading document at ' + directory
        idx = 0
        for filename in os.listdir(directory):
            if idx >= 10000:
                break
            if filename.split('.')[-1] == "txt":
                idx += 1
                # {word_index: freq}
                with open(directory + filename, 'r') as file:
                    line = file.read()
                    bow = self.convertDocument2Bow(line)
                    if polarity == 'pos':
                        self.documentDictPos.update(bow)
                    else:
                        self.documentDictNeg.update(bow)

    def convertDocument2Bow(self, line):
        tokenList = self.tokenizer.tokenize(line.lower())
        bow = Counter()
        for token in tokenList:
            if token in self.wordSet:
                bow[token] += 1

        return bow

    def makeRandVector(self, dims):
        mu, sigma = 0, 1
        vec = np.random.normal(mu, sigma, dims)
        return self.normalize(vec)

    def normalize(self, v):
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

    def writeWordVectors(self, outputFileW2V, outputFilePos, outputFileNeg):
        print 'writing to file...'
        df = pd.DataFrame(data=self.originalVec,
                          index=self.wordList)
        df.to_csv(outputFileW2V, encoding='utf-8')

        dfPos = pd.DataFrame(self.posWeight)
        dfPos.to_csv(outputFilePos)

        dfNeg = pd.DataFrame(self.negWeight)
        dfNeg.to_csv(outputFileNeg)


if __name__ == '__main__':
    sys.stderr.write('Reading vectors from file...\n')

    model = word2vec.Word2Vec.load(w2vDir)
    vectorDim = len(model[model.vocab.iterkeys().next()])
    wordVectors = model
    sys.stderr.write('Loaded vectors from file...\n')
    vocab = {word: model.vocab[word].index for word in model.vocab}
    sys.stderr.write('Finished reading vectors.\n')

    retrofitter = SentimentRetrofit(vectors=wordVectors, vocab=vocab, dim=vectorDim)
    retrofitter.loadVocab('./aclImdb/imdb.vocab')
    retrofitter.loadDocument('./aclImdb/train/pos/', 'pos')
    retrofitter.loadDocument('./aclImdb/train/neg/', 'neg')
    retrofitter.buildVocab()
    retrofitter.writeWordVectors('./output/w2v.csv', './output/posList.csv', './output/negList.csv')
