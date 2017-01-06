# load required libraries
import numpy as np
# from autograd import grad
# from numpy import linalg as la
from sklearn import linear_model
import scipy as sp
import scipy.optimize
import os
from nltk.tokenize import RegexpTokenizer
import re
from collections import Counter
from gensim.models import word2vec
import sys

w2vDir = '/fs/clip-scratch/shing/output/sgWordPhrase'


class SentimentRetrofit(object):
    def __init__(self, vectors=None, vocab=None, dim=50, lambDa=0.05):
        self.vectors = vectors
        self.vocab = vocab
        self.dim = dim
        self.lambDa = lambDa
        # {name: (pos or neg, {word_index: freq)}
        self.documentDictPos = Counter()
        self.documentDictNeg = Counter()
        self.lin_clf = linear_model.LogisticRegression()
        self.posX = []
        self.negX = []
        self.wordSet = set()
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
            for indx in range(len(self.word2indx)):
                vec = self.makeRandVector(self.dim)
                self.originalVec = np.append(self.originalVec, vec)
            self.originalVec = self.originalVec.reshape((len(self.word2indx), self.dim))
        else:
            indx2word = {self.word2indx[word]: word for word in self.word2indx}
            self.originalVec = np.zeros(0)
            for indx in range(len(self.word2indx)):
                word = indx2word[indx]
                vec = self.vectors[word]
                self.originalVec = np.append(self.originalVec, vec)
            self.originalVec = self.originalVec.reshape((len(self.word2indx), self.dim))

        print 'original vec is of dimension:', self.originalVec.shape

    def loadDocument(self, directory, polarity):
        print 'loading document at ' + directory
        for idx, filename in enumerate(os.listdir(directory)):
            if idx > 100:
                break
            if filename.split('.')[-1] == "txt":
                # {word_index: freq}
                with open(directory + filename, 'r') as file:
                    line = file.read()
                    bow = self.convertDocument2Bow(line)
                    if self.vectors:
                        vec = self.convertDocument2Vec(line)
                    if polarity == 'pos':
                        self.documentDictPos.update(bow)
                        if self.vectors:
                            self.posX.append(vec)
                    else:
                        self.documentDictNeg.update(bow)
                        if self.vectors:
                            self.negX.append(vec)

    def convertDocument2Bow(self, line):
        tokenList = self.tokenizer.tokenize(line.lower())
        bow = Counter()
        for token in tokenList:
            if token in self.word2indx:
                bow[self.word2indx[token]] += 1
            elif token in self.wordSet:
                self.word2indx[token] = self.wordNum
                self.wordNum += 1

        return bow

    def convertDocument2Vec(self, line):
        tokenList = self.tokenizer.tokenize(line.lower())
        bow = Counter()
        for token in tokenList:
            if token in self.vocab:
                bow[token] += 1.0

        vec = np.zeros(self.dim)
        for word in bow:
            vec += self.vectors[word] * bow[word]
        vec = self.normalize(vec)

        return vec

    def generateSample(self):
        print 'generating samples...'
        self.x = []
        self.y = []
        for pos, neg in zip(self.posX, self.negX):
            self.y.append(1)
            self.x.append(pos)
            self.y.append(0)
            self.x.append(neg)

    def train(self):
        print 'training...'
        self.lin_clf.fit(self.x, self.y)

    def regresserParam(self):
        self.generateSample()
        self.train()
        return np.append(self.lin_clf.coef_[0], [1.0])

    def initalVal(self):
        smallRand = []
        for indx in range(len(self.word2indx)):
                vec = self.makeSmallRandVector(self.dim)
                smallRand = np.append(smallRand, vec)

        if not self.vectors:
            initialVec = self.makeRandVector(self.dim + 1)
            # for indx in range(len(self.word2indx)):
            #     vec = self.makeRandVector(self.dim)
            #     initialVec = np.append(initialVec, vec)
            initialVec = np.append(initialVec, self.originalVec.reshape(len(self.word2indx)*self.dim)+smallRand)
        else:
            initialVec = self.regresserParam()
            vec = self.originalVec.reshape(len(self.word2indx)*self.dim) + smallRand
            initialVec = np.append(initialVec, vec)
        return initialVec

    def makeRandVector(self, dims):
        mu, sigma = 0, 1
        vec = np.random.normal(mu, sigma, dims)
        return self.normalize(vec)

    def makeSmallRandVector(self, dims):
        mu, sigma = 0, 1
        vec = np.random.normal(mu, sigma, dims)
        return self.normalize(vec) * 0.1

    def normalize(self, v):
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v/norm

    def objectiveSentimentRetrofit(self, param):
        phi = param[:self.dim + 1]
        retroVec = param[self.dim + 1:].reshape((len(self.word2indx), self.dim))
        # {name: (pos or neg, {word_index: freq)}
        score = 0.0
        bow = self.documentDictPos
        for wordId in bow:
            score += np.log(1.0 + np.exp(-np.dot(phi[:-1], retroVec[wordId]) - phi[-1])) * bow[wordId]
        bow = self.documentDictNeg
        for wordId in bow:
            score += np.log(1.0 + np.exp(np.dot(phi[:-1], retroVec[wordId]) + phi[-1])) * bow[wordId]

        score += self.lambDa * np.linalg.norm(retroVec - self.originalVec)**2
        return score

    def word2grad(self, size, phi, vec, wordId):
        grad = np.zeros(size)
        np.put(grad, np.arange(vec.size), vec)
        np.put(grad, [vec.size], [1.0])
        start = self.dim + 1 + wordId * self.dim
        np.put(grad, np.arange(start, start + self.dim), phi[:-1])
        return grad

    def gradient(self, param):
        phi = param[:self.dim + 1]
        retroVec = param[self.dim + 1:].reshape((len(self.word2indx), self.dim))
        grad = np.zeros(param.size)
        bow = self.documentDictPos
        for wordId in bow:
            grad += -self.word2grad(param.size, phi, retroVec[wordId], wordId) / (1.0 + np.exp(np.dot(phi[:-1], retroVec[wordId]) + phi[-1])) * bow[wordId]
        bow = self.documentDictNeg
        for wordId in bow:
            grad += self.word2grad(param.size, phi, retroVec[wordId], wordId) / (1.0 + np.exp(-np.dot(phi[:-1], retroVec[wordId]) - phi[-1])) * bow[wordId]

        grad += 2 * self.lambDa * np.append(np.zeros(self.dim + 1), (retroVec - self.originalVec).reshape(len(self.word2indx)*self.dim))
        return grad

    def minimize(self):
        print 'Start minimization...'
        self.optimLBFGS = sp.optimize.fmin_l_bfgs_b(self.objectiveSentimentRetrofit,
                                                    x0=self.initalVal(),
                                                    fprime=self.gradient,
                                                    pgtol=5e-2, disp=True, maxiter=1000)
        print 'minimization done.'
        newVec = self.optimLBFGS[0][self.dim + 1:].reshape((len(self.word2indx), self.dim))
        self.newVectors = {}
        indx2word = {self.word2indx[word]: word for word in self.word2indx}
        for indx in range(len(self.word2indx)):
            word = indx2word[indx]
            if word not in self.newVectors:
                self.newVectors[word] = newVec[indx]
        print self.optimLBFGS[1:]

    def writeWordVectors(self, outputFileOld, outputFileNew):
        print 'writing to file...'
        indx2word = {self.word2indx[word]: word for word in self.word2indx}
        with open(outputFileNew, 'w') as output:
            for index in range(len(indx2word)):
                vocab = indx2word[index]
                output.write(vocab)
                npVec = self.newVectors[vocab]
                vecStr = np.array2string(npVec, max_line_width='infty', precision=8)
                vecStr = vecStr.replace('[', ' ')
                vecStr = re.sub(r' +', ' ', vecStr)
                output.write(vecStr[:-1])
                output.write('\n')
        with open(outputFileOld, 'w') as output:
            for index in range(len(indx2word)):
                vocab = indx2word[index]
                output.write(vocab)
                npVec = self.originalVec[index]
                vecStr = np.array2string(npVec, max_line_width='infty', precision=8)
                vecStr = vecStr.replace('[', ' ')
                vecStr = re.sub(r' +', ' ', vecStr)
                output.write(vecStr[:-1])
                output.write('\n')

    def debug(self):
        for word in self.word2indx:
            print word

    def checkGrad(self):
        print 'start checking'
        initialVec = self.initalVal()
        print 'initialized', initialVec
        print scipy.optimize.check_grad(func=self.objectiveSentimentRetrofit, grad=self.gradient, x0=initialVec)

if __name__ == '__main__':
    sys.stderr.write('Reading vectors from file...\n')

    model = word2vec.Word2Vec.load(w2vDir)
    vectorDim = len(model[model.vocab.iterkeys().next()])
    wordVectors = model
    sys.stderr.write('Loaded vectors from file...\n')
    vocab = {word: model.vocab[word].index for word in model.vocab}
    sys.stderr.write('Finished reading vectors.\n')

    retrofitter = SentimentRetrofit(vectors=wordVectors, vocab=vocab, dim=vectorDim)
    # retrofitter = SentimentRetrofit()
    retrofitter.loadVocab('./aclImdb/imdb.vocab')
    retrofitter.loadDocument('./aclImdb/train/pos/', 'pos')
    retrofitter.loadDocument('./aclImdb/train/neg/', 'neg')
    retrofitter.buildVocab()
    # retrofitter.debug()
    # retrofitter.checkGrad()
    retrofitter.minimize()
    retrofitter.writeWordVectors('./output/sgOld.txt', './output/sgNew.txt')

    # retrofitter.loadVocab('./aclImdb/imdbTest.vocab')
    # retrofitter.loadDocument('./aclImdb/train/testRunPos/', 'pos')
    # retrofitter.loadDocument('./aclImdb/train/testRunNeg/', 'neg')
    # retrofitter.minimize()
    # retrofitter.writeWordVectors('./output/sentimentVec.txt')
