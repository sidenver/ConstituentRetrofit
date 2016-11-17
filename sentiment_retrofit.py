# load required libraries
import numpy as np
from numpy import linalg as la
import scipy as sp
import scipy.optimize
import os
from nltk.tokenize import RegexpTokenizer
import re
from collections import Counter


class SentimentRetrofit(object):
    def __init__(self, vectors=None, dim=3, lambDa=0.2):
        self.vectors = vectors
        self.dim = dim
        self.lambDa = lambDa
        # {name: (pos or neg, {word_index: freq)}
        self.docummentDictPos = Counter()
        self.docummentDictNeg = Counter()
        self.word2indx = {}
        self.tokenizer = RegexpTokenizer(r"[\w'-]+")

    def loadVocab(self, fname):
        print 'loading vocab...'
        with open(fname, 'r') as vocabFile:
            lineCount = 0
            for line in vocabFile:
                token = line.strip(' \n')
                if token not in self.word2indx and (self.vectors is None or token in self.vectors):
                    self.word2indx[token] = lineCount
                    lineCount += 1

        if self.vectors is None:
            self.originalVec = np.zeros((len(self.word2indx), self.dim))
        else:
            indx2word = {self.word2indx[word]: word for word in self.word2indx}
            self.originalVec = np.zeros(0)
            for indx in range(len(self.word2indx)):
                word = indx2word[indx]
                vec = self.vectors[word]
                self.originalVec = np.append(self.originalVec, vec)
            self.originalVec = self.originalVec.reshape((len(self.word2indx), self.dim))

    def loadDocument(self, directory, polarity):
        print 'loading document at ' + directory
        for idx, filename in enumerate(os.listdir(directory)):
            if idx > 5000:
                break
            if filename.split('.')[-1] == "txt":
                # {word_index: freq}
                with open(directory + filename, 'r') as file:
                    bow = self.convertDocument2Bow(file.read())
                    if polarity == 'pos':
                        self.docummentDictPos.update(bow)
                    else:
                        self.docummentDictNeg.update(bow)

    def convertDocument2Bow(self, line):
        tokenList = self.tokenizer.tokenize(line.lower())
        bow = Counter()
        for token in tokenList:
            if token in self.word2indx:
                bow[self.word2indx[token]] += 1

        return bow

    def initalVal(self):
        return np.zeros(self.dim + 1 + len(self.word2indx) * self.dim)

    def objectiveSentimentRetrofit(self, param):
        phi = param[:self.dim + 1]
        retroVec = param[self.dim + 1:].reshape((len(self.word2indx), self.dim))
        # {name: (pos or neg, {word_index: freq)}
        score = 0.0
        bow = self.docummentDictPos
        for wordId in bow:
            score += np.log(1.0 + np.exp(-np.dot(phi, np.append(retroVec[wordId], 1.0)))) * bow[wordId]
        bow = self.docummentDictNeg
        for wordId in bow:
            score += np.log(1.0 + np.exp(np.dot(phi, np.append(retroVec[wordId], 1.0)))) * bow[wordId]

        score += self.lambDa * la.norm(retroVec - self.originalVec)**2
        return score

    def word2grad(self, size, phi, vec, wordId):
        grad = np.zeros(size)
        np.put(grad, np.arange(vec.size + 1), np.append(vec, 1.0))
        start = self.dim + 1 + wordId * self.dim
        np.put(grad, np.arange(start, start + self.dim), phi[:-1])
        return grad

    def gradient(self, param):
        phi = param[:self.dim + 1]
        retroVec = param[self.dim + 1:].reshape((len(self.word2indx), self.dim))
        grad = np.zeros(param.size)
        bow = self.docummentDictPos
        for wordId in bow:
            grad += -self.word2grad(param.size, phi, retroVec[wordId], wordId) / (1.0 + np.exp(np.dot(phi, np.append(retroVec[wordId], 1.0)))) * bow[wordId]
        bow = self.docummentDictNeg
        for wordId in bow:
            grad += self.word2grad(param.size, phi, retroVec[wordId], wordId) / (1.0 + np.exp(-np.dot(phi, np.append(retroVec[wordId], 1.0)))) * bow[wordId]

        grad += 2 * self.lambDa * np.append(np.zeros(self.dim + 1), (retroVec - self.originalVec).reshape(len(self.word2indx)*self.dim))
        return grad

    def minimize(self):
        print 'Start minimization...'
        self.optimLBFGS = sp.optimize.fmin_l_bfgs_b(self.objectiveSentimentRetrofit,
                                                    x0=self.initalVal(),
                                                    fprime=self.gradient,
                                                    pgtol=1e-3, disp=True)
        print 'minimization done.'
        newVec = self.optimLBFGS[0][self.dim + 1:].reshape((len(self.word2indx), self.dim))
        self.newVectors = {}
        indx2word = {self.word2indx[word]: word for word in self.word2indx}
        for indx in range(len(self.word2indx)):
            word = indx2word[indx]
            if word not in self.newVectors:
                self.newVectors[word] = newVec[indx]
        print self.optimLBFGS[1:]

    def writeWordVectors(self, outputFile):
        print 'writing to file...'
        with open(outputFile, 'w') as output:
            for vocab in self.newVectors:
                output.write(vocab)
                npVec = self.newVectors[vocab]
                vecStr = np.array2string(npVec, max_line_width='infty', precision=6, suppress_small=True)
                vecStr = vecStr.replace('[', ' ')
                vecStr = re.sub(r' +', ' ', vecStr)
                output.write(vecStr[:-1])
                output.write('\n')

if __name__ == '__main__':
    retrofitter = SentimentRetrofit()
    retrofitter.loadVocab('./aclImdb/imdb.vocab')
    retrofitter.loadDocument('./aclImdb/train/pos/', 'pos')
    retrofitter.loadDocument('./aclImdb/train/neg/', 'neg')
    retrofitter.minimize()
    retrofitter.writeWordVectors('./output/sentimentVec.txt')
