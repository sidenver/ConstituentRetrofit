# load required libraries
import autograd.numpy as np
from autograd import grad
from numpy import linalg as la
import scipy as sp
import scipy.optimize
import os
from nltk.tokenize import RegexpTokenizer
import re


class SentimentRetrofit(object):
    def __init__(self, vectors=None, dim=300, lambDa=0.05):
        self.vectors = vectors
        self.dim = dim
        self.lambDa = lambDa
        # {name: (pos or neg, {word_index: freq)}
        self.docummentDict = {}
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
                if filename.split('.')[0] not in self.docummentDict:
                    # {word_index: freq}
                    with open(directory + filename, 'r') as file:
                        bow = self.convertDocument2Bow(file.read())
                        self.docummentDict[filename.split('.')[0]] = (polarity, bow)

    def convertDocument2Bow(self, line):
        tokenList = self.tokenizer.tokenize(line.lower())
        bow = {}
        for token in tokenList:
            if token in self.word2indx:
                if self.word2indx[token] in bow:
                    bow[self.word2indx[token]] += 1
                else:
                    bow[self.word2indx[token]] = 1
        return bow

    def initalVal(self):
        return (np.zeros(self.dim + 1), np.zeros(len(self.word2indx), self.dim))

    def objectiveSentimentRetrofit(self, param):
        phi = param[0]
        retroVec = param[1]
        # {name: (pos or neg, {word_index: freq)}
        score = 0.0
        for document in self.docummentDict:
            polarity = self.docummentDict[document][0]
            bow = self.docummentDict[document][1]
            if polarity == 1:
                for word in bow:
                    score += np.log(1.0 + np.exp(-np.dot(phi, np.append(retroVec[word], 1.0)))) * bow[word]
            else:
                for word in bow:
                    score += np.log(1.0 + np.exp(np.dot(phi, np.append(retroVec[word], 1.0)))) * bow[word]

        score += self.lambDa * la.norm(retroVec - self.originalVec)**2
        return score

    def minimize(self):
        print 'Start minimization...'
        self.optimLBFGS = sp.optimize.fmin_l_bfgs_b(self.objectiveSentimentRetrofit,
                                                    x0=self.initalVal(),
                                                    fprime=grad(self.objectiveSentimentRetrofit),
                                                    pgtol=1e-3, disp=True)
        print 'minimization done.'
        newVec = self.optimLBFGS[0][1]
        self.newVectors = {}
        indx2word = {self.word2indx[word]: word for word in self.word2indx}
        for indx in range(len(self.word2indx)):
            word = indx2word[indx]
            if word not in self.newVectors:
                self.newVectors[word] = newVec[indx]
        print self.optimLBFGS[1:]

    def writeWordVectors(self, outputFile):
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
    retrofitter.loadDocument('./aclImdb/train/pos/', 1)
    retrofitter.loadDocument('./aclImdb/train/neg/', -1)
    retrofitter.minimize()
    retrofitter.writeWordVectors('./output/sentimentVec.txt')
