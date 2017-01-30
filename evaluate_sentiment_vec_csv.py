from sklearn import svm
from sklearn import linear_model
import numpy as np
import os
from nltk.tokenize import RegexpTokenizer
from collections import Counter
import sys

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib


class EvaluateSentimentVec(object):
    def __init__(self):
        self.lin_clf_reg = linear_model.LogisticRegression()
        self.lin_clf_svm = svm.LinearSVC()
        self.tokenizer = RegexpTokenizer(r"[\w'-]+")
        self.word2vec = {}
        self.dim = 0
        self.documentPos = []
        self.documentNeg = []

    def loadVector(self, pathVec, pathWord):
        print 'loading vector...'
        vec = np.loadtxt(open(pathVec, "rb"), delimiter=",")  # , skiprows=1, usecols=tuple(range(1, 301)))
        vocab = []
        with open(pathWord, 'r') as wordFile:
            for line in wordFile:
                if len(line.strip()) > 0:
                    vocab.append(line.strip())

        for idx, word in enumerate(vocab):
            self.word2vec[word] = vec[idx]
            if self.dim == 0:
                self.dim = len(vec[idx])

        print "loaded vector {0} words found ..".format(len(vocab))
        print "loaded w2v size {0} ..".format(vec.shape)

    def loadDocument(self, directory, polarity):
        print 'loading document at ' + directory
        for filename in os.listdir(directory):
            if filename.split('.')[-1] == "txt":
                # {word_index: freq}
                with open(directory + filename, 'r') as file:
                    vec = self.convertDocument2Vec(file.read())
                    if polarity == 'pos':
                        self.documentPos.append(vec)
                    else:
                        self.documentNeg.append(vec)

    def normalize(self, v):
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

    def convertDocument2Vec(self, line):
        tokenList = self.tokenizer.tokenize(line.lower())
        bow = Counter()
        for token in tokenList:
            if token in self.word2vec:
                bow[token] += 1.0

        vec = np.zeros(self.dim)
        for word in bow:
            vec += self.word2vec[word] * bow[word]
        vec = self.normalize(vec)

        return vec

    def generateSample(self):
        print 'generating samples...'
        self.totalCount = len(self.documentPos) + len(self.documentNeg)
        self.x = []
        self.y = []
        for pos, neg in zip(self.documentPos, self.documentNeg):
            self.y.append(1)
            self.x.append(pos)
            self.y.append(0)
            self.x.append(neg)

    def getTrainSample(self):
        return self.y[:self.totalCount*4/5], self.x[:self.totalCount*4/5]

    def getTestSample(self):
        return self.y[self.totalCount*4/5:], self.x[self.totalCount*4/5:]

    def train(self):
        print 'training...'
        y, x = self.getTrainSample()
        print len(y), 'training samples'
        self.lin_clf_reg.fit(x, y)
        self.lin_clf_svm.fit(x, y)

    def test(self):
        print 'testing...'
        y, x = self.getTestSample()
        print len(y), 'testing samples'
        return {'reg': self.lin_clf_reg.score(x, y), 'svm': self.lin_clf_svm.score(x, y)}

if __name__ == '__main__':
    results = []
    for filename in os.listdir(sys.argv[1]):
        if filename.split('.')[-1] == 'csv':
            evaluator = EvaluateSentimentVec()
            evaluator.loadVector(sys.argv[1] + filename, sys.argv[2])
            # evaluator.loadVector('./aclImdb/imdbRandom.vector')
            evaluator.loadDocument('./aclImdb/train/pos/', 'pos')
            evaluator.loadDocument('./aclImdb/train/neg/', 'neg')
            evaluator.generateSample()
            evaluator.train()
            result = evaluator.test()
            print result
            param = float('.'.join(filename.split('_')[-1].split('.')[:-1]))
            # result = np.random.normal(0, 1)

            results.append((param, result))

    results = sorted(results, key=lambda x: x[0])
    print results

    regR = [res[1]['reg'] for res in results]
    svmR = [res[1]['svm'] for res in results]
    x = [res[0] for res in results]

    df = pd.DataFrame({'svm': svmR, 'reg': regR}, index=x)
    matplotlib.style.use('ggplot')
    ax = df.plot()
    ax.set(xlabel='lambda',
           ylabel='accuracy',
           title="lamda's effect on performance")
    ax.set_xscale('log')
    # plt.show()
    plt.savefig('params.png')
