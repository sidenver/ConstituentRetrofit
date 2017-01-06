# from sklearn import svm
from sklearn import linear_model
import numpy as np
import os
from nltk.tokenize import RegexpTokenizer
from collections import Counter
import sys


class EvaluateSentimentVec(object):
    def __init__(self):
        self.lin_clf = linear_model.LogisticRegression()
        # self.lin_clf = svm.LinearSVC()
        self.tokenizer = RegexpTokenizer(r"[\w'-]+")
        self.word2vec = {}
        self.dim = 0
        self.documentPos = []
        self.documentNeg = []

    def loadVector(self, path):
        print 'loading vector...'
        try:
            if path[-3:] == ".gz":
                import gzip
                f = gzip.open(path, "rb")
            else:
                f = open(path, "rb")
        except ValueError:
            print "Oops!  No such file.  Try again .."
        for wn, line in enumerate(f):
            line = line.lower().strip()
            word = line.split()[0]
            self.word2vec[word] = np.array(map(float, line.split()[1:]))
            if self.dim == 0:
                self.dim = len(self.word2vec[word])
        f.close()
        print "loaded vector {0} words found ..".format(len(self.word2vec.keys()))

    def loadDocument(self, directory, polarity):
        print 'loading document at ' + directory
        for idx, filename in enumerate(os.listdir(directory)):
            if idx > 3000:
                break
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
        return self.y[:self.totalCount / 2], self.x[:self.totalCount / 2]

    def getTestSample(self):
        return self.y[self.totalCount / 2:], self.x[self.totalCount / 2:]

    def train(self):
        print 'training...'
        y, x = self.getTrainSample()
        self.model = self.lin_clf.fit(x, y)

    def test(self):
        print 'testing...'
        y, x = self.getTestSample()
        return self.lin_clf.score(x, y)

    def showModel(self):
        print self.lin_clf.coef_

if __name__ == '__main__':
    evaluator = EvaluateSentimentVec()
    evaluator.loadVector(sys.argv[1])
    # evaluator.loadVector('./aclImdb/imdbRandom.vector')
    evaluator.loadDocument('./aclImdb/test/pos/', 'pos')
    evaluator.loadDocument('./aclImdb/test/neg/', 'neg')
    evaluator.generateSample()
    evaluator.train()
    evaluator.showModel()
    print evaluator.test()
