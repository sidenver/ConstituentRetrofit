from collections import Counter
import os
from nltk.tokenize import RegexpTokenizer
import numpy as np
import re


class VocabBuilder(object):
    def __init__(self):
        self.tokenizer = RegexpTokenizer(r"[\w'-]+")
        self.wordCounter = Counter()

    def buildVocab(self, inputDirs):
        for directory in inputDirs:
            for idx, filename in enumerate(os.listdir(directory)):
                if idx > 50:
                    break
                if filename.split('.')[-1] == "txt":
                    # {word_index: freq}
                    with open(directory + filename, 'r') as file:
                        count = self.convertDocument2Bow(file.read())
                        self.wordCounter.update(count)

    def convertDocument2Bow(self, line):
        tokenList = self.tokenizer.tokenize(line.lower())
        bow = Counter()
        for token in tokenList:
            bow[token] += 1
        return bow

    def outputVocab(self, outFile):
        with open(outFile, 'w') as output:
            for token, count in self.wordCounter.most_common():
                output.write(token+'\n')

    def outputRandomWord2Vec(self, outFile, dim):
        print 'writing to file...'
        with open(outFile, 'w') as output:
            for token, count in self.wordCounter.most_common():
                output.write(token)
                npVec = self.makeRandVector(dim)
                vecStr = np.array2string(npVec, max_line_width='infty', precision=8)
                vecStr = vecStr.replace('[', ' ')
                vecStr = re.sub(r' +', ' ', vecStr)
                output.write(vecStr[:-1])
                output.write('\n')

    def makeRandVector(self, dims):
        mu, sigma = 0, 1
        vec = np.random.normal(mu, sigma, dims)
        return self.normalize(vec)

    def normalize(self, v):
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v/norm

if __name__ == '__main__':
    vocabBuilder = VocabBuilder()
    inputDirs = ['./aclImdb/train/pos/', './aclImdb/train/neg/']
    vocabBuilder.buildVocab(inputDirs)
    vocabBuilder.outputVocab('./aclImdb/imdb50.vocab')
    vocabBuilder.outputRandomWord2Vec('./aclImdb/imdbRandom.vector', 50)

    # inputDirs = ['./aclImdb/train/testRunPos/', './aclImdb/train/testRunNeg/']
    # vocabBuilder.buildVocab(inputDirs)
    # vocabBuilder.outputVocab('./aclImdb/imdbTest.vocab')
