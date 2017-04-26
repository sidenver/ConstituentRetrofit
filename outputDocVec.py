"""Usage:
    outputDocVec.py -v <vectorsFile> -d <dataset> [options]
    outputDocVec.py (-h | --help)

Arguments:
-v <vectorsFile>             to specify VSM input file
-d <dataset>                 to specify dataset

Options:
-o <outputDir>               set output directory
-f <w2vFormat>               can be set to gensim, binary, or txt [default: gensim]
-h --help                    (this message is displayed)

Copyright (C) 2017 Han-Chin Shing <shing@cs.umd.edu>
Licenced under the Apache Licence, v2.0 - http://www.apache.org/licenses/LICENSE-2.0

"""

import numpy as np
import os
from nltk.tokenize import RegexpTokenizer
from collections import Counter
from gensim.models import word2vec
import sys
import pandas as pd
from scipy.sparse import lil_matrix
from scipy.io import savemat
import json
import progressbar
from docopt import docopt
from VSMReader import VSMReader

w2vDir = './output/sgWord/sgWordPhrase'


class OutputDocVec(object):
    def __init__(self, vectors, vocab, dim, vocabSize=None):
        self.vectors = vectors
        self.vocab = vocab
        self.dim = dim

        self.documentDictPos = Counter()
        self.documentDictNeg = Counter()
        self.documentDictTol = Counter()
        self.documentLabels = []
        self.documentBow = []
        self.wordSet = set(vocab.keys())
        if not vocabSize:
            self.vocabSize = len(self.wordSet)
        else:
            self.vocabSize = vocabSize
        self.word2indx = {}
        self.wordNum = 0
        self.tokenizer = RegexpTokenizer(r"[\w'-]+")

    def buildVocab(self):
        self.documentDictTol.update(self.documentDictPos)
        self.documentDictTol.update(self.documentDictNeg)
        self.wordList = [word for word, freq in self.documentDictTol.most_common(self.vocabSize) if freq > 1]
        self.wordListVocab = {}

        self.originalVec = np.zeros(len(self.wordList) * self.dim)
        for count, word in enumerate(self.wordList):
            vec = self.vectors[word]
            self.wordListVocab[word] = count
            start = count * self.dim
            np.put(self.originalVec, np.arange(start, start + self.dim), vec)
        self.originalVec = self.originalVec.reshape((len(self.wordList), self.dim))

        print 'original vec is of dimension:', self.originalVec.shape

        self.buildDocVec()

    def loadDocument(self, directory, polarity, trainCount, isDev=False, inputMode='dir'):
        print 'loading document at ' + directory
        idx = 0
        bar = progressbar.ProgressBar(widgets=[
            polarity, ' [', progressbar.Timer(), '] ',
            progressbar.Bar(),
            ' (', progressbar.ETA(), ') ',
        ])
        if inputMode == 'dir':
            for filename in bar(sorted(os.listdir(directory))):
                if not isDev:
                    if idx >= trainCount / 2:
                        break
                else:
                    if idx < trainCount / 2:
                        if filename.split('.')[-1] == "txt":
                            idx += 1
                        continue

                if filename.split('.')[-1] == "txt":
                    idx += 1
                    # {word: freq}
                    with open(directory + filename, 'r') as file:
                        line = file.read()
                        wordBow = self.convertDocument2Bow(line)
                        if polarity == 'pos':
                            self.documentLabels.append(1.0)
                            self.documentDictPos.update(wordBow)
                        else:
                            self.documentLabels.append(-1.0)
                            self.documentDictNeg.update(wordBow)
                        bow = Counter()
                        bow.update(wordBow)
                        self.documentBow.append(bow)
        else:
            filename = directory
            with open(filename, 'r') as file:
                for line in bar(file):
                    if not isDev:
                        if idx >= trainCount / 2:
                            break
                    else:
                        if idx < trainCount / 2:
                            if len(line.split()) > 0:
                                idx += 1
                            continue

                    if len(line.split()) > 0:
                        idx += 1
                        # {word: freq}
                        wordBow = self.convertDocument2Bow(line)
                        if polarity == 'pos':
                            self.documentLabels.append(1.0)
                            self.documentDictPos.update(wordBow)
                        else:
                            self.documentLabels.append(-1.0)
                            self.documentDictNeg.update(wordBow)
                        bow = Counter()
                        bow.update(wordBow)
                        self.documentBow.append(bow)

    def convertDocument2Bow(self, line):
        tokenList = self.tokenizer.tokenize(line.lower())
        wordBow = Counter()
        for token in tokenList:
            if token in self.wordSet:
                wordBow[token] += 1

        # phraseBow = Counter()
        # for token1, token2 in zip(tokenList[:-1], tokenList[1:]):
        #     token = token1 + '|' + token2
        #     if token in self.wordSet:
        #         # print token
        #         phraseBow[token] += 1

        # for token1, token2, token3 in zip(tokenList[:-2], tokenList[1:-1], tokenList[2:]):
        #     token = token1 + '|' + token2 + '|' + token3
        #     if token in self.wordSet:
        #         # print token
        #         phraseBow[token] += 1

        return wordBow

    def buildDocVec(self):
        sys.stderr.write('Building Document Vecter...\n')
        vocabLength = len(self.wordList)
        vocabSet = set(self.wordList)

        docMatrix = lil_matrix((len(self.documentLabels), vocabLength))
        for row, bow in enumerate(self.documentBow):
            for word in bow:
                if word in vocabSet:
                    index = self.wordListVocab[word]
                    docMatrix[row, index] = bow[word]

        sys.stderr.write('Finished building doc matrix.\n')
        self.docMatrix = docMatrix.tocsr()

    def normalize(self, v):
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

    def writeWordVectors(self, outputFileW2V, outputFileWordList):
        print 'writing to file...'
        df = pd.DataFrame(data=self.originalVec,
                          index=self.wordList)
        df.to_csv(outputFileW2V, encoding='utf-8')

        fp = open(outputFileWordList, 'w')
        fp.write(json.dumps(self.wordList))
        fp.close()

    def writeDocVec(self, outputDocVec, outputLabel, docType):
        print 'saving doc matrix to .mat format'
        savemat(outputDocVec, {'docMatrix'+docType: self.docMatrix})

        print 'saving doc label to .mat format'
        savemat(outputLabel, {'docLabel'+docType: np.asarray(self.documentLabels)})

    def clearForDevTest(self):
        self.documentDictPos = Counter()
        self.documentDictNeg = Counter()
        self.documentDictTol = Counter()
        self.documentLabels = []
        self.documentBow = []

if __name__ == '__main__':

    commandParse = docopt(__doc__)

    vsmReader = VSMReader(mode=commandParse['-f'])
    vocab, wordVectors, vectorDim = vsmReader.read(commandParse['-v'])
    outDir = commandParse['-o']
    dataset = commandParse['-d']

    if dataset == 'IMDB':
        trainCount = 20000
        JNN = 'All'

        retrofitter = OutputDocVec(vectors=wordVectors, vocab=vocab, dim=vectorDim)
        retrofitter.loadDocument('./aclImdb/train/pos/', 'pos', trainCount)
        retrofitter.loadDocument('./aclImdb/train/neg/', 'neg', trainCount)
        retrofitter.buildVocab()
        retrofitter.writeWordVectors('{}p2v{}.csv'.format(outDir, JNN),
                                     '{}wordList{}.json'.format(outDir, JNN))
        retrofitter.writeDocVec('{}docVec{}.mat'.format(outDir, JNN),
                                '{}docLabel{}.mat'.format(outDir, JNN),
                                docType='')
        retrofitter.clearForDevTest()
        retrofitter.loadDocument('./aclImdb/train/pos/', 'pos', trainCount, isDev=True)
        retrofitter.loadDocument('./aclImdb/train/neg/', 'neg', trainCount, isDev=True)
        retrofitter.buildDocVec()
        JNN = 'Dev'
        retrofitter.writeDocVec('{}docVec{}.mat'.format(outDir, JNN),
                                '{}docLabel{}.mat'.format(outDir, JNN),
                                docType=JNN)

        retrofitter.clearForDevTest()
        testCount = 25000
        retrofitter.loadDocument('./aclImdb/test/pos/', 'pos', testCount)
        retrofitter.loadDocument('./aclImdb/test/neg/', 'neg', testCount)
        retrofitter.buildDocVec()
        JNN = 'Test'
        retrofitter.writeDocVec('{}docVec{}.mat'.format(outDir, JNN),
                                '{}docLabel{}.mat'.format(outDir, JNN),
                                docType=JNN)
    elif dataset == 'RT':
        trainCount = 2000
        JNN = 'All'
        retrofitter = OutputDocVec(vectors=wordVectors, vocab=vocab, dim=vectorDim)
        retrofitter.loadDocument('./review_polarity/txt_sentoken/pos/', 'pos', trainCount)
        retrofitter.loadDocument('./review_polarity/txt_sentoken/neg/', 'neg', trainCount)
        retrofitter.buildVocab()
        retrofitter.writeWordVectors('{}p2v{}.csv'.format(outDir, JNN),
                                     '{}wordList{}.json'.format(outDir, JNN))
        retrofitter.writeDocVec('{}docVec{}.mat'.format(outDir, JNN),
                                '{}docLabel{}.mat'.format(outDir, JNN),
                                docType='')
    elif dataset == 'RT_s':
        trainCount = 5331*2
        JNN = 'All'

        retrofitter = OutputDocVec(vectors=wordVectors, vocab=vocab, dim=vectorDim)
        retrofitter.loadDocument('./rt-polaritydata/rt-polaritydata/rt-polarity.pos', 'pos', trainCount, inputMode='line')
        retrofitter.loadDocument('./rt-polaritydata/rt-polaritydata/rt-polarity.neg', 'neg', trainCount, inputMode='line')
        retrofitter.buildVocab()
        retrofitter.writeWordVectors('{}p2v{}.csv'.format(outDir, JNN),
                                     '{}wordList{}.json'.format(outDir, JNN))
        retrofitter.writeDocVec('{}docVec{}.mat'.format(outDir, JNN),
                                '{}docLabel{}.mat'.format(outDir, JNN),
                                docType='')
    elif dataset == 'Subj':
        trainCount = 5000*2
        JNN = 'All'

        retrofitter = OutputDocVec(vectors=wordVectors, vocab=vocab, dim=vectorDim)
        retrofitter.loadDocument('./rotten_imdb/quote.tok.gt9.5000', 'pos', trainCount, inputMode='line')
        retrofitter.loadDocument('./rotten_imdb/plot.tok.gt9.5000', 'neg', trainCount, inputMode='line')
        retrofitter.buildVocab()
        retrofitter.writeWordVectors('{}p2v{}.csv'.format(outDir, JNN),
                                     '{}wordList{}.json'.format(outDir, JNN))
        retrofitter.writeDocVec('{}docVec{}.mat'.format(outDir, JNN),
                                '{}docLabel{}.mat'.format(outDir, JNN),
                                docType='')
    else:
        print 'Please specify a valid dataset.'
    print 'Done'
