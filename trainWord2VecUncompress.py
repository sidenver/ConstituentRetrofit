import gensim
import logging
import re
import json
import multiprocessing
import os
import numpy as np

dataPath = "/fs/clip-scratch/shing/webbase_all/"
phraseDir = "/fs/clip-scratch/shing/"
savePath = "/fs/clip-scratch/shing/output/"


class MySentences(object):
    def __init__(self, dirname, phraseDir, concatenate=True, dirList=None):
        self.dirname = dirname
        self.concatenate = concatenate
        if self.concatenate:
            self.phrases = self.loadPhrase(phraseDir)
        if dirList is None:
            self.dirList = os.listdir(dataPath)
        else:
            self.dirList = dirList

    def __iter__(self):
        for filename in self.dirList:
            if filename.split('.')[-1] == "possf2":
                print 'processing ' + filename + '...'
                f = open(dataPath + filename, 'r')
                for line in f.readlines():
                    if len(line) > 2:
                        if self.concatenate:
                            yield self.word2phrase(line).split(' ')
                        else:
                            yield line.split(' ')

    def repl(self, matchobj):
        if matchobj.group(0) in self.phrases:
            return re.sub(r' ', '|', matchobj.group(0))
        else:
            return matchobj.group(0)

    def word2phrase(self, content):
        return re.sub(r"([-.,'@:\\/\w]+_J\w+ [-.,'@:\\/\w]+_N\w+)", self.repl, content)

    def loadPhrase(self, phraseDir):
        phrase = []
        with open(phraseDir + "trainPhrase") as json_file:
            phrase = phrase + json.load(json_file)
        with open(phraseDir + "testPhrase") as json_file:
            phrase = phrase + json.load(json_file)
        return set(phrase)

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    myDirList = os.listdir(dataPath)
    # np.random.shuffle(myDirList)
    sentences = MySentences(dataPath, phraseDir, concatenate=False, dirList=myDirList)  # a memory-friendly iterator
    model = gensim.models.Word2Vec(sentences, size=300, min_count=5, window=5, workers=multiprocessing.cpu_count())
    model.save(savePath + 'wordOnly')
