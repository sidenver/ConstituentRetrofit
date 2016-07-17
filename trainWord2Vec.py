import gensim
import logging
import re
import tarfile
import json
import multiprocessing

dataPath = "data/umbc_webbase_corpus.tar.gz"
phraseDir = "data/"


class MySentences(object):
    def __init__(self, dirname, phraseDir):
        self.dirname = dirname
        self.phrases = self.loadPhrase(phraseDir)

    def __iter__(self):
        count = 0
        with tarfile.open(self.dirname, 'r:gz') as tar:
            for member in tar:
                if member.isreg():
                    if member.name.split('.')[-1] == "possf2":
                        if count > 1:
                            break
                        print 'processing ' + member.name + '...'
                        count = count + 1
                        f = tar.extractfile(member)
                        for line in f.readlines():
                            if len(line) > 2:
                                yield self.word2phrase(line).split(' ')

    def repl(self, matchobj):
        if matchobj.group(0) in self.phrases:
            return re.sub(r' ', '|', matchobj.group(0))
        else:
            return matchobj.group(0)

    def word2phrase(self, content):
        return re.sub(r'([-\w]+_J\w+ [-\w]+_N\w+)', self.repl, content)

    def loadPhrase(self, phraseDir):
        phrase = []
        with open(phraseDir + "trainPhrase.txt") as json_file:
            phrase = phrase + json.load(json_file)
        with open(phraseDir + "testPhrase.txt") as json_file:
            phrase = phrase + json.load(json_file)
        return set(phrase)

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = MySentences(dataPath, phraseDir)  # a memory-friendly iterator
    model = gensim.models.Word2Vec(sentences, size=300, min_count=10, window=5, workers=multiprocessing.cpu_count())
    model.save('output/mymodel')
