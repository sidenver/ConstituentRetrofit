from collections import Counter
import os
from nltk.tokenize import RegexpTokenizer


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

if __name__ == '__main__':
    vocabBuilder = VocabBuilder()
    inputDirs = ['./aclImdb/train/pos/', './aclImdb/train/neg/']
    vocabBuilder.buildVocab(inputDirs)
    vocabBuilder.outputVocab('./aclImdb/imdb50.vocab')

    # inputDirs = ['./aclImdb/train/testRunPos/', './aclImdb/train/testRunNeg/']
    # vocabBuilder.buildVocab(inputDirs)
    # vocabBuilder.outputVocab('./aclImdb/imdbTest.vocab')
