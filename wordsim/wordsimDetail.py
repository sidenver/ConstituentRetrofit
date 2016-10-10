"""Usage:
    wordsimDetail.py -v <vectorsFile> -d <dataset> [-n <number>]
    wordsimDetail.py -h | --help

take a word embedding file and evaluate it with word similarity
task using spearman rho, output the n most differntly ranked pair.

"""
import os
import logging
from docopt import docopt
import numpy
from collections import defaultdict
from scipy import linalg, stats
DATA_ROOT = os.path.dirname(os.path.abspath(__file__)) + "/data/en/"


class WordsimDetail:
    def __init__(self, dataset):
        logging.info("collecting datasets ..")
        self.dataset = defaultdict(list)
        for line in open(dataset):
            self.dataset[dataset.split('/')[-1].split('.')[0]].append([float(w) if i == 2 else w for i, w in enumerate(line.strip().split())])

    @staticmethod
    def cos(vec1, vec2):
        return vec1.dot(vec2)/(linalg.norm(vec1)*linalg.norm(vec2))

    @staticmethod
    def rho(vec1, vec2):
        return stats.stats.spearmanr(vec1, vec2)[0]

    @staticmethod
    def load_vector(path):
        try:
            logging.info("loading vector ..")
            if path[-3:] == ".gz":
                import gzip
                f = gzip.open(path, "rb")
            else:
                f = open(path, "rb")
        except ValueError:
            print "Oops!  No such file.  Try again .."
        word2vec = {}
        for wn, line in enumerate(f):
            line = line.lower().strip()
            word = line.split()[0]
            word2vec[word] = numpy.array(map(float, line.split()[1:]))
        logging.info("loaded vector {0} words found ..".format(len(word2vec.keys())))
        return word2vec

    @staticmethod
    def pprint(result, mostDifferent, n_max=None):
        from prettytable import PrettyTable
        x = PrettyTable(["Dataset", "Found", "Not Found", "Score (rho)"])
        x.align["Dataset"] = "l"
        for k in sorted(result):
            x.add_row([k, result[k][0], result[k][1], result[k][2]])
        print x

        if n_max:
            detail = PrettyTable(["Word1", "Word2", "Pred", "Label"])
            detail.align["Word1"] = "l"
            detail.align["Word2"] = "l"
            for dif in mostDifferent[:n_max]:
                detail.add_row([dif[2][0], dif[2][1], dif[0], dif[1]])
            print detail

    @staticmethod
    def listToRank(input):
        indices = list(range(len(input)))
        indices.sort(key=lambda x: input[x], reverse=True)
        output = [0] * len(indices)
        for i, x in enumerate(indices):
            output[x] = i
        return output

    def rankByDifference(self, wordPairs, pred, label):
        rankedPred = self.listToRank(pred)
        rankedLabel = self.listToRank(label)
        mostDifferent = sorted(zip(rankedPred, rankedLabel, wordPairs), key=lambda x: abs(x[0]-x[1]), reverse=True)
        return mostDifferent

    def evaluate(self, word_dict):
        result = {}
        vocab = word_dict.keys()
        for file_name, data in self.dataset.items():
            pred, label, found, notfound = [], [], 0, 0
            wordPairs = []
            for datum in data:
                if datum[0] in vocab and datum[1] in vocab:
                    found += 1
                    pred.append(self.cos(word_dict[datum[0]], word_dict[datum[1]]))
                    label.append(datum[2])
                    wordPairs.append((datum[0], datum[1]))
                else:
                    notfound += 1
            result[file_name] = (found, notfound, self.rho(label, pred)*100)
            mostDifferent = self.rankByDifference(wordPairs, pred, label)
        return result, mostDifferent

if __name__ == "__main__":
    commandParse = docopt(__doc__)
    wordsim = WordsimDetail(commandParse['<dataset>'])
    word2vec = wordsim.load_vector(commandParse['<vectorsFile>'])
    result, mostDifferent = wordsim.evaluate(word2vec)
    wordsim.pprint(result, mostDifferent, int(commandParse['<number>']))
