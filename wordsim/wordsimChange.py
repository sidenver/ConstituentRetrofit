"""Usage:
    wordsimChange.py -v <vectorsFile> -o <vectorOriginal> -d <dataset> [-n <number>]
    wordsimChange.py -h | --help

take two word embedding file and evaluate it with word similarity
output the n most differntly ranked pair.

"""
import logging
from docopt import docopt
import numpy
from collections import defaultdict
from scipy import linalg, stats


class WordsimChange:
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
            detail = PrettyTable(["Word1", "Word2", "PredCon", "PredOri", "Label", "Diff"])
            detail.align["Word1"] = "l"
            detail.align["Word2"] = "l"
            for dif in mostDifferent[:n_max]:
                detail.add_row([dif[0][0], dif[0][1], dif[1], dif[3], dif[4], abs(dif[1]-dif[2])-abs(dif[3]-dif[4])])
            print detail

    @staticmethod
    def listToRank(input):
        indices = list(range(len(input)))
        indices.sort(key=lambda x: input[x], reverse=True)
        output = [0] * len(indices)
        for i, x in enumerate(indices):
            output[x] = i
        return output

    def ranked(self, wordPairs, pred, label):
        rankedPred = self.listToRank(pred)
        rankedLabel = self.listToRank(label)
        return zip(rankedPred, rankedLabel, wordPairs)

    def rankByDifference(self, wordPairsRank1, wordPairsRank2):
        wordPairs = [wordPair[2] for wordPair in wordPairsRank1]
        pred1 = [wordPair[0] for wordPair in wordPairsRank1]
        label1 = [wordPair[1] for wordPair in wordPairsRank1]
        pred2 = [wordPair[0] for wordPair in wordPairsRank2]
        label2 = [wordPair[1] for wordPair in wordPairsRank2]
        return sorted(zip(wordPairs, pred1, label1, pred2, label2), key=lambda x: (abs(x[1]-x[2])-abs(x[3]-x[4])), reverse=True)

    def evaluate(self, word_dict, vect_name=None):
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
            if vect_name:
                file_name += "_"+vect_name
            result[file_name] = (found, notfound, self.rho(label, pred)*100)
            wordPairsRank = self.ranked(wordPairs, pred, label)
        return result, wordPairsRank

if __name__ == "__main__":
    commandParse = docopt(__doc__)
    wordsim = WordsimChange(commandParse['<dataset>'])
    word2vec1 = wordsim.load_vector(commandParse['<vectorsFile>'])
    word2vec2 = wordsim.load_vector(commandParse['<vectorOriginal>'])
    result1, wordPairsRank1 = wordsim.evaluate(word2vec1, "Con")
    result2, wordPairsRank2 = wordsim.evaluate(word2vec2, "Ori")
    result = result1.copy()
    result.update(result2)
    mostDifferent = wordsim.rankByDifference(wordPairsRank1, wordPairsRank2)
    wordsim.pprint(result, mostDifferent, int(commandParse['<number>']))
