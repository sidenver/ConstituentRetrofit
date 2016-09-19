from gensim.models import word2vec
import numpy as np
from scipy import linalg, stats
import sys

savePath = "/fs/clip-scratch/shing/output/"


def cos(vec1, vec2):
        return vec1.dot(vec2)/(linalg.norm(vec1)*linalg.norm(vec2))


def rho(vec1, vec2):
        return stats.stats.spearmanr(vec1, vec2)


if __name__ == '__main__':
    modelWord = word2vec.Word2Vec.load(savePath + sys.argv[1])
    modelPhrase = word2vec.Word2Vec.load(savePath + sys.argv[2])

    phraseVocab = set([word for word in modelPhrase.vocab])
    wordVocab = set([word for word in modelWord.vocab])
    mutualVocab = phraseVocab & wordVocab

    print sys.argv[1] + ' num: ' + str(len(wordVocab))
    print sys.argv[2] + ' num: ' + str(len(phraseVocab))
    print 'mutual num: ' + str(len(mutualVocab))

    pairsToCompare = [np.random.choice(mutualVocab, 2, replace=False) for i in range(3000)]
    scoreWord = [cos(modelWord[pair[0]], modelWord[pair[1]]) for pair in pairsToCompare]
    scorePhrase = [cos(modelPhrase[pair[0]], modelPhrase[pair[1]]) for pair in pairsToCompare]

    rhoScore = rho(scoreWord, scorePhrase)
    print 'rho is: ', rhoScore[0]
    print 'p value is: ', rhoScore[1]
