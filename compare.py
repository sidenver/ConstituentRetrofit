from gensim.models import word2vec
import numpy as np


savePath = "/fs/clip-scratch/shing/output/"


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v/norm

if __name__ == '__main__':
    modelPhrase = word2vec.Word2Vec.load(savePath + 'mymodel')
    modelWord = word2vec.Word2Vec.load(savePath + 'wordOnly')

    phraseVocab = set([word for word in modelPhrase.vocab])
    wordVocab = set([word for word in modelWord.vocab])
    mutualVocab = phraseVocab & wordVocab

    print 'phrase num: ' + str(len(phraseVocab))
    print 'word num: ' + str(len(wordVocab))
    print 'mutual num: ' + str(len(mutualVocab))

    euclidian = {word: np.linalg.norm(modelPhrase[word]-modelWord[word]) for word in mutualVocab}
    normalizeWordVec = {word: normalize(modelWord[word] for word in mutualVocab)}
    normalizePhraseVec = {word: normalize(modelPhrase[word] for word in mutualVocab)}
    cosinceSim = {word: np.dot(normalizeWordVec[word], normalizePhraseVec[word]) for word in mutualVocab}

    print 'Euclidian:'
    print 'Average: ' + str(np.average(euclidian.values()))
    print 'STD: ' + str(np.std(euclidian.values()))
    print 'Max: ' + max(euclidian, key=euclidian.get)

    print 'Cosine:'
    print 'Average: ' + str(np.average(cosinceSim.values()))
    print 'STD: ' + str(np.std(cosinceSim.values()))
    print 'Max: ' + max(cosinceSim, key=cosinceSim.get)
