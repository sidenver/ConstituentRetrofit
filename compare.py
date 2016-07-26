from gensim.models import word2vec
import numpy as np
import sys

savePath = "/fs/clip-scratch/shing/output/"


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v/norm

if __name__ == '__main__':
    modelWord = word2vec.Word2Vec.load(savePath + sys.argv[1])
    modelPhrase = word2vec.Word2Vec.load(savePath + sys.argv[2])

    phraseVocab = set([word for word in modelPhrase.vocab])
    wordVocab = set([word for word in modelWord.vocab])
    mutualVocab = phraseVocab & wordVocab

    print 'phrase num: ' + str(len(phraseVocab))
    print 'word num: ' + str(len(wordVocab))
    print 'mutual num: ' + str(len(mutualVocab))

    euclidian = {word: np.linalg.norm(modelPhrase[word]-modelWord[word]) for word in mutualVocab}
    normalizeWordVec = {word: normalize(modelWord[word]) for word in mutualVocab}
    normalizePhraseVec = {word: normalize(modelPhrase[word]) for word in mutualVocab}
    cosinceSim = {word: np.dot(normalizeWordVec[word], normalizePhraseVec[word]) for word in mutualVocab}

    print '\nEuclidian:'
    print 'Average: ' + str(np.average(euclidian.values()))
    print 'STD: ' + str(np.std(euclidian.values()))
    print 'Max: ' + str(np.amax(euclidian.values()))
    print 'Argmax: ' + max(euclidian, key=euclidian.get)

    print '\nCosine:'
    print 'Average: ' + str(np.average(cosinceSim.values()))
    print 'STD: ' + str(np.std(cosinceSim.values()))
    print 'Max: ' + str(np.amax(cosinceSim.values()))
    print 'Argmax: ' + max(cosinceSim, key=cosinceSim.get)

    euclidian2 = [np.linalg.norm(modelWord[word1]-modelWord[word2]) for word1 in mutualVocab for word2 in mutualVocab]
    cosinceSim2 = [np.dot(normalizeWordVec[word1], normalizeWordVec[word2]) for word1 in mutualVocab for word2 in mutualVocab]

    print '\n\nWithin Words:'
    print 'Euclidian:'
    print 'Average: ' + str(np.average(euclidian2))
    print 'STD: ' + str(np.std(euclidian2))
    print 'Max: ' + str(np.amax(euclidian2))

    print 'Cosine:'
    print 'Average: ' + str(np.average(cosinceSim2))
    print 'STD: ' + str(np.std(cosinceSim2))
    print 'Max: ' + str(np.amax(cosinceSim2))

    euclidian2 = [np.linalg.norm(modelPhrase[word1]-modelPhrase[word2]) for word1 in mutualVocab for word2 in mutualVocab]
    cosinceSim2 = [np.dot(normalizePhraseVec[word1], normalizePhraseVec[word2]) for word1 in mutualVocab for word2 in mutualVocab]

    print '\nWithin Phrases:'
    print 'Euclidian:'
    print 'Average: ' + str(np.average(euclidian2))
    print 'STD: ' + str(np.std(euclidian2))
    print 'Max: ' + str(np.amax(euclidian2))

    print 'Cosine:'
    print 'Average: ' + str(np.average(cosinceSim2))
    print 'STD: ' + str(np.std(cosinceSim2))
    print 'Max: ' + str(np.amax(cosinceSim2))
