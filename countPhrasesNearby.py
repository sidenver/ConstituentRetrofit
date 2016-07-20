import re
import os
import json

dataPath = "/fs/clip-scratch/shing/webbase_all/"
phraseDir = "/fs/clip-scratch/shing/"


def calNearbyCount(matchIndexes):
    if len(matchIndexes) <= 1:
        return (0, 0)
    pre = None
    nearbyCount3 = 0
    nearbyCount5 = 0
    for ths, nxt in zip(matchIndexes, matchIndexes[1:]):
        if nxt - ths < 3 or (pre is not None and ths - pre < 3):
            nearbyCount3 = nearbyCount3 + 1
        if nxt - ths < 5 or (pre is not None and ths - pre < 5):
            nearbyCount5 = nearbyCount5 + 1
        pre = ths
    return (nearbyCount3, nearbyCount5)


phrase = []
with open(phraseDir + "trainPhraseOri") as json_file:
    phrase = phrase + json.load(json_file)
with open(phraseDir + "testPhraseOri") as json_file:
    phrase = phrase + json.load(json_file)
phraseSet = set(phrase)


def repl(matchobj):
    if matchobj.group(0) in phraseSet:
        return re.sub(r' ', '|', matchobj.group(0))
    else:
        return matchobj.group(0)


totalNearbyCount3 = 0
totalNearbyCount5 = 0
totalPhraseCount = 0


for filename in os.listdir(dataPath):
    if filename.split('.')[-1] == "possf2":
        print 'processing ' + filename + '...'
        with open(dataPath + filename, 'r') as f:
            for line in f.readlines():
                if len(line) > 2:
                    newLine = re.sub(r'([-\w]+_J\w+ [-\w]+_N\w+)', repl, line)
                    tokens = newLine.split(' ')
                    matchIndexes = [idx for idx, token in enumerate(tokens)
                                    if '|' in token]
                    nearbyCount3, nearbyCount5 = calNearbyCount(matchIndexes)
                    totalNearbyCount3 = totalNearbyCount3 + nearbyCount3
                    totalNearbyCount5 = totalNearbyCount5 + nearbyCount5
                    totalPhraseCount = totalPhraseCount + len(matchIndexes)


print 'Total Phrase Count: ' + str(totalPhraseCount)
print 'Total Nearby Count 3: ' + str(totalNearbyCount3)
print 'Total Nearby Count 5: ' + str(totalNearbyCount5)
print 'Nearby Count 3 Ratio: ' + str(float(totalNearbyCount3)/float(totalPhraseCount))
print 'Nearby Count 5 Ratio: ' + str(float(totalNearbyCount5)/float(totalPhraseCount))
