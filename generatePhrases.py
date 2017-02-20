import re
from collections import Counter
import json
import numpy as np
import os

dataPath = "/fs/clip-scratch/shing/webbase_all/"
phraseDir = "/fs/clip-scratch/shing/phrase/"

phraseJNNCount = Counter()
phraseJNCount = Counter()
phraseNNCount = Counter()

for filename in os.listdir(dataPath):
    if filename.split('.')[-1] == "possf2":
        print 'processing ' + filename + '...'
        with open(dataPath + filename, 'r') as f:
            content = f.read()
            match = re.findall(r"([-.,'@:\\/\w]+_J\w+ [-.,'@:\\/\w]+_N\w+)", content)
            phraseJNCount.update(match)
            match = re.findall(r"([-.,'@:\\/\w]+_N\w+ [-.,'@:\\/\w]+_N\w+)", content)
            phraseNNCount.update(match)
            match = re.findall(r"([-.,'@:\\/\w]+_J\w+ [-.,'@:\\/\w]+_N\w+ [-.,'@:\\/\w]+_N\w+)", content)
            phraseJNNCount.update(match)

phraseCount = Counter()
phraseCount.update(phraseJNCount)
phraseCount.update(phraseNNCount)
phraseCount.update(phraseJNNCount)

print phraseCount.most_common(20)

mostcommon = 1000000
trainNum = 400000
testNum = 100000

mostcommonPhrase = phraseCount.most_common(mostcommon)

trainPhrase = open(phraseDir + 'trainPhraseJNN', 'w')
trainList = [phrase[0] for phrase in mostcommonPhrase[:trainNum]]
trainPhrase.write(json.dumps(trainList))
trainPhrase.close()

testPhrase = open(phraseDir + 'testPhraseJNN', 'w')
testList = [phrase[0] for phrase in mostcommonPhrase[trainNum:]]
randomPhrase = np.random.choice(testList, testNum, replace=False)
testPhrase.write(json.dumps(randomPhrase.tolist()))
testPhrase.close()

print "phrase count is " + str(len(phraseCount)) + "."
