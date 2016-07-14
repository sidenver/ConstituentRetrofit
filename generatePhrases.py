import re
from collections import Counter
import tarfile
import json
import numpy as np

dataPath = "data/umbc_webbase_corpus.tar.gz"

tar = tarfile.open(dataPath, 'r:gz')

phraseCount = Counter()

for member in tar:
    if member.isreg():
        if member.name.split('.')[-1] == "possf2":
            print 'processing ' + member.name + '...'
            f = tar.extractfile(member)
            content = f.read()
            match = re.findall(r'([-\w]+_J\w+ [-\w]+_N\w+)', content)
            phraseCount.update(match)

tar.close()

print phraseCount.most_common(20)

trainNum = 1000000
testNum = 100000

trainPhrase = open('data/trainPhrase.txt', 'w')
trainList = [phrase[0] for phrase in phraseCount.most_common(trainNum)]
trainPhrase.write(json.dumps(trainList))
trainPhrase.close()

testPhrase = open('data/testPhrase.txt', 'w')
testList = [phrase[0] for phrase in phraseCount.most_common()[trainNum:]]
randomPhrase = np.random.choice(testList, testNum, replace=False)
testPhrase.write(json.dumps(randomPhrase.tolist()))
testPhrase.close()

print "phrase count is " + str(len(phraseCount)) + "."

allPhrase = open('data/allPhrase.txt', 'w')
allPhrase.write(json.dumps(phraseCount))
allPhrase.close()
