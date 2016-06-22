import re

datapath = "./data/"
outputpath = "./output/"

vectorfile = open(datapath + 'doc2Dep20MWU57k_1000concat2000.tab', 'r')
wordfile = open(datapath + 'doc2Dep20MWU57k_1000concat2000.txt', 'r')

num_lines = sum([1 for line in wordfile])
vecNum = 1000
wordfile.seek(0)

outputfile = open(outputpath + 'dependencyVec.txt', 'w')

outputfile.write('%d %d\n' % (num_lines, vecNum))

for i in range(num_lines):
    outputfile.write(wordfile.readline().strip())
    dependencyVec = re.split(r'\t+', vectorfile.readline().rstrip('\t\n'))[1000:]
    for value in dependencyVec:
        outputfile.write(' %s' % (value))

    outputfile.write('\n')

outputfile.close()
