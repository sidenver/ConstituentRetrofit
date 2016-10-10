import re

f = open('data/delorme.com_shu.pages_0.possf2', 'r')
st = f.readline()

testList = set(['new_JJ breed_NN', 'fantastic_JJ device_NN'])

print st


def repl(matchobj):
    if matchobj.group(0) in testList:
        return re.sub(r' ', '|', matchobj.group(0))
    else:
        return matchobj.group(0)

newst = re.sub(r'([-\w]+_J\w+ [-\w]+_N\w+)', repl, st)

print newst

print st
