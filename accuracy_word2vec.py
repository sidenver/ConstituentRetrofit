import sys
import getopt
import gensim
import logging

help_message = '''
$ python accuracy_word2vec.py -v <vectorsFile> -q <questions>[-h]
-v or --vectors to specify path to the word vectors input file (gzip or txt files are acceptable)
-q or --questions to evaluate accuracy)
-h or --help (this message is displayed)
'''


class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg


# Read command line arguments
def readCommandLineInput(argv):
    try:
        try:
            # specify the possible option switches
            opts, _ = getopt.getopt(argv[1:], "hv:q:", ["help", "vectors=", "questions="])
        except getopt.error, msg:
            raise Usage(msg)

        # default values
        vectorsFile = None
        questionsFile = None
        # option processing
        for option, value in opts:
            if option in ("-h", "--help"):
                raise Usage(help_message)
            elif option in ("-v", "--vectors"):
                vectorsFile = value
            elif option in ("-q", "--questions"):
                questionsFile = value
            else:
                raise Usage(help_message)

        if (vectorsFile is None) or (questionsFile is None):
            raise Usage(help_message)
        else:
            return (vectorsFile, questionsFile)

    except Usage, err:
        print str(err.msg)
        return 2


if __name__ == "__main__":
    # parse command line input
    commandParse = readCommandLineInput(sys.argv)
    # failed command line input
    if commandParse == 2:
        sys.exit(2)

    sys.stderr.write('reading file...\n')
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = gensim.models.word2vec.Word2Vec.load_word2vec_format(commandParse[0], binary=True)
    sys.stderr.write('start evaluation...\n')
    accuracy_word = model.accuracy(commandParse[1])
