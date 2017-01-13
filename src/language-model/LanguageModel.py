import numpy as np


class LanguageModel():

    def __init__(self, corpusFile):
        corpus = open(corpusFile, 'r')
        self.numWord = self.countWords(corpus)
        self.numSentences = self.countSentences(corpus)


    def countWords(self, corpus):
        result = set()
        print len(corpus)
        for line in corpus:
            currentWords = line.split(" ")
            for word in currentWords:
                result.add(word)
            return len(result)

    def countSentences(self, corpus):
        # TODO
        counter = 0
        result = set()
        for line in corpus:
            print line
            #currentSentences = line.split(' . ')
            #print currentSentences
            #print len(currentSentences)
            break



corpusFile = '../../data/lm/corpus'
lm = LanguageModel(corpusFile)

