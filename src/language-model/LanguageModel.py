import numpy as np
import re
import logging
from Util import *
import json
import operator
logging.basicConfig(level=logging.DEBUG)


class LanguageModel():

    def __init__(self, corpusFile):

        self.punctuations = ['.', ',', ':', ';', '!', '-', '?']
        self.numRunningWords = None
        self.wordFrequencies = None
        self.numSentences = 0
        self.allSentenceLength = {}
        self.totalSentenceLength = 0
        self.sortedWordFrequencies = 0

        self.initLM(corpusFile)
        self.writeDictToFile(self.sortedWordFrequencies, 'wordFrequencies')

        self.averageSentenceLength = self.totalSentenceLength/self.numSentences

        # Testing output
        logging.debug('# words: ' +  str(self.numRunningWords))
        logging.debug('# sentences: ' + str(self.numSentences))
        logging.debug('Occurrence of all sentence length: ' + str(self.allSentenceLength))
        logging.debug('Average sentence length: ' + str(self.averageSentenceLength))
        #print json.dumps(self.wordFrequencies, indent=2)
        #logging.debug('Word frequencies: ' + str(self.wordFrequencies))


    def initLM(self, corpusFile):

        corpus = open(corpusFile, 'r')
        runningWords = set()
        wordFrequencies = {}
        for line in corpus:
            currentWords = line.split(' ')
            for word in currentWords:
                runningWords.add(word)
                # 2a
                if word.replace('\n', '') not in self.punctuations:
                    if word not in wordFrequencies:
                        wordFrequencies[word] = 1
                    else:
                        wordFrequencies[word] += 1

            # 1b
            currentSentences = line.replace(' ! ', ' . ').replace(' ? ', ' . ').split(' . ')
            self.numSentences += len(currentSentences)
            # 1c
            for sentence in currentSentences:
                sentenceLength = len(sentence.split(' '))
                if sentenceLength not in self.allSentenceLength:
                    self.allSentenceLength[sentenceLength] = 1
                else:
                    self.allSentenceLength[sentenceLength] += 1
        for elem in self.allSentenceLength.iteritems():
            self.totalSentenceLength += elem[0] * elem[1]
        corpus.close()

        # 1a
        self.numRunningWords = len(runningWords)
        # 2a (sort)
        self.sortedWordFrequencies = sorted(wordFrequencies.items(), key=operator.itemgetter(1), reverse=True)

    def writeDictToFile(self, data, outputFile):
        output = open(outputFile, 'w')
        for elem in data:
            output.write(elem[0].replace('\n', '') + ' ' + str(elem[1]) + '\n')
        output.close()

corpusFile = '../../data/lm/corpus'
lm = LanguageModel(corpusFile)

plotRelativeSentenceLength(lm.allSentenceLength, lm.totalSentenceLength, lm.averageSentenceLength)