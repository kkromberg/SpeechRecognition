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
        self.allTrigramOccurrence = self.computeNGramOccurrence(corpusFile, 3) # 2b
        self.allTrigramFrequencies = self.countNGramsFrequencies(self.allTrigramOccurrence) # 2c

        self.initLM(corpusFile)
        # write files
        self.writeListToFile(self.sortedWordFrequencies, 'wordFrequencies')
        self.writeListToFile(self.allTrigramOccurrence, 'trigramOccurrence')
        self.writeDictToFile(self.allTrigramFrequencies, 'trigramFrequencies')

        self.averageSentenceLength = self.totalSentenceLength/self.numSentences

        # Testing output
        logging.debug('# words: ' +  str(self.numRunningWords))
        logging.debug('# sentences: ' + str(self.numSentences))
        logging.debug('Occurrence of all sentence length: ' + str(self.allSentenceLength))
        logging.debug('Average sentence length: ' + str(self.averageSentenceLength))
        #print json.dumps(self.wordFrequencies, indent=2)
        #logging.debug('Word frequencies: ' + str(self.wordFrequencies))

    def initLM(self, corpusFile):
        """
        :param corpusFile: path to the corpus
        :return:
        """

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

    def writeListToFile(self, data, outputFile):
        """
        Write data (list) into file (outputFile) separated by space
        :param data:
        :param outputFile:
        :return:
        """
        output = open(outputFile, 'w')
        for elem in data:
            output.write(str(elem[0]).replace('\n', '') + ' ' + str(elem[1]) + '\n')
        output.close()

    def writeDictToFile(self, data, outputFile):
        """
        Write data (dict) into file (outputFile) separated by space
        :param data:
        :param outputFile:
        :return:
        """

        output = open(outputFile, 'w')
        for elem in data.iteritems():
            output.write(str(elem[0]) + ' ' + str(elem[1]) + '\n')
        output.close()

    def computeNGramOccurrence(self, corpusFile, n):
        """
        Identify and compute n-grams from the given corpus file
        :param corpusFile: path to the corpus
        :param n: integer
        :return:
        """

        corpus = open(corpusFile, 'r')
        nGram = {}
        for line in corpus:
            currentSentences = line.replace(' ! ', ' . ').replace(' ? ', ' . ').split(' . ')
            for sentence in currentSentences:
                for punctuation in self.punctuations:
                    if punctuation in sentence:
                        sentence = sentence.replace(punctuation, '')
                currentWords = filter(None, sentence.split(' ')) # remove empty entries caused by deleting punctuations
                for i in range(0, len(currentWords)):
                    currentNGram = ''
                    if i < len(currentWords) -n:
                        # add <s> for begin of the sentence
                        if i == 0:
                            currentNGram = '<s>'
                        # add next n-1 elements
                        for j in range(i, i+3):
                            currentNGram += currentWords[j] + '|'

                        # add </s> for end of the sentence
                        if i == len(currentWords) - n - 1:
                            currentNGram += '</s>'
                    if currentNGram:
                        if currentNGram not in nGram:
                            nGram[currentNGram] = 1
                        else:
                            nGram[currentNGram] += 1

        return sorted(nGram.items(), key=operator.itemgetter(1), reverse=True)

    def countNGramsFrequencies(self, nGramOccurrence=dict()):

        nGramFrequencies = {}
        for elem in nGramOccurrence:
            if elem[1] not in nGramFrequencies:
                nGramFrequencies[elem[1]] = 1
            else:
                nGramFrequencies[elem[1]] += 1
        #print nGramFrequencies
        return nGramFrequencies



corpusFile = '../../data/lm/corpus'
lm = LanguageModel(corpusFile)

#plotRelativeSentenceLength(lm.allSentenceLength, lm.totalSentenceLength, lm.averageSentenceLength)
#plotDict(lm.allTrigramFrequencies)