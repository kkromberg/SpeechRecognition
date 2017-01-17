import numpy as np
import re
import logging
from Util import *
from Vocabulary import Vocabulary
import json
import operator
logging.basicConfig(level=logging.DEBUG)


class LanguageModel():

    def __init__(self, corpusFile):

        self.punctuations = ['.', ',', ':', ';', '!', '-', '?']
        self.corpusVocabulary = Vocabulary()


        self.numRunningWords = None
        self.wordFrequencies = None
        self.numSentences = 0
        self.allSentenceLength = {}
        self.sortedWordFrequencies = 0
        self.allTrigramOccurrence = self.computeNGramOccurrence(corpusFile, 3) # 2b
        self.allBigramOccurrence = self.computeNGramOccurrence(corpusFile, 2)  # 3
        self.allUnigramOccurrence = self.computeNGramOccurrence(corpusFile, 1)  # 3
        self.allTrigramFrequencies = self.countNGramsFrequencies(self.allTrigramOccurrence) # 2c
        self.allBigramFrequencies = self.countNGramsFrequencies(self.allBigramOccurrence)  # 3
        self.allUnigramFrequencies = self.countNGramsFrequencies(self.allUnigramOccurrence)  # 3

        self.recomputedBigramOccurrence = self.recomputeNGramOccurrence(self.allTrigramOccurrence) # 3
        self.recomputedUnigramOccurrence = self.recomputeNGramOccurrence(self.allBigramOccurrence) # 3

        self.initLM(corpusFile)

        self.averageSentenceLength = self.numRunningWords/self.numSentences

        '''
        # write files
        self.writeListToFile(self.sortedWordFrequencies, 'wordFrequencies')
        self.writeListToFile(self.allTrigramOccurrence, 'trigramOccurrence')
        self.writeDictToFile(self.allTrigramFrequencies, 'trigramFrequencies')

        self.writeListToFile(self.allBigramOccurrence, 'bigramOccurrence') # 3
        self.writeDictToFile(self.allBigramFrequencies, 'bigramFrequencies') # 3

        self.writeListToFile(self.allUnigramOccurrence, 'unigramOccurrence') # 3
        self.writeDictToFile(self.allUnigramFrequencies, 'unigramFrequencies') # 3

        self.writeListToFile(self.recomputedBigramOccurrence, 'recomputedBigramOccurrence') # 3
        self.writeListToFile(self.recomputedUnigramOccurrence, 'recomputedUnigramOccurrence') # 3


        # Testing output
        logging.debug('# words: ' +  str(self.numRunningWords))
        logging.debug('# sentences: ' + str(self.numSentences))
        logging.debug('Occurrence of all sentence length: ' + str(self.allSentenceLength))
        logging.debug('Average sentence length: ' + str(self.averageSentenceLength))
        #print json.dumps(self.wordFrequencies, indent=2)
        #logging.debug('Word frequencies: ' + str(self.wordFrequencies))
        '''
    def initLM(self, corpusFile):
        """
        :param corpusFile: path to the corpus
        :return:
        """
        corpus = open(corpusFile, 'r')
        for line in corpus:
            currentWords = line.strip().split(' ')
            for word in currentWords:
                self.corpusVocabulary.addSymbol(word)
            self.numSentences += 1
            # 1c
            sentenceLength = len(currentWords)
            if sentenceLength not in self.allSentenceLength:
                self.allSentenceLength[sentenceLength] = 1
            else:
                self.allSentenceLength[sentenceLength] += 1

        # compute total sentence length
        for elem in self.allSentenceLength.iteritems():
            self.numRunningWords += elem[0] * elem[1]

        corpus.close()
        # 2a (sort)
        #self.sortedWordFrequencies = sorted(wordFrequencies.items(), key=operator.itemgetter(1), reverse=True)

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
            currentStringWords = line.strip().split(' ')
            currentWords = [self.corpusVocabulary.addSymbol(x) for x in currentStringWords]

            #currentNGram = [self.corpusVocabulary.index("<s>") for i in range(0, n)]
            for i in range(0, len(currentWords)):
                currentNGram = ''
                if i < len(currentWords) -n:
                    # add <s> for begin of the sentence
                    if i == 0:
                        currentNGram = '<s>'
                    # add next n-1 elements
                    for j in range(i, i+n):
                        pass
                        # TODO
                        #currentNGram += currentWords[j] + '|'

                    # add </s> for end of the sentence
                    if i == len(currentWords) - n - 1:
                        currentNGram += '</s>'
                #print currentNGram
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

    def recomputeNGramOccurrence(self, nGramOccurrence=dict()):
        nGram = {}
        sep = '|'
        for elem in nGramOccurrence:
            currentNGram = elem[0].split(sep, 1)[1]
            #print currentNGram
            if currentNGram not in nGram:
                nGram[currentNGram] = 1
            else:
                nGram[currentNGram] += 1
        return sorted(nGram.items(), key=operator.itemgetter(1), reverse=True)


corpusFile = '../../data/lm/corpus'
vocabulary = '../../data/lm/vocabulary'
lm = LanguageModel(corpusFile)
#voc = Vocabulary(vocabulary)


#plotRelativeSentenceLength(lm.allSentenceLength, lm.totalSentenceLength, lm.averageSentenceLength)
#plotDict(lm.allTrigramFrequencies)