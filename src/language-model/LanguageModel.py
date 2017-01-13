import numpy as np
import re
import logging
logging.basicConfig(level=logging.DEBUG)


class LanguageModel():

    def __init__(self, corpusFile):
        self.numWords = self.countWords(corpusFile)
        self.numSentences = self.countSentences(corpusFile)
        self.allSentenceLenght, self.totalSentenceLength = self.computeSentenceLength(corpusFile)
        self.averageSentenceLength = self.totalSentenceLength/self.numSentences
        # Testing output
        logging.debug('# words: ' +  str(self.numWords))
        logging.debug('# sentences: ' + str(self.numSentences))
        logging.debug('Occurrence of all sentence length: ' + str(self.allSentenceLenght))
        logging.debug('Average sentence length: ' + str(self.averageSentenceLength))

    def countWords(self, corpusFile):
        """
        Counts all words including punctuation marks
        :param corpusFile: path to corpus file
        :return:
        """
        corpus = open(corpusFile, 'r')
        result = set()
        for line in corpus:
            currentWords = line.split(' ')
            for word in currentWords:
                result.add(word)
        corpus.close()
        return len(result)

    def countSentences(self, corpusFile):
        """
        Count all sentences separated by ., !, ?
        :param corpusFile: path to corpus file
        :return:
        """
        corpus = open(corpusFile, 'r')
        counter = 0
        for line in corpus:
            currentSentences = line.replace(' ! ', ' . ').replace(' ? ', ' . ').split(' . ')
            counter += len(currentSentences)
        corpus.close()
        return counter

    def computeSentenceLength(self, corpusFile):
        """
        each sentence length is represented by a key in the dictionary
        :param corpusFile: path to corpus file
        :return:
        """
        corpus = open(corpusFile, 'r')
        allSentenceLengths = {}
        totalSentenceLength = 0
        # all sentence length
        for line in corpus:
            currentSentences = line.replace(' ! ', ' . ').replace(' ? ', ' . ').split(' . ')
            for sentence in currentSentences:
                sentenceLenght = len(sentence.split(' '))
                if sentenceLenght not in allSentenceLengths:
                    allSentenceLengths[sentenceLenght] = 1
                else:
                    allSentenceLengths[sentenceLenght] += 1
        # average sentence length
        for elem in allSentenceLengths.iteritems():
            totalSentenceLength += elem[0] * elem[1]
        return allSentenceLengths,totalSentenceLength

corpusFile = '../../data/lm/corpus'
lm = LanguageModel(corpusFile)

