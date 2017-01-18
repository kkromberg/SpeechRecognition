import numpy as np
import copy
import re
import logging
from Util import *
from Vocabulary import Vocabulary
from PrefixTree import PrefixTreeNode
import json
import operator
from Queue import Queue
logging.basicConfig(level=logging.DEBUG)


class LanguageModel():

    def __init__(self, corpusFile):
        ######################## 1 ##########################
        self.corpusVocabulary = Vocabulary()
        self.numRunningWords = 0      # 1a
        self.numSentences = 0         # 1b
        self.allSentenceLengths = {}  # 1c
        self.initLM(corpusFile)
        self.averageSentenceLength = self.numRunningWords / self.numSentences


        logging.debug('# words: ' +  str(self.numRunningWords))
        logging.debug('# sentences: ' + str(self.numSentences))
        logging.debug('Occurrence of all sentence lengths: ' + str(self.allSentenceLengths))
        logging.debug('Average sentence length: ' + str(self.averageSentenceLength))



        self.nGramPrefixTreeRoot = PrefixTreeNode()
        self.discountingParameters = []


        self.wordFrequencies = None


        self.sortedWordFrequencies = 0
        '''
        print "Counting 3-grams: "
        self.allTrigramOccurrence = self.computeNGramOccurrence(corpusFile, 3) # 2b

        print "Discounting parameters: "
        self.computeDiscountingParameters(3)
        print self.discountingParameters

        print self.score(4, [0, 3])
        print self.score(2, [3, 4])

        #self.allBigramOccurrence = self.computeNGramOccurrence(corpusFile, 2)  # 3
        #self.allUnigramOccurrence = self.computeNGramOccurrence(corpusFile, 1)  # 3\


        self.allTrigramFrequencies = self.countNGramsFrequencies(self.allTrigramOccurrence) # 2c
        self.allBigramFrequencies = self.countNGramsFrequencies(self.allBigramOccurrence)  # 3
        self.allUnigramFrequencies = self.countNGramsFrequencies(self.allUnigramOccurrence)  # 3

        self.recomputedBigramOccurrence = self.recomputeNGramOccurrence(self.allTrigramOccurrence) # 3
        self.recomputedUnigramOccurrence = self.recomputeNGramOccurrence(self.allBigramOccurrence) # 3




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
            if sentenceLength not in self.allSentenceLengths:
                self.allSentenceLengths[sentenceLength] = 1
            else:
                self.allSentenceLengths[sentenceLength] += 1

        # compute total sentence length
        for elem in self.allSentenceLengths.iteritems():
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
        sentenceCounter = 0
        for line in corpus:
            currentStringWords = line.strip().split(' ')
            currentWordIDs =  [self.corpusVocabulary.index("<s>")]
            currentWordIDs += [self.corpusVocabulary.addSymbol(x) for x in currentStringWords]
            currentWordIDs.append(self.corpusVocabulary.index("</s>"))

            for i in range(n-1, len(currentWordIDs) + 1):
                currentNGram = currentWordIDs[i-n+1:i+1]
                self.nGramPrefixTreeRoot.recursiveAddNGram(currentNGram)

            sentenceCounter += 1

            if sentenceCounter % 10000 == 0:
                print sentenceCounter

        print "Outputting n-gram frequencies"

        # breadth first search to get n-gram counts
        queue = Queue()
        wordIDQueue = Queue()
        queue.put(self.nGramPrefixTreeRoot)

        emptyList = []
        wordIDQueue.put(emptyList)

        output = open(str(n) + "-gram.counts", 'w')
        while not queue.empty():
            nextNode = queue.get()
            nextNGram = wordIDQueue.get()

            # Fill the queue
            for childWordID, childNode in nextNode.children.items():
                childNGram = copy.copy(nextNGram)
                childNGram.append(childWordID)

                queue.put(childNode)
                wordIDQueue.put(childNGram)

            if len(nextNGram) == n:
                for wordID in nextNGram:
                    output.write(self.corpusVocabulary.symbol(wordID) + ' ')
                output.write(str(nextNode.count) + '\n')

        output.close()

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

    def computeDiscountingParameters(self, nGramLength):
        """
        Compute discounting parameters up to a specified n-gram length
        :param nGramLength: integer for the n in n-gram
        """
        self.discountingParameters = []

        # descend into the depth nGramLength of the prefix tree
        nextDepthQueue, currentDepthQueue = Queue(), Queue()
        currentDepthQueue.put(self.nGramPrefixTreeRoot)
        for i in range(0, nGramLength):
            currentNode = None

            # Count the number of singletons (n-grams that appeared only once)
            # and doubletons (n-grams that appeared only twice) while parsing the children
            numberOfSingletons, numberOfDoubletons = 0, 0

            # Fill the queue with the children of all nodes of the current depth
            while not currentDepthQueue.empty():
                currentNode = currentDepthQueue.get()
                for childWordID, childNode in currentNode.children.items():
                    nextDepthQueue.put(childNode)

                    # count
                    if childNode.count == 1:
                        numberOfSingletons += 1
                    elif childNode.count == 2:
                        numberOfDoubletons += 1

            # Calculate the discounting parameter
            discountingParameter = float(numberOfSingletons) / (numberOfSingletons + 2 * numberOfDoubletons)
            self.discountingParameters.append(discountingParameter)

            # Swap the current depth queue (empty) with the next depth queue
            # for the next iteration
            nextDepthQueue, currentDepthQueue = currentDepthQueue, nextDepthQueue

    def score(self, wordID, wordHistory):
        """
        Evaluate the language model of a word given a word history
        :param wordID: word identifier to score
        :return The language model probability of wordID given wordHistory
                and the new word history including wordID
        """
        # Recursion base case for unknown words and unigrams
        if wordID == self.corpusVocabulary.unknownWordID() or len(wordHistory) == 0:
            probability = self.discountingParameters[0] \
                          * (float(self.nGramPrefixTreeRoot.numberOfFollowingContexts) / float(
                self.nGramPrefixTreeRoot.count))

            if wordID != self.corpusVocabulary.unknownWordID():
                wordNode = self.nGramPrefixTreeRoot.getNGramNode([wordID])
                probability += max(
                    (wordNode.count - self.discountingParameters[0]) / float(self.nGramPrefixTreeRoot.count), 0)

            return probability

        # get the nodes from the prefix tree for the computation of the probabilities
        historyNode = self.nGramPrefixTreeRoot.getNGramNode(wordHistory)
        if historyNode == None:
            # N-gram does not need to be scored since the history is not valid
            # Use lower order probabilities
            return self.score(wordID, wordHistory[1:])

        # Get the probability for the current n-gram by recursively interpolating the (n-1)-gram prob.
        currentDiscountParameter = self.discountingParameters[len(wordHistory)]
        probability = currentDiscountParameter \
                      * (float(historyNode.numberOfFollowingContexts) / float(historyNode.count)) \
                      * self.score(wordID, wordHistory[1:])

        # Add the probability for the n-gram, if available
        nGramNode = historyNode.getNGramNode([wordID])
        if nGramNode != None:
            probability += max((nGramNode.count - currentDiscountParameter) / float(historyNode.count), 0.0)

        return probability

corpusFile = '../../data/lm/corpus'
vocabulary = '../../data/lm/vocabulary'
lm = LanguageModel(corpusFile)
#voc = Vocabulary(vocabulary)

######################## plots ##########################
plotAllSentenceLengths(lm.allSentenceLengths, lm.averageSentenceLength)
#plotDict(lm.allTrigramFrequencies)