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
import string
logging.basicConfig(level=logging.DEBUG)


class LanguageModel():

    def __init__(self, corpusFile, testCorpusFile, vocabularyFile=None):
        ######################## 1 ##########################
        self.corpusVocabulary = Vocabulary()
        if vocabularyFile:
            self.givenVocabulary  = Vocabulary(vocabularyFile)
        self.numRunningWords = 0      # 1a
        self.numSentences = 0         # 1b
        self.allSentenceLengths = {}  # 1c
        self.initLM(corpusFile)
        self.averageSentenceLength = self.numRunningWords / self.numSentences

        ######################## 2 ##########################
        # 2a
        self.sortedWordFrequencies = {}
        for word_idx in range(0, self.corpusVocabulary.size()):
            self.sortedWordFrequencies[self.corpusVocabulary.int2word[word_idx]] = self.corpusVocabulary.wordFrequencies[word_idx]
        self.sortedWordFrequencies = sorted(self.sortedWordFrequencies.items(), key=operator.itemgetter(1), reverse=True)
        self.writeListToFile(self.sortedWordFrequencies, 'sortedWordFrequencies')
        # 2b
        self.oov = 0.0
        self.nGramPrefixTreeRoot = PrefixTreeNode()
        self.discountingParameters = []

        #print "Counting 3-grams: "
        #self.allTrigramOccurrence = self.computeNGramOccurrence(corpusFile, 3, self.corpusVocabulary) # 2b
        #self.allBigramOccurrence = self.computeNGramOccurrence(corpusFile, 2, self.corpusVocabulary)  # 3
        print "Number of nodes in tree: ", self.nGramPrefixTreeRoot.subtreeSize()
        print "Discounting parameters: "


        self.nGramPrefixTreeRoot = PrefixTreeNode()
        self.discountingParameters = []
        self.wordFrequencies = None

        #print "Counting 3-grams: "
        self.allBigramOccurrence = self.computeNGramOccurrence(corpusFile, 2, self.corpusVocabulary) # 2b

        #print "Number of nodes in tree: ", self.nGramPrefixTreeRoot.subtreeSize()
        #print "Discounting parameters: "
        self.computeDiscountingParameters(2)
        print self.discountingParameters


        ######################## 3 ##########################
        print "Unknown probability: ", self.score(self.corpusVocabulary.unknownWordID(), [])
        print "Size of vocabulary:", self.corpusVocabulary.size()
        #print self.score(4, [0, 3])
        #print self.score(2, [3, 4])

        #print "Unknown probability: ", self.score(self.corpusVocabulary.unknownWordID(), [])
        #print "Size of vocabulary:", self.corpusVocabulary.size()
        unigramProbabilities, bigramProbabilities = 0.0, 0.0
        for word in range(0, self.corpusVocabulary.size()):
            unigramProbabilities += self.score(word, [])
            #print "p(", self.corpusVocabulary.symbol(word), ") =", self.score(word, [])
            bigramProbabilities  += self.score(word, [1])
            print "p(", self.corpusVocabulary.symbol(word), "|", self.corpusVocabulary.symbol(1), ") =", self.score(word, [1])
            bigramProbabilities  += self.score(word, [5])
            #print "p(", self.corpusVocabulary.symbol(word), "|", self.corpusVocabulary.symbol(5), ") =", self.score(word, [5])

        print "Sum of unigram probabilities:", unigramProbabilities
        print "Sum of bigram probabilities:", bigramProbabilities
        print "Perplexity: ", self.perplexity(testCorpusFile, self.corpusVocabulary)
        ######################## 4 ##########################

        print "Computing lower n-gram counts from 3-gram counts..."
        self.recomputeNGramOccurrence(3)

        print "Computing 2-gram counts..."
        self.nGramPrefixTreeRoot = PrefixTreeNode()
        self.computeNGramOccurrence(corpusFile, 2, self.corpusVocabulary)

        ######################## Outputs ##########################
        """
        logging.debug('# words: ' +  str(self.numRunningWords))
        logging.debug('# sentences: ' + str(self.numSentences))
        logging.debug('Occurrence of all sentence lengths: ' + str(self.allSentenceLengths))
        logging.debug('Average sentence length: ' + str(self.averageSentenceLength))
        logging.debug('Out of vocabulary rate: ' + str(self.oov))
        """
    def initLM(self, corpusFile):
        """
        Iterate through corpus file, create corpus vocabulary
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

    def computeNGramOccurrence(self, corpusFile, n, vocabulary=Vocabulary()):
        """
        Identify and compute n-grams from the given corpus file
        :param corpusFile: path to the corpus
        :param n: integer
        :return:
        """

        corpus = open(corpusFile, 'r')
        sentenceCounter = 0
        unknownWords = 0.0
        unknowWordId = vocabulary.index('<unk>')

        for line in corpus:
            currentStringWords = line.strip().split(' ')
            currentWordIDs =  [vocabulary.index("<s>")]
            for word_idx in range(0, len(currentStringWords)):
                currentWordID = vocabulary.index(currentStringWords[word_idx])
                currentWordIDs.append(currentWordID)
                # count amount of unknown words
                if currentWordID == unknowWordId:

                    unknownWords += 1
                    #print unknownWords
            currentWordIDs.append(vocabulary.index("</s>"))

            for i in range(n-1, len(currentWordIDs) + 1):
                currentNGram = currentWordIDs[i-n+1:i+1]
                #if len(currentNGram) != n:
                    #break
                self.nGramPrefixTreeRoot.recursiveAddNGram(currentNGram)
            sentenceCounter += 1

            if sentenceCounter % 10000 == 0:
                pass
                #print sentenceCounter

        self.oov = unknownWords / self.numRunningWords
        #print "Outputting n-gram frequencies"

        # breadth first search to get n-gram counts
        queue = Queue()
        wordIDQueue = Queue()

        queue.put(self.nGramPrefixTreeRoot)
        wordIDQueue.put([])

        output = open(str(n) + "-gram.counts", 'w')
        while not queue.empty():
            nextNode = queue.get()
            nextNGram = wordIDQueue.get()

            # Fill the queue
            for childWordID, childNode in nextNode.getChildren().items():
                childNGram = copy.copy(nextNGram)
                childNGram.append(childWordID)

                queue.put(childNode)
                wordIDQueue.put(childNGram)

            if len(nextNGram) == n:
                for wordID in nextNGram:
                    output.write(vocabulary.symbol(wordID) + ' ')
                output.write(str(nextNode.count) + '\n')

        output.close()

    def recomputeNGramOccurrence(self, maxNGramLength):
        """
        Compute counts for the lower n-grams in a prefix tree and output them to a file
        :param maxNGramLength: integer for the maximum n-gram length
        """

        self.discountingParameters = []

        # descend into the depth nGramLength of the prefix tree
        nextDepthQueue, currentDepthQueue = Queue(), Queue()
        wordIDQueue = Queue()
        wordIDQueue.put([])
        currentDepthQueue.put(self.nGramPrefixTreeRoot)
        for i in range(0, maxNGramLength - 1):

            # Open the output stream
            with open(str(i+1) + '-gram.recomputed.counts', 'w') as output:

                # BFS search
                while not currentDepthQueue.empty():
                    currentNode = currentDepthQueue.get()
                    currentNGram = wordIDQueue.get()

                    # Fill the queue with the children of all nodes of the current depth
                    for childWordID, childNode in currentNode.getChildren().items():
                        childNGram = copy.copy(currentNGram)
                        childNGram.append(childWordID)

                        wordIDQueue.put(childNGram)
                        nextDepthQueue.put(childNode)

                        # write the counts to the output stream
                        for wordID in childNGram:
                            output.write(self.corpusVocabulary.symbol(wordID) + ' ')
                        output.write(str(childNode.count) + '\n')

            # Swap the current depth queue (empty) with the next depth queue
            # for the next iteration
            nextDepthQueue, currentDepthQueue = currentDepthQueue, nextDepthQueue

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
                for childWordID, childNode in currentNode.getChildren().items():
                    nextDepthQueue.put(childNode)

                    # count
                    if childNode.count == 1:
                        numberOfSingletons += 1
                    elif childNode.count == 2:
                        numberOfDoubletons += 1

            # Calculate the discounting parameter
            #print (numberOfSingletons)
            discountingParameter = float(numberOfSingletons) / (numberOfSingletons + 2 * numberOfDoubletons)
            self.discountingParameters.append(discountingParameter)

            # Swap the current depth queue (empty) with the next depth queue
            # for the next iteration
            nextDepthQueue, currentDepthQueue = currentDepthQueue, nextDepthQueue

    def score(self, wordID, wordHistory):
        """
        Evaluate the language model of a word given a word history
        :param wordID: word identifier to score
        :param wordHistory: list of word identifiers to condition the scoring of wordID
        :return The language model probability of wordID given wordHistory
                and the new word history including wordID
        """

        # Recursion base case for unknown words and unigrams
        if wordID == self.corpusVocabulary.unknownWordID() or len(wordHistory) == 0:
            probability = self.discountingParameters[0]
            probability /= float(self.nGramPrefixTreeRoot.count) * float(self.corpusVocabulary.size())
            probability *= self.nGramPrefixTreeRoot.getNumberOfChildren()

            if wordID != self.corpusVocabulary.unknownWordID():
                wordNode = self.nGramPrefixTreeRoot.getNGramNode([wordID])
                if wordNode != None:
                    probability += max((wordNode.count - self.discountingParameters[0]) /
                                       float(self.nGramPrefixTreeRoot.count), 0.0)

            return probability

        # get the nodes from the prefix tree for the computation of the probabilities
        historyNode = self.nGramPrefixTreeRoot.getNGramNode(wordHistory)
        if historyNode == None:
            # N-gram does not need to be scored since the history is not valid
            # Use lower order probabilities
            return self.score(wordID, wordHistory[1:])
        # Get the probability for the current n-gram by recursively interpolating the (n-1)-gram prob.
        probability = self.discountingParameters[len(wordHistory)]
        probability *= historyNode.getNumberOfChildren() / float(historyNode.count)
        probability *= self.score(wordID, wordHistory[1:])

        # Add the probability for the n-gram, if available
        nGramNode = historyNode.getNGramNode([wordID])
        if nGramNode != None:
            discountedCount = nGramNode.count - self.discountingParameters[len(wordHistory)]
            probability += max(discountedCount / float(historyNode.count), 0.0)
        return probability
    def perplexity (self, testCorpusFile, vocabulary=Vocabulary()):
        LL = 0.0
        corpus = open(testCorpusFile, 'r')
        unknowWordId = vocabulary.index('<unk>')
        #print unknowWordId
        numRunningWords = 0;
        for line in corpus:
            currentStringWords = line.strip().split(' ')
            currentWordIDs = [vocabulary.index("<s>")]
            for word_idx in range(0, len(currentStringWords)):
                currentWordID = vocabulary.index(currentStringWords[word_idx])
                currentWordIDs.append(currentWordID)
                #if word_idx != 0:
                LL += np.log(self.score(currentWordIDs[word_idx + 1], [currentWordIDs[word_idx]]))
            currentWordIDs.append(vocabulary.index("</s>"))
            LL += np.log(self.score(currentWordIDs[len(currentWordIDs)-1], [currentWordIDs[len(currentWordIDs) - 2]]))
            numRunningWords += len(currentStringWords)-1
            #print len(currentStringWords)
        PP = np.exp(-LL/numRunningWords)
        return PP
vocabulary = '../../data/lm/vocabulary'
testCorpus = '../../data/lm/test'
corpusFile = '../../data/lm/corpus'
lm = LanguageModel(corpusFile, testCorpus)

#voc = Vocabulary(vocabulary)

######################## plots ##########################
#plotAllSentenceLengths(lm.allSentenceLengths, lm.averageSentenceLength)
#plotCountCountsFromFile('3-gram.counts')


