import matplotlib.pyplot as plt
import math
import numpy as np


def plotAllSentenceLengths(sentenceLengths, averageSentenceLength):
    #print totalSentenceLength
    X = []
    Y = []
    counter = 0
    for elem in sentenceLengths.iteritems():
        X.append(elem[0])
        Y.append(math.log10(float(elem[1])))
        counter += 1
    plt.title('Occurrence of all sentence lengths in log space')
    plt.xlabel('Sentence length')
    plt.ylabel('log(sentenceLength)')
    plt.xticks(np.arange(0, max(X), 5), rotation=90)
    plt.plot(X, Y)
    plt.plot(X, Y, 'or')
    plt.axvline(x=averageSentenceLength, label='average sentence length:'+str(averageSentenceLength), color='green')
    plt.legend()
    # saving manually results in better quality
    #plt.savefig('plots/allSentenceLengthsOccurrence.svg', format='svg', dpi=1200)
    plt.show()

def plotCountCountsFromFile(nGramOccurrencesFile):
    print "here"
    nGramOccurrences = open(nGramOccurrencesFile, 'r')
    nGramFrequencies = {}
    # collect n-gram frequencies
    for line in nGramOccurrences:
        lineSplitted = line.strip().split(' ')
        count = int(lineSplitted[-1])

        if count not in nGramFrequencies:
            nGramFrequencies[count] = 1
        else:
            nGramFrequencies[count] += 1

        #print currentNGram
    nGramOccurrences.close()
    print "plotting"
    plotDict(nGramFrequencies)



def plotDict(data=dict()):
    X = []
    x_range = []
    Y = []
    counter = 0
    for elem in data.iteritems():
        X.append(elem[0])
        Y.append(elem[1])
        x_range.append(counter)
        counter += 1
    plt.title('Count-counts')
    plt.xlabel('# nGrams frequency')
    plt.ylabel('Count-counts in logspace')
    plt.xticks(np.arange(0, counter, 50), rotation=90)
    plt.semilogy(x_range, Y)
    plt.semilogy(x_range, Y, 'or')
    plt.show()
