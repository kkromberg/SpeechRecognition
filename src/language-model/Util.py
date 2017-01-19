import matplotlib.pyplot as plt
import math
import numpy as np
import operator


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
    #plt.plot(X, Y, 'or')
    plt.axvline(x=averageSentenceLength, label='average sentence length:'+str(averageSentenceLength), color='green')
    plt.legend()
    # saving manually results in better quality
    #plt.savefig('plots/allSentenceLengthsOccurrence.svg', format='svg', dpi=1200)
    plt.show()

def plotCountCountsFromFile(nGramOccurrencesFile):
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
    nGramOccurrences.close()

    # write count counts to file
    fileName = nGramOccurrencesFile.split('.')
    outputFileName = fileName[0] + '.count' + fileName[1]
    output = open(outputFileName, 'w')
    for elem in nGramFrequencies.iteritems():
        output.write(str(elem[0]) + ' ' +  str(elem[1]) + '\n')
    output.close()
    plotDict(nGramFrequencies)



def plotDict(data=dict()):
    X = []
    x_range = []
    Y = []
    counter = 0
    for elem in sorted(data.iteritems(), key=operator.itemgetter(0)):
        X.append(int(elem[0]))
        Y.append(elem[1])
        x_range.append(counter)
        counter += 1

    plt.title('Count-counts')
    plt.xlabel('nGram frequency')
    plt.ylabel('Count-counts in logspace')
    plt.xticks(np.arange(0, max(X), 1000), rotation=90)
    #plt.semilogy(X, Y)
    plt.loglog(X, Y)
    plt.show()
