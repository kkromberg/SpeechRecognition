import matplotlib.pyplot as plt
import math
import numpy as np


def plotRelativeSentenceLength(relativeSentenceLength, totalSentenceLength, averageSentenceLength):
    print totalSentenceLength
    X = []
    x_range = []
    Y = []
    counter = 0
    for elem in relativeSentenceLength.iteritems():
        X.append(elem[0])
        Y.append(-1*math.log10(float(elem[0])*elem[1]/totalSentenceLength))
        x_range.append(counter)
        counter += 1
    plt.title('Relative sentence length frequencies')
    plt.xlabel('Sentence length')
    plt.ylabel('negative log-likelihood')
    plt.xticks(x_range, X, rotation=90)
    plt.plot(x_range, Y)
    plt.plot(x_range, Y, 'or')
    plt.axvline(x=averageSentenceLength-1)
    plt.show()
