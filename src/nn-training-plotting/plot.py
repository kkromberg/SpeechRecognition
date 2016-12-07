import sys
import numpy as np
import matplotlib.pyplot as plt

def plot(train, test):
    plt.plot(range(1, len(train)+1), train)
    plt.plot(range(1, len(test)+1), test)
    
    plt.ylim([0, 1])
    plt.xticks(range(1, len(train)+1))
        
    plt.xlabel('Epoch')
    plt.ylabel('Frame error rate [%]')
    plt.legend()

    plt.grid(True)
    plt.savefig("plot.png")
    #plt.show()    

# Number of iterations
training_error = []
testing_error = []

with open('training.error.tmp', 'r') as f:
    for line in f:
        error = float(line.strip().split(' ')[0])
        training_error.append(error)

with open('testing.error.tmp', 'r') as f:
    for line in f:
        error = float(line.strip().split(' ')[0])
        testing_error.append(error)   
        
training_error = np.array(training_error)
testing_error = np.array(testing_error)

plot(training_error, testing_error)
            

