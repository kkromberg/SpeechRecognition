import sys
import numpy as np
import matplotlib.pyplot as plt

def print_usage():
    print "Usage: plot.py <file1> <file2> ... <fileN>"
    
def plot(X, Y, labels):
    for i, label in enumerate(labels):
        energies = plt.plot(X, Y[i,:] , label=label)
        
    plt.ylim([np.min(Y), np.max(Y)+5])
    plt.xticks(X)
        
    plt.xlabel('Iteration')
    plt.ylabel('Average score per frame')
    plt.legend()

    plt.grid(True)
    plt.savefig("plot.png")
    plt.show()    


print_usage()

# Number of iterations
iterations = []
scores = []
names = []

# Go through each input file
for i in range(1, len(sys.argv)):

    # Store the file name (the last entry of a path)
    names.append(sys.argv[i].split('/')[-1])
    
    # Open the file 
    with open(sys.argv[i], 'r') as f:
    
        iteration_counter = 0
        new_scores = []
    
        for line in f:
            # Split the line based on whitespace
            split_line = line.strip().split(' ')
            
            if len(split_line) != 4:
                continue
                
            # Add the iteration counter if necessary
            if not iteration_counter in iterations:
                iterations.append(iteration_counter)
                
            # Add the score of the current iteration to the file 
            score = float(split_line[3])
            new_scores.append(score)
            
            iteration_counter += 1
            
        scores.append(new_scores)
         
iterations = np.array(iterations)
scores = np.array(scores)

plot(iterations, scores, names)
            

