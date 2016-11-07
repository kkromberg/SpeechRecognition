import matplotlib.pyplot as plt
import numpy as np
import argparse	

parser = argparse.ArgumentParser()
parser.add_argument("file", type=str, default='',
	                help="Path to the file with 2D columns of time steps and energies")
args = parser.parse_args()

x = []
y = []
boundaries = []
with open(args.file, 'r') as f:
    for line in f:
        split_line = line.strip().split(' ')
        if len(split_line) != 2:
            continue
            
        if float(split_line[1]) == 0.15 or float(split_line[1]) == -0.1:
            boundaries.append(int(split_line[0]))
            continue
            
        x.append( int(split_line[0]) )
        y.append( float(split_line[1]) )
        
    x = np.array(x)
    y = np.array(y)
    boundaries = np.array(boundaries)
    
energies = plt.plot(x, y, label="Energies")
boundaries_plt = plt.vlines(boundaries, np.min(y), np.max(y), colors='r', label="Speech boundaries")
plt.xlabel('time frame')
plt.ylabel('log(energy)')
plt.legend()

plt.grid(True)
plt.savefig(args.file + ".png")
plt.show()

