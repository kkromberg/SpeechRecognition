import matplotlib.pyplot as plt
import numpy as np
import argparse

y = np.fromfile('prior.txt', sep=' ')
x = np.array(xrange(0, len(y)))

plt.plot(x, y, label='State probabilities')
#plt.xticks(x)

plt.xlabel('State')
plt.ylabel('Probability')
plt.legend()
plt.grid(True)
plt.savefig('prior.png')
plt.show()
