import matplotlib.pyplot as plt
import numpy as np
import argparse

y = np.fromfile('prior_alignment.txt', sep=' ')
y2 = np.fromfile('prior_model1.txt', sep=' ')
y3 = np.fromfile('prior_model5.txt', sep=' ')
x = np.array(xrange(0, len(y)))

plt.plot(x, y, label='alignment')
plt.plot(x, y2, label='model_1')
plt.plot(x, y3, label='model_5')
#plt.xticks(x)
plt.xlabel('State')
plt.ylabel('Probability')
plt.legend()
plt.grid(True)
plt.savefig('prior.png')
plt.show()
