'''
This script is to visualize the activation 
from the student network
'''

import numpy as np
import matplotlib.pyplot as plt

filepath = '/Users/chih-wei/Desktop/chi/0_cw_workspace/unlabeledDrumDataset/evaluation_enst/Activations/drummer2/115_min.npy'

all = np.load(filepath)
print len(all[0])
plt.plot(all[0])
plt.xlabel('block index')
plt.ylabel('activity')
plt.title('kd')
plt.show()