import numpy as np
import matplotlib.pylab as plt
import sys
from mpl_toolkits.mplot3d import Axes3D


infilename = '../electron_decay_and_PEPV/data/HE.txt'
dataset = np.loadtxt(infilename,dtype='float',unpack=True)
energy = dataset[1]
time = dataset[0]
data_array = dataset[0:2]


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#
# Create an X-Y mesh of the same dimension as the 2D data. You can
# think of this as the floor of the plot.
#
x_data, y_data = np.meshgrid( np.arange(data_array.shape[1]),
                              np.arange(data_array.shape[0]) )
#
# Flatten out the arrays so that they may be passed to "ax.bar3d".
# Basically, ax.bar3d expects three one-dimensional arrays:
# x_data, y_data, z_data. The following call boils down to picking
# one entry from each array and plotting a bar to from
# (x_data[i], y_data[i], 0) to (x_data[i], y_data[i], z_data[i]).
#
x_data = x_data.flatten()
y_data = y_data.flatten()
z_data = data_array.flatten()
ax.bar3d( x_data,
          y_data,
          np.zeros(len(z_data)),
          1, 1, z_data, bins=10 )
#
# Finally, display the plot.
#
plt.show()
