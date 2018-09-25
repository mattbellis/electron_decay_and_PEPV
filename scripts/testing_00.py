#Imports
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import interactive
interactive(True)

#Load in the data - May want to add try/catch block
data = np.loadtxt('../data/data_for_testing_00.dat',)
#Load Columns
day = data[:,0]
energy = data[:,1]
riseTime = data[:,2]

#Test Plot 1 - Plotting a histo of the energy vs time
n_bins = 20
fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

# We can set the number of bins with the `bins` kwarg
axs[0].hist(day, bins=n_bins)
axs[1].hist(energy, bins=n_bins)
