import numpy as np
import matplotlib.pylab as plt
import sys
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp


infilename = '../electron_decay_and_PEPV/data/HE.txt'
dataset = np.loadtxt(infilename,dtype='float',unpack=True)
energy = dataset[1]
time = dataset[0]


# Data -> array you would like to split
# numSlices -> number of intervals the data will be split into
# function returns a list of arrays of the Data, in even
#       intervals. Defaults to 10
def create_slices(Data, numSlices=10):
    tot = len(Data)
    interval = tot/numSlices
    Slices = []

    for s in range(0,numSlices):
        temp = []
        for i in range(0,tot):
            if(i%numSlices == s):
                temp.append(Data[i])
        Slices.append(temp)
    return Slices

def slice_at_energy(time,energy, energy_level):
    tot = len(time)
    temp = []
    for i in range(0,tot):
        if(np.abs(energy[i]-energy_level)<.2):
            temp.append(time[i])
    return temp

# arr -> List of slices, can be obtained from create_slices
#       function
# bins -> the bins of the plot, defaults to 100
# function returns a plot of all slices, overlaying each
#       other, in different colors
def plot_slices(arr, bins=100):
    c = 0
    for i in arr:
        lb = str(c)
        plt.hist(i, bins=bins, label=lb)
        c = c+1
    plt.legend()
    plt.show()


def expon(t,A0,k):
    return A0*exp(-k*t)

def fit_exponential(arr, initial, bins=100):
    y, x, _ = plt.hist(arr,bins=bins)
    x = x[1:] 
    popt,pcov = curve_fit(expon,x,y,p0=initial)
    plt.plot(x,y,'b+:',label='data')
    plt.plot(x,expon(x,*popt),'ro:',label='fit')
    plt.legend()
    return [popt, pcov]







    
