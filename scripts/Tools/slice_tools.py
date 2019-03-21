import numpy as np
import matplotlib.pylab as plt
import sys
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp


infilename = '../electron_decay_and_PEPV/data/HE.txt'
dataset = np.loadtxt(infilename,dtype='float',unpack=True)
energy = dataset[1]
time = dataset[0] / 60 / 60 / 24


# Data -> array you would like to split
# numSlices -> number of intervals the data will be split into
# function returns a list of arrays of the Data, in even
#       intervals. Defaults to 10



def limit_data(arr1, arr2,  low, high):
    tempE = []
    tempT = []
    for i in range(0,len(arr1)):
        if(arr1[i]>low and arr1[i]<high):
            tempE.append(arr1[i])
            tempT.append(arr2[i])
    return tempE, tempT

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

def slice_at_value(arr1,arr2, value, threshold=.2):
    tot = len(arr1)
    temp = []
    for i in range(0,tot):
        if(np.abs(arr2[i]-value)<threshold):
            temp.append(arr1[i])
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


def hflife_convert(hlorlambda):
    return np.log(2)/hlorlambda

def time_v_energy_plot(nbins=100, x1=8, x2=12):
    newEnergy, newTime = limit_data(energy, time, x1, x2)
    fig, ax = plt.subplots()
    counts, xedges, yedges, im = ax.hist2d(newEnergy, newTime, bins=nbins)
    plt.colorbar(im, ax=ax)
    plt.xlabel('Energy (kEv)')
    plt.ylabel('Time (days)')
    plt.title('Time vs Energy')
    plt.show()

