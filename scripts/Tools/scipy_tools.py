import numpy as np
import matplotlib.pylab as plt
import sys
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp


infilename = '../../data/HE.txt'
dataset = np.loadtxt(infilename,dtype='float',unpack=True)
energy = dataset[1]
time = dataset[0] / 60 / 60 / 24




#----------Data Tools----------------------------------------------
# arr1 -> limiting array
# arr2 -> Secondary Array
# low, high -> The limits with respect to arr1
# function returns two arrays of the same size based on the limits
#       put forth on arr1.
# Ex: If arr1 is energy and arr2 is time, then both arrays will be
#       returned based on an interval of energy
def limit_data(arr1, arr2,  low, high):
    tempE = []
    tempT = []
    for i in range(0,len(arr1)):
        if(arr1[i]>low and arr1[i]<high):
            tempE.append(arr1[i])
            tempT.append(arr2[i])
    return tempE, tempT

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

# arr1 -> Array to be sliced
# arr2 -> Array containing value
# value -> Value where slice will occur
# function returns a slice of arr1 at the value desired in arr2
# Ex: If arr1 is energy and arr2 is time, then a slice of energy will
#       be returned at a specific time. (Energies at day 200)
def slice_at_value(arr1,arr2, value, threshold=.2):
    tot = len(arr1)
    temp = []
    for i in range(0,tot):
        if(np.abs(arr2[i]-value)<threshold):
            temp.append(arr1[i])
    return temp

def energy_after_time(time, energy, value):
    temp = []
    for i in range(len(time)):
        if(time[i]>value):
            temp.append(energy[i])
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
# -----------------------------------------------------------------



# ------------Gaussian Tools---------------------------------------
def gauss(x,a,x0,sigma, B):
    return a*exp(-(x-x0)**2/(2*sigma**2)) + B

def find_background(arr):
    background = np.array([])
    for i in arr:
        if(i<9.5 or i>10.5):
            background = np.append(background, [i])
        
    return np.mean(background)

def fit_gauss(arr, initial, bins=100):
    y, x, _ = plt.hist(arr,bins=bins)
    x = x[1:] 
    popt,pcov = curve_fit(gauss,x,y,p0=initial)
    plt.plot(x,y,'b+:',label='data')
    plt.plot(x,expon(x,*popt),'ro:',label='fit')
    plt.legend()
    return [popt, pcov]
# -----------------------------------------------------------------



# ------------Exponential Tools------------------------------------
def expon(t,A0,k, B):
    return A0*exp(-k*t) + B

def fit_exponential(arr, initial, bins=100, colors=['c', 'b', 'r']):
    y, x, _ = plt.hist(arr,bins=bins, color=colors[0])
    x = x[1:] 
    popt,pcov = curve_fit(expon,x,y,p0=initial)
    plt.plot(x,y,'+:',label='data', color=colors[1])
    plt.plot(x,expon(x,*popt),'o:',label='fit', color=colors[2])
    plt.legend()
    return [popt, pcov]

def hflife_convert(hlorlambda):
    return np.log(2)/hlorlambda
# -----------------------------------------------------------------


#------------Generate Plots--------------------------------------
def time_v_energy_plot(nbins=100, x1=8, x2=12):
    newEnergy, newTime = limit_data(energy, time, x1, x2)
    fig, ax = plt.subplots()
    counts, xedges, yedges, im = ax.hist2d(newEnergy, newTime, bins=nbins)
    plt.colorbar(im, ax=ax)
    plt.xlabel('Energy (kEv)')
    plt.ylabel('Time (days)')
    plt.title('Time vs Energy')
    plt.show()

def decay_rates_plot(nbins=200):
    timeGa = slice_at_value(time, energy, 10.3)
    timeZn = slice_at_value(time, energy, 9.6)
    timeAr = slice_at_value(time, energy, 11.1)
    fit_exponential(timeGa, [100, .008, 4], bins=200)
    fit_exponential(timeZn, [200, .008, 4], bins=200, colors=['y', 'c', 'g'])
    fit_exponential(timeAr, [10, .017, 4], bins=200, colors=['m', 'c', 'k'])
    plt.xlabel('Time (days)')
    plt.ylabel('Counts (bins=200)')
    plt.title('Decay Rates of Ga, Zn, Ar')
    plt.show()
