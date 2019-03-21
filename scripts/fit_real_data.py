import numpy as np
import matplotlib.pylab as plt

import scipy.stats as stats

from scipy.optimize import approx_fprime,fmin_bfgs

from lichen.fit import Parameter,get_numbers,reset_parameters,pois,errfunc,pretty_print_parameters,get_values_and_bounds,fit_emlm
import lichen as lch

import numpy as np

import sys

np.random.seed(0)

################################################################################
def peak0(pars, x, frange=None):

    mean = pars["peak0"]["mean"].value
    sigma = pars["peak0"]["sigma"].value

    pdfvals = stats.norm(mean,sigma).pdf(x)

    return pdfvals
################################################################################
################################################################################
def peak1(pars, x, frange=None):

    mean = pars["peak1"]["mean"].value
    sigma = pars["peak1"]["sigma"].value

    pdfvals = stats.norm(mean,sigma).pdf(x)

    return pdfvals
################################################################################

################################################################################
def peak2(pars, x, frange=None):

    mean = pars["peak2"]["mean"].value
    sigma = pars["peak2"]["sigma"].value

    pdfvals = stats.norm(mean,sigma).pdf(x)

    return pdfvals
################################################################################

################################################################################
def peak3(pars, x, frange=None):

    mean = pars["peak3"]["mean"].value
    sigma = pars["peak3"]["sigma"].value

    pdfvals = stats.norm(mean,sigma).pdf(x)

    return pdfvals
################################################################################

################################################################################
def background(x, frange=None):

    # Flat
    #print(frange)
    height = 1.0/(frange[1] - frange[0])

    pdfvals = height*np.ones(len(x))

    return pdfvals
################################################################################

################################################################################
def pdf(pars,x,frange=None):

    npeak0 = pars["peak0"]["number"].value
    npeak1 = pars["peak1"]["number"].value
    npeak2 = pars["peak2"]["number"].value
    npeak3 = pars["peak3"]["number"].value
    nbkg = pars["bkg"]["number"].value

    ntot = float(npeak0 + npeak1 + npeak2 + npeak3 + nbkg)

    bkg = background(x,frange=(8,12))
    p0 = peak0(pars,x,frange=frange) 
    p1 = peak1(pars,x,frange=frange) 
    p2 = peak2(pars,x,frange=frange)
    p3 = peak3(pars,x,frange=frange) 

   
    totpdf = (npeak0/ntot)*p0 + (npeak1/ntot)*p1 + (npeak2/ntot)*p2 + (npeak3/ntot)*p3  + (nbkg/ntot)*bkg

    return totpdf
################################################################################

################################################################################
# Set up your parameters
################################################################################
#Restricted 
pars = {}
pars["peak0"] = {"number":Parameter(2250,(0,5000)), "mean":Parameter(8.9,(8.5,9.1)), "sigma":Parameter(0.25,(0.10,1))}
pars["peak1"] = {"number":Parameter(600,(0,5000)), "mean":Parameter(9.7,(9.6,9.8)),  "sigma":Parameter(0.15,(0.05,.3))}
pars["peak2"] = {"number":Parameter(2200,(0,10000)), "mean":Parameter(10.3,(10.1,10.32)),  "sigma":Parameter(0.05,(0.01,.08))}
pars["peak3"] = {"number":Parameter(3400,(0,10000)), "mean":Parameter(10.39,(10.3,10.5)),  "sigma":Parameter(0.06,(0.01,.07))}
pars["bkg"] = {"number":Parameter(3000,(0,10000))}
#UnRestricted
#pars = {}
#pars["peak0"] = {"number":Parameter(2250,(0,5000)), "mean":Parameter(8.9,(8.5,9.1)), "sigma":Parameter(0.25,(0.10,1))}
#pars["peak1"] = {"number":Parameter(600,(0,5000)), "mean":Parameter(9.7,(9.6,9.8)),  "sigma":Parameter(0.15,(0.05,.3))}
##pars["peak2"] = {"number":Parameter(2200,(0,10000)), "mean":Parameter(10.33,(10.1,10.36)),  "sigma":Parameter(0.05,(0.01,.2))}
#pars["peak3"] = {"number":Parameter(3400,(0,10000)), "mean":Parameter(10.39,(10.3,10.5)),  "sigma":Parameter(0.06,(0.01,.1))}
#pars["bkg"] = {"number":Parameter(3000,(0,10000))}

################################################################################

################################################################################
# Read in
################################################################################
infilename = "../data/HE.txt"
dataset = np.loadtxt(infilename,dtype='float',unpack=True)
data = dataset[1]

# Select a subset of the data
data = data[(data>8)*(data<12)]
#data += stats.norm(pars["signal2"]["mean"].value,pars["signal2"]["sigma"].value).rvs(size=500).tolist()
#data += (10*np.random.random(1000)).tolist()

#print(data)

initvals,finalvals = fit_emlm(pdf,pars,[data])
print("Done with fit!")
pretty_print_parameters(pars)

################################################################################
# Plot the results!
################################################################################

xpts = np.linspace(8,12,1000)

plt.figure()
binwidth=(4/200)
plt.hist(data,bins=200,range=(8,12),alpha=0.2)
lch.hist(data,bins=200,range=(8,12),alpha=0.2)

#plt.show()

#exit()

ysig0 = pars['peak0']['number'].value*peak0(pars,xpts) * binwidth
plt.plot(xpts,ysig0,linewidth=3)

ysig1 = pars['peak1']['number'].value*peak1(pars,xpts) * binwidth
plt.plot(xpts,ysig1,'--',linewidth=3)

ysig2 = pars['peak2']['number'].value*peak2(pars,xpts) * binwidth
plt.plot(xpts,ysig2,'--',linewidth=3)

ysig3 = pars['peak3']['number'].value*peak3(pars,xpts) * binwidth
plt.plot(xpts,ysig3,'--',linewidth=3)

ybkg = pars['bkg']['number'].value*np.ones(len(xpts))/(12-8) * binwidth
plt.plot(xpts,ybkg,'-.',linewidth=3)

##plt.plot(xpts,ybkg + ysig,linewidth=3)
ntot = sum(get_numbers(pars))
print(ntot)
ytot = ntot*pdf(pars,xpts,frange=(8,12)) * binwidth
plt.plot(xpts,ytot,linewidth=3,color='k')

plt.show()
