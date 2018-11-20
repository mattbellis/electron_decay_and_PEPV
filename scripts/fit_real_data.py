import numpy as np
import matplotlib.pylab as plt

import scipy.stats as stats

from scipy.optimize import approx_fprime,fmin_bfgs

from lichen.fit import Parameter,get_numbers,reset_parameters,pois,errfunc,pretty_print_parameters,get_values_and_bounds,fit_emlm
import lichen.lichen as lch

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
    nbkg = pars["bkg"]["number"].value

    ntot = float(npeak0 + npeak1 + npeak2 + nbkg)

    p0 = peak0(pars,x,frange=frange)
    p1 = peak1(pars,x,frange=frange)
    p2 = peak2(pars,x,frange=frange)
    bkg = background(x,frange=(8,12))

    totpdf = (npeak0/ntot)*p0 + (npeak1/ntot)*p1 +  (npeak2/ntot)*p2 + (nbkg/ntot)*bkg

    return totpdf
################################################################################

################################################################################
# Set up your parameters
################################################################################
pars = {}
pars["peak0"] = {"number":Parameter(1000,(0,5000)), "mean":Parameter(8.9,(8.5,9.0)), "sigma":Parameter(0.1,(0.01,1.0))}
pars["peak1"] = {"number":Parameter(500,(0,5000)), "mean":Parameter(9.7,(9.5,10.0)),  "sigma":Parameter(0.1,(0.01,1.0))}
pars["peak2"] = {"number":Parameter(500,(0,10000)), "mean":Parameter(10.3,(10.0,11.0)),  "sigma":Parameter(0.1,(0.01,1.0))}
pars["bkg"] = {"number":Parameter(1000,(0,10000))}

################################################################################

################################################################################
# Read in
################################################################################
infilename = sys.argv[1]
dataset = np.loadtxt(infilename,dtype='float',unpack=True)
data = dataset[1]

# Select a subset of the data
data = data[(data>8)*(data<12)]
#data += stats.norm(pars["signal2"]["mean"].value,pars["signal2"]["sigma"].value).rvs(size=500).tolist()
#data += (10*np.random.random(1000)).tolist()

#print(data)

initvals,finalvals = fit_emlm(pdf,pars,data)
print("Done with fit!")
pretty_print_parameters(pars)

################################################################################
# Plot the results!
################################################################################

xpts = np.linspace(8,12,1000)

plt.figure()
binwidth=(4/200)
plt.hist(data,bins=200,range=(8,12),alpha=0.2)
lch.hist_err(data,bins=200,range=(8,12),alpha=0.2)

#plt.show()

#exit()

ysig0 = pars['peak0']['number'].value*peak0(pars,xpts) * binwidth
plt.plot(xpts,ysig0,linewidth=3)

ysig1 = pars['peak1']['number'].value*peak1(pars,xpts) * binwidth
plt.plot(xpts,ysig1,'--',linewidth=3)

ysig2 = pars['peak2']['number'].value*peak2(pars,xpts) * binwidth
plt.plot(xpts,ysig2,'--',linewidth=3)

ybkg = pars['bkg']['number'].value*np.ones(len(xpts))/(12-8) * binwidth
plt.plot(xpts,ybkg,'-.',linewidth=3)

##plt.plot(xpts,ybkg + ysig,linewidth=3)
ntot = sum(get_numbers(pars))
ytot = ntot*pdf(pars,xpts,frange=(8,12)) * binwidth
plt.plot(xpts,ytot,linewidth=3,color='k')

plt.show()
