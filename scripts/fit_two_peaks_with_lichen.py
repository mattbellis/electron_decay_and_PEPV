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
def signal(pars, x, frange=None):

    mean = pars["signal"]["mean"].value
    sigma = pars["signal"]["sigma"].value

    pdfvals = stats.norm(mean,sigma).pdf(x)

    return pdfvals
################################################################################
################################################################################
def signal2(pars, x, frange=None):

    mean = pars["signal2"]["mean"].value
    sigma = pars["signal2"]["sigma"].value

    pdfvals = stats.norm(mean,sigma).pdf(x)

    return pdfvals
################################################################################

################################################################################
def background(x, frange=None):

    # Flat
    print(frange)
    height = 1.0/(frange[1] - frange[0])

    pdfvals = height*np.ones(len(x))

    return pdfvals
################################################################################

################################################################################
def pdf(pars,x,frange=None):

    nsig = pars["signal"]["number"].value
    nsig2 = pars["signal2"]["number"].value
    nbkg = pars["bkg"]["number"].value

    ntot = float(nsig + nsig2 + nbkg)

    sig = signal(pars,x,frange=frange)
    sig2 = signal2(pars,x,frange=frange)
    bkg = background(x,frange=(8,12))

    totpdf = (nsig/ntot)*sig + (nsig2/ntot)*sig2 +  (nbkg/ntot)*bkg

    return totpdf
################################################################################

################################################################################
# Set up your parameters
################################################################################
pars = {}
pars["signal"] = {"number":Parameter(1000,(0,2000)), "mean":Parameter(9.5,(9.0,10.0)), "sigma":Parameter(0.1,(0.01,1.0))}
pars["signal2"] = {"number":Parameter(500,(0,2000)), "mean":Parameter(10.5,(10.0,12.0)),  "sigma":Parameter(0.1,(0.01,1.0))}
pars["bkg"] = {"number":Parameter(1000,(0,2000))}

################################################################################

################################################################################
# Read in
################################################################################
infilename = sys.argv[1]
dataset = np.loadtxt(infilename,dtype='float',unpack=True)
data = dataset[1]
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
binwidth=(4/100)
plt.hist(data,bins=100,range=(8,12),alpha=0.2)
lch.hist_err(data,bins=100,range=(8,12),alpha=0.2)

#plt.show()

#exit()

ysig = pars['signal']['number'].value*signal(pars,xpts) * binwidth
plt.plot(xpts,ysig,linewidth=3)

ysig2 = pars['signal2']['number'].value*signal2(pars,xpts) * binwidth
plt.plot(xpts,ysig2,'--',linewidth=3)

ybkg = pars['bkg']['number'].value*np.ones(len(xpts))/(12-8) * binwidth
plt.plot(xpts,ybkg,'-.',linewidth=3)

##plt.plot(xpts,ybkg + ysig,linewidth=3)
ntot = sum(get_numbers(pars))
ytot = ntot*pdf(pars,xpts,frange=(8,12)) * binwidth
plt.plot(xpts,ytot,linewidth=3,color='k')

plt.show()
