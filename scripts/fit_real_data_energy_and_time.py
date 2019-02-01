import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm


import scipy.stats as stats

from scipy.optimize import approx_fprime,fmin_bfgs
from scipy.integrate import dblquad

from lichen.fit import Parameter,get_numbers,reset_parameters,pois,errfunc,pretty_print_parameters,get_values_and_bounds,fit_emlm
import lichen.lichen as lch
import lichen.pdfs as pdfs

import numpy as np

import sys

first_event = 2750361.2
subnormranges = [[1,68],[75,102],[108,306],[309,459],[551,1238]]

np.random.seed(0)

LN2 = np.log(2)

################################################################################
def peak(pars, x, frange=None, key=None,subnormranges=None):

    mean = pars[key]["mean"].value
    sigma = pars[key]["sigma"].value
    lifetime = pars[key]["lifetime"].value
    #print(lifetime)

    rv = stats.expon(scale=lifetime)
    pdftime,xnorm = pdfs.nnf(rv,normrange=frange,data=x[1],subnormranges=subnormranges)

    pdfenergy = stats.norm(mean,sigma).pdf(x[0])
    
    pdfvals = pdfenergy*pdftime

    #print("peak")
    #print(pdfvals)

    return pdfvals,[pdfenergy,pdftime]
################################################################################

################################################################################
def background(x, frange=None):

    # Flat
    #print(frange)
    height0 = 1.0/(frange[1] - frange[0])
    #height1 = 1.0/(1238.0)
    height1 = 1.0
    delta = 0.0
    for sr in subnormranges:
        delta += sr[1]-sr[0]
    height1 = 1/delta

    xpts = np.ones(len(x[0]))
    pdfvals = height0*height1*xpts

    return pdfvals,[height0*xpts,height1*xpts]
################################################################################

################################################################################
def pdf(pars,x,frange=None):

    npeak0 = pars["peak0"]["number"].value
    npeak1 = pars["peak1"]["number"].value
    npeak2 = pars["peak2"]["number"].value
    npeak3 = pars["peak3"]["number"].value
    npeak4 = pars["peak3"]["number"].value
    nbkg = pars["bkg"]["number"].value

    ntot = float(npeak0 + npeak1 + npeak2 + npeak3 + npeak4 + nbkg)
    #print("ntot: ",ntot)

    bkg = background(x,frange=(8,12))[0]
    p0 = peak(pars,x,frange=(0,1238),key="peak0")[0]
    p1 = peak(pars,x,frange=(0,1238),key="peak1")[0] 
    p2 = peak(pars,x,frange=(0,1238),key="peak2")[0]
    p3 = peak(pars,x,frange=(0,1238),key="peak3")[0] 
    p4 = peak(pars,x,frange=(0,1238),key="peak4")[0] 

    totpdf = (npeak0/ntot)*p0 + (npeak1/ntot)*p1 + (npeak2/ntot)*p2 + (npeak3/ntot)*p3 + (npeak4/ntot)*p4   + (nbkg/ntot)*bkg

    return totpdf
################################################################################

################################################################################
# Set up your parameters
################################################################################
#Restricted 
pars = {}
pars["peak0"] = {"number":Parameter(2250,(1800,2300)), "mean":Parameter(8.9,(8.8,9.0)), "sigma":Parameter(0.135,(0.10,1)), "lifetime":Parameter(244/LN2,None)}
pars["peak1"] = {"number":Parameter(600,(500,800)), "mean":Parameter(9.7,(9.6,9.8)),  "sigma":Parameter(0.097,(0.05,.15)), "lifetime":Parameter(271/LN2,None)}
pars["peak2"] = {"number":Parameter(2200,(2000,10000)), "mean":Parameter(10.3,(10.1,10.32)),  "sigma":Parameter(0.078,(0.01,.15)), "lifetime":Parameter(271/LN2,None)}
pars["peak3"] = {"number":Parameter(3400,(2000,10000)), "mean":Parameter(10.39,(10.3,10.5)),  "sigma":Parameter(0.08,(0.01,.15)), "lifetime":Parameter(271/LN2,None)}
pars["peak4"] = {"number":Parameter(50,(2,10000)), "mean":Parameter(11.10,(11.0,11.2)),  "sigma":Parameter(0.08,(0.01,.15)), "lifetime":Parameter(80/LN2,None)}
pars["bkg"] = {"number":Parameter(2900,(2000,10000))}
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
infilename = sys.argv[1]
dataset = np.loadtxt(infilename,dtype='float',unpack=True)
energy = dataset[1]
tdays = (dataset[0]-first_event)/(24.0*3600.0) + 1.0

#print(tdays)

# Select a subset of the data
idx = (energy>8)*(energy<12)
tidx = np.zeros(len(tdays),dtype=bool)
# Cut out the subranges, just to be sure
#subnormranges = [[600,1238]]
for sr in subnormranges:
    print(sr)
    tidx += (tdays>sr[0])*(tdays<sr[1])
    print(len(tidx[tidx]))
idx *= tidx
tdays = tdays[idx]
energy = energy[idx]
data = [energy,tdays]
#print(max(tdays))
print("# events: {0}".format(len(tdays)))
#exit()

plt.figure()
plt.plot(energy,tdays,'.',alpha=0.3)
#plt.show()
#exit()

############## 3D PLOTS ####################################
#X,Y = np.meshgrid(np.linspace(min(energy),max(energy),25),np.linspace(min(tdays),max(tdays),25))
#Z0 = peak(pars,[X,Y],frange=(0,1238),key='peak0')
#Z1 = peak(pars,[X,Y],frange=(0,1238),key='peak1')
#Z2 = peak(pars,[X,Y],frange=(0,1238),key='peak2')
#Z3 = peak(pars,[X,Y],frange=(0,1238),key='peak3')

#fig = plt.figure()
#ax = fig.add_subplot(1, 1, 1, projection='3d')
#p0 = ax.plot_wireframe(X, Y, Z0[0], rstride=1, cstride=1)#, cmap=cm.coolwarm, linewidth=0)#, antialiased=False)
#p1 = ax.plot_wireframe(X, Y, Z1[0], rstride=1, cstride=1)#, cmap=cm.coolwarm, linewidth=0)#, antialiased=False)
#p2 = ax.plot_wireframe(X, Y, Z2[0], rstride=1, cstride=1)#, cmap=cm.coolwarm, linewidth=0)#, antialiased=False)
#p3 = ax.plot_wireframe(X, Y, Z3[0], rstride=1, cstride=1)#, cmap=cm.coolwarm, linewidth=0)#, antialiased=False)
#plt.show()

rv = stats.expon(scale=250)
pdftime,xnorm = pdfs.nnf(rv,normrange=(0,1238),data=data[1],subnormranges=subnormranges)
#print(xnorm)
ynorm,xnorm = pdfs.nnf(rv,normrange=(0,1238),data=xnorm,subnormranges=subnormranges)

#print(xnorm)
#plt.figure()
#plt.plot(xnorm,ynorm,'.')
#plt.show()
#exit()



#exit()


################################################################################
# Test to see if it is normalized
def tmppeak(y,x,args):
    #print(args)
    ret = peak(args,[x,y],frange=(0,1238),key='peak1')
    return ret[0]

print(tmppeak(tdays,energy,pars))
#tmp = dblquad(tmppeak,min(energy),max(energy),gfun=min(tdays),hfun=max(tdays),args=[pars])
#print(tmp)
#exit()
################################################################################



initvals,finalvals = fit_emlm(pdf,pars,data,verbose=True)
print("Done with fit!")
pretty_print_parameters(pars)

################################################################################
# Plot the results!
################################################################################

xpts0 = np.linspace(8,12,1000)
xpts1 = np.linspace(0,1238,1000)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
binwidth=(4/200)
plt.hist(data[0],bins=200,range=(8,12),alpha=0.2)
lch.hist_err(data[0],bins=200,range=(8,12),alpha=0.2)

plt.subplot(1,2,2)
binwidth=(1238/200)
plt.hist(data[1],bins=200,range=(0,1238),alpha=0.2)
lch.hist_err(data[1],bins=200,range=(0,1238),alpha=0.2)
#plt.show()

#exit()

plt.subplot(1,2,1)
ytot = np.zeros_like(xpts0)
for key in ['peak0','peak1','peak2','peak3','peak4']:
    ysig = peak(pars,[xpts0,xpts1],frange=(0,1238),key=key)
    y = pars[key]['number'].value*ysig[1][0]*(4/200.)
    ytot += y
    plt.plot(xpts0,y)
ysig = background([xpts0,xpts1],frange=(8,12))
y = pars['bkg']['number'].value*ysig[1][0]*(4/200.)
ytot += y
plt.plot(xpts0,y)
plt.plot(xpts0,ytot,'k-',linewidth=2)


plt.subplot(1,2,2)
xpts0 = np.linspace(8,12,100)
for sr in subnormranges:
    xptstime = np.linspace(sr[0],sr[1],100)
    ytot = np.zeros_like(xptstime)
    for key,color in zip(['peak0','peak1','peak2','peak3','peak4'],['b','g','r','m','b']):
        ysig = peak(pars,[xpts0,xptstime],frange=(0,1238),key=key)
        y = pars[key]['number'].value*ysig[1][1]*(1238/200.)
        ytot += y
        plt.plot(xptstime,y,'-',color=color)
    ysig = background([xpts0,xptstime],frange=(8,12))
    y = pars['bkg']['number'].value*ysig[1][1]*(1238/200.)
    ytot += y
    plt.plot(xptstime,y,'y.')
    plt.plot(xptstime,ytot,'k-')

'''
ysig = pars['peak0']['number'].value*peak(pars,[xpts0,xpts1],frange=(8,12),key="peak0") * binwidth
plt.subfigure(1,2,1)
plt.plot(xpts0,ysig[1][0],linewidth=3)
plt.subfigure(1,2,2)
plt.plot(xpts1,ysig[1][1],linewidth=3)
'''

'''
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
'''

plt.show()
