#----------------IMPORTS-------------------------------
# Import System and append tools path
import sys
sys.path.append('./Tools/')

# Lichen Imports
from lichen.fit import Parameter

#Tool Imports
from fit_gauss_lichen import run_fit


pars = {}
pars["peak0"] = {"number":Parameter(2250,(0,5000)), "mean":Parameter(8.9,(8.5,9.1)), "sigma":Parameter(0.25,(0.10,1))}
pars["peak1"] = {"number":Parameter(600,(0,5000)), "mean":Parameter(9.7,(9.6,9.8)),  "sigma":Parameter(0.15,(0.05,.3))}
pars["peak2"] = {"number":Parameter(2200,(0,10000)), "mean":Parameter(10.3,(10.1,10.32)),  "sigma":Parameter(0.05,(0.01,.08))}
pars["peak3"] = {"number":Parameter(3400,(0,10000)), "mean":Parameter(10.39,(10.3,10.5)),  "sigma":Parameter(0.06,(0.01,.07))}
pars["bkg"] = {"number":Parameter(3000,(0,10000))}

run_fit(pars, '../data/HE.txt')
