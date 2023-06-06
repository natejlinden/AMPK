import numpy as np
from SALib.sample import sobol as sobol_samp
from SALib.sample import morris as morris_samp
from SALib.analyze import sobol as sobol_analyze
from SALib.analyze import morris as morris_analyze
from SALib.analyze.hdmr import analyze as hdmr_analyze
import os
import sys

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from scipy.stats import t

############################################
# get user inputs
############################################
if len(sys.argv) < 5:
    print("Incorrect Usage: TODO: add usage") # TODO: add usage
    sys.exit(1)

base_name       = sys.argv[1]
nominals_file   = sys.argv[2]
bounds_file     = sys.argv[3]
figpath         = sys.argv[4]

############################################
# Matplotlib settings
############################################
plt.style.use('~/.matplotlib/custom.mplstyle')
mpl.rcParams['figure.autolayout'] = True

# construct save dir
savedir = base_name + '/'

####### PREPROCESSING #######
# load parameter samples and simulation results
param_vals_sobol_corr = np.load('./' + savedir + 'param_vals_sobol_MA_corr.npy')
sobol_sols_corr = np.load('./' + savedir + 'sols_sobol_corr.npy')

# reshape into 2D array
sobol_sols_corr = sobol_sols_corr.reshape(sobol_sols_corr.shape[0]*sobol_sols_corr.shape[1], sobol_sols_corr.shape[2])
nsols, nqoi = sobol_sols_corr.shape

############################################
# Bounds and other info for the GSA #
############################################
# define the bounds for the AMPK parameters
# we use plus or minus on order of magnitude of any known values and then make reasonable assumptions for unknowns
# Note we fix all off rates to 1.0 and dont bother sampling these or computing sensitivities
############################################
try:
    nominals_file = pd.read_csv(nominals_file)
except:
    print("An error occurred while reading the nominal values.")

try:
    bounds = pd.read_csv(bounds_file)
except:
    print("An error occurred while reading the nominal values.")

nominals = nominals_file['value'].to_list()
param_names = nominals_file['parameter'].to_list()
nparam = len(param_names)
bounds = [[lb, ub] for lb, ub in zip(bounds['lb'].to_list(), bounds['ub'].to_list())]

problem = {'num_vars':nparam, 'names':param_names, 'bounds': bounds,}
    
################################################
# Load parameter samples and qois
################################################
param_vals_sobol = np.load('./' + savedir + 'param_vals_sobol.npy')
sobol_sols = np.load('./' + savedir + 'sols_sobol.npy')

# reshape into 2D array
sobol_sols = sobol_sols.reshape(sobol_sols.shape[0]*sobol_sols.shape[1],
                                          sobol_sols.shape[2])
nsols, nqoi = sobol_sols.shape

################################################
#  HISTOGRAM OF QOI
################################################
fig, ax = plt.subplots(1,1, figsize=(2.25,3))
ax.hist(sobol_sols_corr[:,0])
ax.set_xlabel(r'$pAMPKAR/AMPKAR_{tot}$')
ax.set_ylabel('count')
fig.savefig(figpath + savedir + 'qoi_hist.pdf')
plt.show()

################################################
####### COMPUTE SENSITIVITIY INDICES #######
################################################
Si_sobol = sobol_analyze.analyze(problem, sobol_sols_corr[:,2], calc_second_order=True)
np.save('./' + savedir + 'sobol_pampkar_final.npy', Si_sobol)
Si_hdmr = hdmr_analyze(problem, param_vals_sobol_corr, sobol_sols_corr[:,2])
np.save('./' + savedir + 'hdmr_pampkar_final.npy', Si_hdmr)

################################################
####### SORT #######
################################################
# sort sobol indices by ST
dtype = [('name', 'U10'), ('S1', float), ('S1_conf', float), 
            ('ST', float), ('ST_conf', float)]

Si_sobol_sorted = np.array([(name, S1, S1_conf, ST, ST_conf) for 
                            name, S1, S1_conf, ST, ST_conf 
                            in zip(param_names, Si_sobol['S1'], 
                                    Si_sobol['S1_conf'], Si_sobol['ST'], 
                                    Si_sobol['ST_conf'])], dtype=dtype)
Si_sobol_sorted = np.sort(Si_sobol_sorted, order='ST')[::-1]

# sort hdmr by S
dtype = [('name', 'U10'), ('Sa', float), ('Sa_conf', float), ('Sb', float), 
            ('Sb_conf', float), ('S', float), ('S_conf', float), ('ST', float), 
            ('ST_conf', float)]
Si_hdmr_sorted = np.array([(name, Sa, Sa_conf, Sb, Sb_conf, S, S_conf, ST, ST_conf) 
                                    for name, Sa, Sa_conf, Sb, Sb_conf, S, S_conf, ST, ST_conf 
                                    in zip(param_names, Si_hdmr['Sa'], Si_hdmr['Sa_conf'], 
                                        Si_hdmr['Sb'], Si_hdmr['Sb_conf'],
                                        Si_hdmr['S'], Si_hdmr['S_conf'],
                                        Si_hdmr['ST'], Si_hdmr['ST_conf'])], dtype=dtype)
Si_hdmr_sorted = np.sort(Si_hdmr_sorted, order='S')[::-1]

################################################
####### PLOT #######
################################################
# sobol first order
fig, ax = plt.subplots(figsize=(4,3))
ax.bar(np.arange(0,len(param_names)), Si_sobol_sorted['S1'],
        yerr=Si_sobol_sorted['S1_conf'],
        log=False)
plt.xticks(np.arange(0,len(param_names)), Si_sobol_sorted['name'], rotation='vertical')
plt.ylabel('Sobol First Order')
fig.savefig(figpath + savedir + 'S1.pdf')
plt.show()

# sobol total order
fig, ax = plt.subplots(figsize=(4,3))
ax.bar(np.arange(0,len(param_names)), Si_sobol_sorted['ST'],
        yerr=Si_sobol_sorted['ST_conf'],
        log=False)
plt.xticks(np.arange(0,len(param_names)), Si_sobol_sorted['name'], rotation='vertical')
plt.ylabel('Sobol Total Order')
fig.savefig(figpath + savedir + 'ST.pdf')
plt.show()

# TODO - plot second order sobol indices

# HDMR indices
fig, ax = plt.subplots(figsize=(4,3))
ax.bar(np.arange(0,len(Si_hdmr_sorted['Sa'][0:len(param_names)])), 
    Si_hdmr_sorted['Sa'][0:len(param_names)],
    bottom=np.zeros(len(Si_hdmr_sorted['Sa'][0:len(param_names)])), 
    label=r'$Sa$ (Structural)')
ax.bar(np.arange(0,len(Si_hdmr_sorted['Sb'][0:len(param_names)])), 
    Si_hdmr_sorted['Sb'][0:len(param_names)],
    bottom=Si_hdmr_sorted['Sa'][0:len(param_names)], 
    label=r'$Sb$ (Correlated)',
    yerr=Si_hdmr_sorted['S_conf'])
plt.xticks(np.arange(0, len(Si_hdmr_sorted['name'][0:len(param_names)])), 
            Si_hdmr_sorted['name'][0:len(param_names)], rotation='vertical')
plt.ylabel('HDMR Contribution')
plt.legend()
plt.show()
fig.savefig(figpath + savedir + 'hdmr.pdf')









