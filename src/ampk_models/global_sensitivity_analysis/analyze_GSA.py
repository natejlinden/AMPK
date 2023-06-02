import numpy as np
from SALib.sample import sobol as sobol_samp
from SALib.sample import morris as morris_samp
from SALib.analyze import sobol as sobol_analyze
from SALib.analyze import morris as morris_analyze
from SALib.analyze.hdmr import analyze as hdmr_analyze
import os
import sys

import jax  
import jax.numpy as jnp
import jax.scipy as jsp
import scipy.optimize as opt
from jax import lax

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from scipy.stats import t

jax.config.update('jax_enable_x64', True)

plt.style.use('~/.matplotlib/custom.mplstyle')
mpl.rcParams['figure.autolayout'] = True

# we will peform the analysis over all models
model_names_nominals = [('MA_double_mech', 'nominal_params_MA.csv'), 
                        ('newmech_MA_single', 'nominal_params_newmech_MA.csv')]
lb_mult, ub_mult = 0.1, 10.0

for model_info in model_names_nominals:
    model, nominals_file = model_info
    savedir = model + '/'

    ####### PREPROCESSING #######
    # load parameter samples and simulation results
    param_vals_sobol_corr = np.load('./' + savedir + 'param_vals_sobol_MA_corr.npy')
    sobol_sols_corr = np.load('./' + savedir + 'sols_sobol_corr.npy')

    # reshape into 2D array
    sobol_sols_corr = sobol_sols_corr.reshape(sobol_sols_corr.shape[0]*sobol_sols_corr.shape[1], sobol_sols_corr.shape[2])
    nsols, nqoi = sobol_sols_corr.shape

    # load nominal vals and compute bounds
    nominal_df = pd.read_csv(nominals_file)
    nominal_vals = nominal_df['value'].to_list()
    param_names = nominal_df['parameter'].to_list()
    n_params = len(param_names)
    bounds = [[lb_mult*param, ub_mult*param] for param in nominal_vals]

    # check if bounds match min/max in sample and update accordingly
    # if the difference is greater the 10% of the original bounds keep the min/max
    # otherwise keep the bounds
    thresh = 0.1
    min_max = np.vstack((np.min(param_vals_sobol_corr, axis=0),
                         np.max(param_vals_sobol_corr, axis=0))).transpose()
    condition = np.abs(bounds - min_max)/bounds > thresh
    bounds = np.where(condition, min_max, bounds)

    problem = {'num_vars':n_params, 'names':param_names, 'bounds': bounds,} # dict for SALib

    ####### HISTOGRAM OF QOI #######
    fig, ax = plt.subplots(1,1, figsize=(2.25,3))
    ax.hist(sobol_sols_corr[:,1], bins=20)
    ax.set_xlabel('change \n pAMPKAR/AMPKARtot', labelpad=5)
    ax.set_ylabel('count')
    fig.savefig('../../../figures/' + savedir + 'change_hist.pdf')
    plt.show()

    ####### COMPUTE SENSITIVITIY INDICES #######
    Si_sobol = sobol_analyze.analyze(problem, sobol_sols_corr[:,1], calc_second_order=True)
    np.save('./' + savedir + 'sobol_change.npy', Si_sobol)
    Si_hdmr = hdmr_analyze(problem, param_vals_sobol_corr, sobol_sols_corr[:,1])
    np.save('./' + savedir + 'hdmr_change.npy', Si_hdmr)

    ####### SORT #######
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

    ####### PLOT #######
    # sobol first order
    fig, ax = plt.subplots(figsize=(4,3))
    ax.bar(np.arange(0,len(param_names)), Si_sobol_sorted['S1'],
            yerr=Si_sobol_sorted['S1_conf'],
            log=False)
    plt.xticks(np.arange(0,len(param_names)), Si_sobol_sorted['name'], rotation='vertical')
    plt.ylabel('Sobol First Order')
    fig.savefig('../../../figures/' + savedir + 'S1_change.pdf')
    plt.show()

    # sobol total order
    fig, ax = plt.subplots(figsize=(4,3))
    ax.bar(np.arange(0,len(param_names)), Si_sobol_sorted['ST'],
            yerr=Si_sobol_sorted['ST_conf'],
            log=False)
    plt.xticks(np.arange(0,len(param_names)), Si_sobol_sorted['name'], rotation='vertical')
    plt.ylabel('Sobol Total Order')
    fig.savefig('../../../figures/' + savedir + 'ST_change.pdf')
    plt.show()

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
    fig.savefig('../../../figures/' + savedir + 'hdmr_change.pdf')









