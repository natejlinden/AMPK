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
qoi_names = [('normalized change \n pAMPKAR/AMPKAR', 'norm_change'), 
             ('change \n pAMPKAR/AMPKAR', 'change')]

for model_info in model_names_nominals:
    model, nominals_file = model_info
    savedir = './' + model + '/'

    ####### PREPROCESSING #######
    # load parameter samples and simulation results
    param_vals_sobol_corr = np.load(savedir + 'param_vals_sobol_MA_corr.npy')
    sobol_sols_corr = np.load(savedir + 'sols_sobol_corr.npy')

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
    print(min_max)
    condition = np.abs(bounds - min_max)/bounds > thresh
    bounds = np.where(condition, min_max, bounds)

    problem = {'num_vars':n_params, 'names':param_names, 'bounds': bounds,} # dict for SALib

    ####### HISTOGRAM OF QOI #######
    for i in range(len(qoi_names)):
        fig, ax = plt.subplots(1,1, figsize=(2.25,3))
        ax.hist(sobol_sols_corr[:,0], bins=20)
        ax.set_xlabel(qoi_names[i][0], labelpad=2)
        ax.set_ylabel('count')
        fig.savefig(savedir + qoi_names[i][1] + '_hist.pdf')
        plt.show()









