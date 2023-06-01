import numpy as np
from SALib.sample import sobol as sobol_samp
from SALib.sample import morris as morris_samp
from SALib.analyze import sobol as sobol_analyze
from SALib.analyze import morris as morris_analyze
from SALib.analyze.hdmr import analyze as hdmr_analyze
import os
import sys
import multiprocessing as mp
import time
import math
import pandas as pd

# get user inputs
dir = sys.argv[1]
cpu_mult = int(sys.argv[2])

# use all available cores (do before loading jax)
# required for parallel cpu runs
n_cores = mp.cpu_count()
print('Using {} cores'.format(n_cores))
n_devices = int(2**np.ceil(math.log(cpu_mult*n_cores, 2))) # sets n_devices to the next largest power of 2
print('Set {} XLA devices'.format(n_devices))

xla_flag = '--xla_force_host_platform_device_count={}'.format(n_devices)
os.environ['XLA_FLAGS']=xla_flag

import jax
import jax.numpy as jnp
from jax import lax
import equinox as eqx
import diffrax as dfrx

# load code
sys.path.insert(0, '../odes')
import ampk_MA_double_mech_diffrax as model
from gsa_utils import *

jax.config.update('jax_enable_x64', True)
jax.config.update('jax_platform_name', 'cpu')
print(len(jax.devices()))
############################################
# Setup output directory #
############################################
print('Saving to: ', dir)

fname = 'MA_double_mech_newQoI/'
savedir = dir+fname
if not os.path.exists(savedir):
    os.makedirs(savedir)
    print('Created directory: ', savedir)

############################################
# Bounds and other info for the GSA #
############################################
# define the bounds for the AMPK parameters
# we use plus or minus on order of magnitude of any known values and then make reasonable assumptions for unknowns
# Note we fix all off rates to 1.0 and dont bother sampling these or computing sensitivities
############################################
MA_nominals = pd.read_csv('nominal_params_MA.csv')
MM_nominals = pd.read_csv('nominal_params_MM.csv')
nominal_vals_MA = MA_nominals['value'].to_list()
param_names_MA = MA_nominals['parameter'].to_list()
nominal_vals_MM = MM_nominals['value'].to_list()
param_names_MM = MM_nominals['parameter'].to_list()
# bounds_MA_df = pd.read_csv('param_bounds_MA_tmp.csv')
# bounds_MM_df = pd.read_csv('param_bounds_MM_tmp.csv')

# metabolism_params
metab_parms_basal = {'kGly': 1300.0,'kHydro':1.4e-3,'kForAK':40.44,
                     'kRevAK':1.1e-3,'VmaxOxPhos':0.5,'Kadp': 5.8e-2,'n': 2.568,}
metab_parms_stress = {'kGly': 4.5,'kHydro':1.4e-3,'kForAK':40.44,
                      'kRevAK':1.1e-3,'VmaxOxPhos':0.5,'Kadp': 5.8e-2,'n': 2.568,}

# bounds and problem dicts for SALib
# bounds_MA = [bounds_MA_df['lower'].to_list(), bounds_MA_df['upper'].to_list()]
# bounds_MA = [[lb, ub] for lb, ub in zip(bounds_MA[0], bounds_MA[1])]
# bounds_MM = [bounds_MM_df['lower'].to_list(), bounds_MM_df['upper'].to_list()]
# bounds_MM = [[lb, ub] for lb, ub in zip(bounds_MM[0], bounds_MM[1])]
lb_mult = 0.5
ub_mult = 1.5
bounds_MA = [[lb_mult*param, ub_mult*param] for param in nominal_vals_MA]
bounds_MM = [[lb_mult*param, ub_mult*param] for param in nominal_vals_MM]

# dictionary of the problem for SALib
bounds_MA = {'num_vars':14, 'names':param_names_MA, 'bounds': bounds_MA,}
bounds_MM = {'num_vars':14, 'names':param_names_MM, 'bounds': bounds_MM,}


# states and initial conditions
state_names = ['AMP', 'ADP', 'ATP', 'AMPK', 'pAMPK', 'AMP_AMPK', 'ADP_AMPK', 'ATP_AMPK', 'AMP_pAMPK', 'ADP_pAMPK', 'ATP_pAMPK', 'AMP_AMP_AMPK', 'AMP_ADP_AMPK', 'AMP_ATP_AMPK', 'ADP_ADP_AMPK', 'ADP_ATP_AMPK', 'ATP_ATP_AMPK', 'AMP_AMP_pAMPK', 'AMP_ADP_pAMPK', 'AMP_ATP_pAMPK', 'ADP_ADP_pAMPK', 'ADP_ATP_pAMPK', 'ATP_ATP_pAMPK', 'CaMKK', 'CaMKK_AMPK', 'CaMKK_AMP_AMPK', 'CaMKK_ADP_AMPK', 'CaMKK_ATP_AMPK', 'CaMKK_AMP_AMP_AMPK', 'CaMKK_AMP_ADP_AMPK', 'CaMKK_AMP_ATP_AMPK', 'CaMKK_ADP_ADP_AMPK', 'CaMKK_ADP_ATP_AMPK', 'CaMKK_ATP_ATP_AMPK', 'LKB1', 'LKB1_AMP_AMPK', 'LKB1_ADP_AMPK', 'LKB1_AMP_AMP_AMPK', 'LKB1_AMP_ADP_AMPK', 'LKB1_ADP_ADP_AMPK', 'PP', 'PP_pAMPK', 'PP_ATP_pAMPK', 'PP_AMP_ATP_pAMPK', 'PP_ADP_ATP_pAMPK', 'PP_ATP_ATP_pAMPK', 'AMPKAR', 'pAMPKAR', 'AMPKAR_AMP_pAMPK', 'AMPKAR_AMP_AMP_pAMPK', 'AMPKAR_AMP_ADP_pAMPK', 'PP1', 'PP1_pAMPKAR']

ampkar_idx = state_names.index('AMPKAR')
pampkar_idx = state_names.index('pAMPKAR')


# Set initial conditions
to_set = ['AMP', 'ADP', 'ATP', 'AMPK', 'CaMKK', 'LKB1', 'PP', 'AMPKAR', 'PP1']
idxs = [state_names.index(item) for item in to_set]
y0 = np.zeros((53,))
y0[idxs[0]] = 2e-5   # AMP
y0[idxs[1]] = 1.3e-1 # ADP
y0[idxs[2]] = 8.2   # ATP
y0[idxs[3]] = 0.6   # AMPK
y0[idxs[4]] = 1.0   # CaMKK
y0[idxs[5]] = 1.0   # LKB1
y0[idxs[6]] = 1.0   # PP
y0[idxs[7]] = 0.1   # AMPKAR
y0[idxs[8]] = 1.0   # PP1

# random seed for reproducibility
seed = np.random.seed(seed=2048)

################################################
#                   Model RHS                  #
################################################
rhs = model.ampk_MA_double_mech(**metab_parms_basal)
rhs_stress = model.ampk_MA_double_mech(**metab_parms_stress)
rhs = dfrx.ODETerm(rhs)
rhs_stress = dfrx.ODETerm(rhs_stress)

################################################
# Full scale case with large number of samples #
################################################
# generate samples using the Sobol sampling method
nsamps = 1024
param_vals_sobol_MA = sobol_samp.sample(bounds_MA, nsamps, calc_second_order=True, seed=seed)
param_vals_sobol_MM = sobol_samp.sample(bounds_MM, nsamps, calc_second_order=True, seed=seed)

# copy MA to np arrays
param_vals_sobol_MA_corr = np.array(param_vals_sobol_MA)
param_vals_sobol_MM_np = np.array(param_vals_sobol_MM)

# now compute all kons using samples of Km and from the MM model and Kcat from MA model
to_set = ['kPhosCaMKK', 'kPhosLKB1','kDephosPP','kPhosAMPK', 'kDephosPP1']
kcat_idxs_MA = [param_names_MA.index(item) for item in to_set]
to_set = ['kOnCaMKK', 'kOnLKB1','kOnPP','kOnAMPK','kOnPP1']
kon_idxs_MA = [param_names_MA.index(item) for item in to_set]
to_set = ['KmCaMKK','KmLKB1', 'KmPP', 'KmAMPK', 'KmPP1']
km_idxs_MM = [param_names_MM.index(item) for item in to_set]

for kcat_i, kon_i, km_i in zip(kcat_idxs_MA, kon_idxs_MA, km_idxs_MM):
    _, param_vals_sobol_MA_corr[:,kon_i] =  michaelis_menten_to_mass_action(None, 
                                            param_vals_sobol_MM_np[:,km_i], 
                                            None, k_rev=1.0, 
                                            k_cat=+param_vals_sobol_MA_corr[:,kcat_i])
    
# save parameter samples
np.save(savedir + 'param_vals_sobol_MA_corr.npy', np.array(param_vals_sobol_MA_corr))

# Convert from parameter samples to full parameter sets, because we do not sample
# all parameters in the model
n_params_true = 22
temp = np.empty(shape=(param_vals_sobol_MA_corr.shape[0], n_params_true))

for i in range(param_vals_sobol_MA_corr.shape[0]):
    temp[i,:] = np.array(compute_MA_params(param_vals_sobol_MA_corr[i,:]))

param_vals_sobol_MA_corr = temp

# Run simulations
print('Reshaping input parameters...')
# qoi_fn = lambda params: single_model_eval(params, rhs, rhs_stress, y0)
# qoi_fn_vmap = jax.vmap(single_model_eval, in_axes=(0,None,None,None))
qoi_fn_pmap = jax.pmap(single_model_eval_nansafe, in_axes=(0,None,None,None,None,None))

params_shape = param_vals_sobol_MA_corr.shape # get shape of parameter vectors (n_sampls, n_params)
# the parameter vector needs to be shape (n_loops, n_devices, n_params)
# n_loops needs to be the integer which is larger than n_sampls//n_devices
# thus we need to pad any extra entries added with nans
# for example if we have 120 parameter set, then we need to add 8 row of nans
pad = int((np.ceil(params_shape[0]/n_devices)*n_devices)-params_shape[0]) #params_shape[0] % n_devices
if pad:
    pad_mat = np.empty((pad, params_shape[1]))
    pad_mat[:] = np.nan
    param_vals_sobol_MA_corr = np.vstack((param_vals_sobol_MA_corr, pad_mat))

# now we can reshape the parameter vector accordingly
n_loops = int(np.ceil(params_shape[0]/n_devices))
new_params = jnp.array(param_vals_sobol_MA_corr).reshape((n_loops,n_devices,params_shape[1]))

# we are now ready to run simulations
print('Running simulations...')
sols_sobol_MA =[]
tnow = time.time()
for i in range(n_loops):
    sol = qoi_fn_pmap(new_params[i,:,:], rhs, rhs_stress, y0, ampkar_idx, pampkar_idx)
    sols_sobol_MA.append(sol)
    print('loop', i, 'of', n_loops, 'complete')
tend = time.time()

print('Simulations took {} seconds'.format(tend-tnow))
print('Saving results...')
np.save(savedir + 'sols_sobol_corr.npy', jnp.array(sols_sobol_MA))

print('Complete!')
quit()