from os import environ
environ['OMP_NUM_THREADS'] = '1'

import numpy as np
from SALib.sample import sobol as sobol_samp
from SALib.sample import morris as morris_samp
from SALib.analyze import sobol as sobol_analyze
from SALib.analyze import morris as morris_analyze
from SALib.analyze.hdmr import analyze as hdmr_analyze
from tqdm import tqdm
import os
import sys
import importlib
import multiprocessing as mp
import time
import math
import json
import pandas as pd
from gsa_utils import * # GSA utility functions

############################################
# get user inputs
############################################
if len(sys.argv) < 9:
    print("Incorrect Usage: TODO: add usage") # TODO: add usage
    sys.exit(1)

cpu_mult = float(sys.argv[1])
dir = sys.argv[2]
base_name = sys.argv[3]
diffrax_model = sys.argv[4]
model_info_json = sys.argv[5]
nominals_file = sys.argv[6]
bounds_file = sys.argv[7]
param_comp_function = sys.argv[8]

############################################
# jax loading
############################################
# use all available cores (do before loading jax)
# required for parallel cpu runs
n_cores = mp.cpu_count()
print('Using {} cores'.format(n_cores))
n_devices = int(2**np.ceil(math.log(cpu_mult*n_cores, 2))) # sets n_devices to the next largest power of 2
print('Set {} XLA devices'.format(n_devices))

xla_flag = '--xla_force_host_platform_device_count={}'.format(n_devices)
environ['XLA_FLAGS']=xla_flag

import jax
import jax.numpy as jnp
from jax import lax
import equinox as eqx
import diffrax as dfrx

jax.config.update('jax_enable_x64', True)
jax.config.update('jax_platform_name', 'cpu')
print('Using', len(jax.devices()), "'devices'")
############################################
# load diffrax model
############################################
sys.path.insert(0, '../odes')
try:
    model = importlib.import_module(diffrax_model)
except ImportError:
    print("Module '{}' not found.".format(diffrax_model))
    sys.exit(1)

############################################
# Setup output directory #
############################################
try:
    if os.path.exists(dir):
        print('Saving to: ', dir)
    else:
        print(dir,'does not exist.')
except:
    print("An error occurred while checking the path.")

fname = base_name + '/' # TODO: add check to make sure base_name is a string
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

# metabolism_params
metab_parms_basal = {'kGly': 0.5,'kHydro':0.1,
                     'VforAK': 14.66, 'KeqAK': 2.21, 'kmm': 0.32, 'kmd': 0.35, 'kmt': 0.27,
                     'VmaxOxPhos':0.5,'Kadp': 5.8e-2,'n': 2.568,}
metab_parms_stress = {'kGly': 0.005,'kHydro':0.1,
                      'VforAK': 14.66, 'KeqAK': 2.21, 'kmm': 0.32, 'kmd': 0.35, 'kmt': 0.27,
                     'VmaxOxPhos':0.5,'Kadp': 5.8e-2,'n': 2.568,}

# dictionary of the problem for SALib
bounds = {'num_vars':nparam, 'names':param_names, 'bounds': bounds,}

############################################
# states and initial conditions
############################################
try:
    with open(model_info_json, 'r') as file:
        model_data = json.load(file)
        print("JSON file loaded successfully.")
        print("Data:", model_data)
except FileNotFoundError:
    print("File not found.")
except json.JSONDecodeError:
    print("Invalid JSON format.")
except Exception as e:
    print("An error occurred while loading the JSON file:", str(e))

# unpack loaded model data dictionary
state_names = model_data['state_names']
ampkar_states = model_data['ampkar_states']
pampkar_states = model_data['pampkar_states']
n_states = len(state_names)
y0_states_to_set = model_data['y0']['set_states']
y0_vals_to_set = model_data['y0']['set_ics']

# get the indices of the states
ampkar_idxs = [state_names.index(item) for item in ampkar_states]
pampkar_idxs = [state_names.index(item) for item in pampkar_states]
ampkar_idx = state_names.index('AMPKAR')
pampkar_idx = state_names.index('pAMPKAR')


# Set initial conditions
y0 = np.zeros(n_states)
for state, val in zip(y0_states_to_set, y0_vals_to_set):
    y0[state_names.index(state)] = val

# random seed for reproducibility
seed = np.random.seed(seed=2048)

################################################
#                   Model RHS                  #
################################################
rhs = model.vector_field(**metab_parms_basal)
rhs_stress = model.vector_field(**metab_parms_stress)
rhs = dfrx.ODETerm(rhs)
rhs_stress = dfrx.ODETerm(rhs_stress)

################################################
# param comp function
################################################
try:
    module = importlib.import_module('gsa_utils')
    if hasattr(module, param_comp_function):
        compute_params = getattr(module, param_comp_function)
    else:
        print("Function '{}' not found in module '{}'.".format(param_comp_function, 'gsa_utils'))
except ImportError:
    print("'gsa_utils.{}' not found.".format(param_comp_function))
    sys.exit(1)

######################################################
# # generate samples using the Sobol sampling method #
######################################################
nsamps = 4096
param_vals_sobol = sobol_samp.sample(bounds, nsamps, calc_second_order=True, seed=seed)
np.save(savedir + 'param_vals_sobol.npy', np.array(param_vals_sobol))

# Convert from parameter samples to full parameter sets, because we do not sample
# all parameters in the model
n_params_true = len(compute_params(param_vals_sobol[0,:])) # get true length of parameter vector
temp = np.empty(shape=(param_vals_sobol.shape[0], n_params_true))

for i in range(param_vals_sobol.shape[0]):
    temp[i,:] = np.array(compute_params(param_vals_sobol[i,:]))

param_vals_sobol = temp

######################################################
# Run simulations
######################################################
print('Reshaping input parameters...')
qoi_fn_pmap = jax.pmap(single_model_eval_nansafe, in_axes=(0,None,None,None,None,None,None))

params_shape = param_vals_sobol.shape # get shape of parameter vectors (n_sampls, n_params)
# the parameter vector needs to be shape (n_loops, n_devices, n_params)
# n_loops needs to be the integer which is larger than n_sampls//n_devices
# thus we need to pad any extra entries added with nans
# for example if we have 120 parameter set, then we need to add 8 row of nans
pad = int((np.ceil(params_shape[0]/n_devices)*n_devices)-params_shape[0]) #params_shape[0] % n_devices
if pad:
    padt = np.empty((pad, params_shape[1]))
    padt[:] = np.nan
    param_vals_sobol = np.vstack((param_vals_sobol, padt))

# now we can reshape the parameter vector accordingly
n_loops = int(np.ceil(params_shape[0]/n_devices))
new_params = jnp.array(param_vals_sobol).reshape((n_loops,n_devices,params_shape[1]))

# we are now ready to run simulations
print('Running simulations...')
sols_sobol =[]
tnow = time.time()
for i in tqdm(range(n_loops)):
    sol = qoi_fn_pmap(new_params[i,:,:], rhs, rhs_stress, y0, ampkar_idx, ampkar_idxs, pampkar_idxs)
    sols_sobol.append(sol)
    # print('loop', i, 'of', n_loops, 'complete')
tend = time.time()

print('Simulations took {} seconds'.format(tend-tnow))
print('Saving results...')
np.save(savedir + 'sols_sobol.npy', jnp.array(sols_sobol))

print('Complete!')
quit()