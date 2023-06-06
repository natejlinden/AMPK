import numpy as np
import os
import sys
import json
import importlib

import jax  
from jax import lax
import equinox as eqx
import diffrax as dfrx

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from gsa_utils import *

############################################
# Jax settings
############################################
jax.config.update('jax_enable_x64', True)
sys.path.insert(0, '../odes')

############################################
# Matplotlib settings
############################################
plt.style.use('~/.matplotlib/custom.mplstyle')
mpl.rcParams['figure.autolayout'] = True

############################################
# get user inputs
############################################
if len(sys.argv) < 8:
    print("Incorrect Usage: TODO: add usage") # TODO: add usage
    sys.exit(1)

savedir             = sys.argv[1]
base_name           = sys.argv[2]
diffrax_model       = sys.argv[3]
model_info_json     = sys.argv[4]
param_comp_function = sys.argv[5]
n_traj              = int(sys.argv[6])
figdir              = sys.argv[7]
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

############################################
# metabolism_params
############################################
metab_parms_basal = {'kGly': 0.5,'kHydro':0.1, 'VforAK': 14.66,
                     'KeqAK': 2.21, 'kmm': 0.32, 'kmd': 0.35, 'kmt': 0.27,
                     'VmaxOxPhos':0.5,'Kadp': 5.8e-2,'n': 2.568,}
metab_parms_stress = {'kGly': 0.005,'kHydro':0.1, 'VforAK': 14.66,
                      'KeqAK': 2.21, 'kmm': 0.32, 'kmd': 0.35, 'kmt': 0.27,
                     'VmaxOxPhos':0.5,'Kadp': 5.8e-2,'n': 2.568,}

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


################################################
# Load parameter samples and qois
################################################
param_vals_sobol = np.load(savedir + base_name + 'param_vals_sobol.npy')
sobol_sols = np.load(savedir + base_name + 'sols_sobol.npy')

# reshape into 2D array
sobol_sols = sobol_sols.reshape(sobol_sols.shape[0]*sobol_sols.shape[1],
                                          sobol_sols.shape[2])
nsols, nqoi = sobol_sols.shape


################################################
# function to solve the model
################################################
@jax.jit
def solve_model(params, rhs, y0, t1, times):
    y0 = y0.at[ampkar_idx].set(params[-1])
    params = compute_params(params)
    solver=dfrx.Kvaerno5()
    stepsize_controller = dfrx.PIDController(rtol=1e-10, atol=1e-10)
    t0 = times[0]
    times = jnp.arange(t0, t1, 0.5)
    dt0 = 1e-8 # initial time step
    saveat=dfrx.SaveAt(ts=times)

    # solve
    sol = dfrx.diffeqsolve(
        rhs, 
        solver, 
        t0, t1, dt0, 
        y0, 
        saveat=saveat, stepsize_controller=stepsize_controller,
        args=params)
    
    return sol # returns the final state

# run funcrtion once to compile
times = jnp.arange(0.0, 1000.0, 0.5)
sol = solve_model(param_vals_sobol[0,:], rhs_stress, y0, 1000.0, times)


################################################
# function to solve the model
################################################
# draw some random samples from the parameter vector
idxs = np.random.choice(np.arange(len(param_vals_sobol)), size=(n_traj,), replace=False)

fig_init, ax_init = plt.subplots(1, 1, figsize=(4, 3)) # this is a plot for the full initialization + stress response
fig_stress, ax_stress = plt.subplots(1, 1, figsize=(4, 3)) # plot for only stress response
for i in range(n_traj):
    # get the sample
    sample = param_vals_sobol[idxs[i],:]
    sol = sobol_sols[idxs[i],:]
    # run the model to initial steady-state
    times = jnp.arange(0.0, sol[2], sol[2]/1000)
    sol_basal = solve_model(sample, rhs, y0, sol[2], times)
    # now actually run to get a solution
    # note sol[0] will be the ic and sol[1] will be the time to steady-state
    times = jnp.arange(0.0, sol[3], sol[3]/1000)
    sol_stress = solve_model(sample, rhs_stress, sol_basal.ys[-1,:], sol[3], times)

    # comput qois
    AMPKAR_tot_basal = np.sum(sol_basal.ys[:,ampkar_idxs], axis=1)
    pAMPKAR_basal = np.sum(sol_basal.ys[:,pampkar_idxs], axis=1)
    pAMPKAR_stress = np.sum(sol_stress.ys[:,pampkar_idxs], axis=1)
    basal = pAMPKAR_basal/AMPKAR_tot_basal
    stress = pAMPKAR_stress/AMPKAR_tot_basal
    qoi_stress = (pAMPKAR_stress/AMPKAR_tot_basal)-(pAMPKAR_basal[-1]/AMPKAR_tot_basal[-1])
    
    # plot the solution
    ax_init.plot(sol_basal.ts, basal, 'k', alpha=0.5)
    ax_init.plot(sol_basal.ts[-1]+sol_stress.ts, stress, 'b', alpha=0.5)
    ax_stress.plot(sol_stress.ts, qoi_stress, 'b', alpha=0.5)

# format and save plots
ax_init.plot([np.nan, np.nan], [np.nan, np.nan], 'k', alpha=0.5, label='basal')
ax_init.plot([np.nan, np.nan], [np.nan, np.nan], 'b', alpha=0.5, label='stress')
ax_init.legend()
ax_init.set_ylim([0, 1.1])
ax_init.set_xlabel('time (s)')
ax_init.set_ylabel(r'$pAMPKAR/AMPKAR_{tot}$')
fig_init.savefig(figdir + base_name + 'stress_responses_init.pdf', bbox_inches='tight')

ax_stress.set_ylim([0, 1.1])
ax_stress.set_xlabel('time (s)')
ax_stress.set_ylabel(r'$pAMPKAR/AMPKAR_{tot}$')
fig_stress.savefig(figdir + base_name + 'stress_responses_only.pdf', bbox_inches='tight')
