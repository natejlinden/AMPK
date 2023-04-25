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
import ampk_MM_single_mech_diffrax as model
from gsa_utils import *

jax.config.update('jax_enable_x64', True)
jax.config.update('jax_platform_name', 'cpu')
print(len(jax.devices()))
############################################
# Setup output directory #
############################################
print('Saving to: ', dir)

fname = 'MM_single_mech/'
savedir = dir+fname
if not os.path.exists(savedir):
    os.makedirs(savedir)
    print('Created directory: ', savedir)

############################################
# Bounds and info for params #
############################################

# define the bounds for the AMPK parameters
# we use plus or minus on order of magnitude of any known values and then make reasonable assumptions for unknowns
# Note we fix all off rates to 1.0 and dont bother sampling these or computing sensitivities
############################################
# initial conditions
state_names = ['AMP', 'ADP', 'ATP', 'AMPK', 'pAMPK', 'AMP_AMPK', 'ADP_AMPK', 'ATP_AMPK', 'AMP_pAMPK', 'ADP_pAMPK', 'ATP_pAMPK', 'AMPKAR', 'pAMPKAR']

# initial conditions
to_set = ['AMP', 'ADP', 'ATP', 'AMPK', 'AMPKAR']
idxs = [state_names.index(item) for item in to_set]

print(idxs)

y0 = np.zeros((13,))
y0[idxs[0]] = 2e-5   # 'AMP' mM
y0[idxs[1]] = 1.3e-1 # 'ADP mM
y0[idxs[2]] = 8.2   # 'ATP mM
y0[idxs[3]] = 0.6   # 'AMPK mM
y0[idxs[4]] = 1.0   # 'AMPKAR mM

y0 = np.array(y0)

# function to compute all MA params from sampled params
def compute_MM_params(params):
    return (
        1.0, # kOnAMP
        params[0], # kOffAMP
        1.0 , # kOnADP
        params[1], # kOffADP
        1.0, # kOnATP
        params[2], # kOffATP
        params[3], # kCaMKK
        params[4],  # KmCaMKK
        params[5], # kLKB1
        params[6],  # KmLKB1
        params[7], # kPP
        params[8], # KmPP
        params[9], # kAMPK
        params[10], # KmAMPK
        params[11], # kPP1
        params[12], # KmPP1
        1.0, # CaMKKtot
        1.0, # LKB1tot
        1.0, # PPtot
        1.0, # PP1tot
    )

# parameters
# metabolism_params
metab_parms_basal = {'kGly': 1300.0,'kHydro':1.4e-3,'kForAK':40.44,
                     'kRevAK':1.1e-3,'VmaxOxPhos':0.5,'Kadp': 5.8e-2,'n': 2.568,}
metab_parms_stress = {'kGly': 4.5,'kHydro':1.4e-3,'kForAK':40.44,
                      'kRevAK':1.1e-3,'VmaxOxPhos':0.5,'Kadp': 5.8e-2,'n': 2.568,}

nominal_vals_MM = [
   2.5e-3, # KdAMP
   1.5e-3, # KdADP
   1.7e-3, # KdATP
   0.357, # kPhosCaMKK
   1.5e-2,  # KmCaMKK
   3.92e-2, # kPhosLKB1
   1.4,  # KmLKB1
   1.1e-1, # kDephosPP
   6.7e-2, # KmPP
   6.33, # kPhosAMPK
   4.67e-3, # KmAMPK
   1.1e-1, # kDephosPP1
   6.7e-2, # KmPP1
   1e-3, # AMPKAR
]

lb_mult = 0.5
ub_mult = 1.5
bounds_MM = [[lb_mult*param, ub_mult*param] for param in nominal_vals_MM]
param_names_MM = ['KdAMP', 'KdADP', 'KdATP', 'kPhosCaMKK', 'KmCamKK', 'kPhosLKB1', 'KmLKB1', 'kDephosPP', 'KmPP', 'kPhosAMPK', 'KmAMPK', 
               'kDephosPP1', 'KmPP1', 'AMPKAR']

# dictionary of the problem for SALib
bounds_MM = {'num_vars':14, 'names':param_names_MM, 'bounds': bounds_MM,}

# random seed for reproducibility
seed = np.random.seed(seed=2048)

#####################################################################
# SET UP Jitable functions to solve to steady-state and compute qoi #
#####################################################################
## Now test a solve to steady-state function
# jitable function to run the system to steady-state
@jax.jit
def solve_to_steady_state(params, rhs, y0, thresh=1e-10):
    solver=dfrx.Kvaerno5()
    stepsize_controller = dfrx.PIDController(rtol=1e-8, atol=1e-8)
    t0 = 0.0
    t1 = 1000.0 # 1000 seconds
    times = np.arange(t0, t1, t1/100)
    dt0 = 1e-6 # initial time step
    saveat=dfrx.SaveAt(ts=times)
    times_repeat = np.arange(t0, 100, 100/100)
    saveat_repeat=dfrx.SaveAt(ts=times_repeat)

    # initial solve
    sol = dfrx.diffeqsolve(
        rhs, 
        solver, 
        t0, t1, dt0, 
        y0, 
        saveat=saveat, stepsize_controller=stepsize_controller,
        args=params)

    # solve to steady-state
    # cond_fun checks for steady-state by checking the norm of the RHS
    cond_fun = lambda sol: jnp.linalg.norm(rhs.vector_field(0.0, sol[0].ys[-1,:], params)) > thresh
    # body_fun solves the ODEs for 100 more seconds
    body_fun = lambda sol: (dfrx.diffeqsolve(rhs, solver, t0, 100, dt0, sol[0].ys[-1,:], 
        saveat=saveat_repeat, stepsize_controller=stepsize_controller, args=params),
        sol[1]+100)
    
    # while loop until steady state condition is reached
    sol_final = lax.while_loop(cond_fun, body_fun, (sol, 1000))
    
    return sol_final # returns the final state

# jitable function to compute the qois
@jax.jit
def single_model_eval(params, rhs_basal, rhs_stress, y0, ampkar_idx=11, pampkar_idx=12):
    # update y0 for AMPKAR
    y0 = y0.at[ampkar_idx].set(params[-1])

    # find basal steady-state
    sol_basal = solve_to_steady_state(params, rhs_basal, y0, 1e-12)

    # apply stimulus and run again
    sol_stress = solve_to_steady_state(params, rhs_stress, sol_basal[0].ys[-1,:], 1e-10)

    # compute the ratio of pAMPKAR/AMPKAR at the end of the stress simulation
    basal_ratio = sol_basal[0].ys[-1,pampkar_idx]/sol_basal[0].ys[-1,ampkar_idx]
    stress_ratio = sol_stress[0].ys[-1,pampkar_idx]/sol_stress[0].ys[-1,ampkar_idx]
    norm_change = (stress_ratio - basal_ratio) / basal_ratio
    
    return jnp.array([norm_change, basal_ratio, stress_ratio, sol_basal[1], sol_stress[1]])

@jax.jit
def single_model_eval_nansafe(params, rhs_basal, rhs_stress, y0):
    pred = jnp.sum(jnp.isnan(params))
    false_fun = lambda params: single_model_eval(params, rhs_basal, rhs_stress, y0)
    true_fun = lambda params: jnp.array([jnp.nan, jnp.nan, jnp.nan, jnp.nan, jnp.nan])
    return lax.cond(pred, true_fun, false_fun, params)

################################################
#                   Model RHS                  #
################################################
rhs = model.ampk_MM_single_mech(**metab_parms_basal)
rhs_stress = model.ampk_MM_single_mech(**metab_parms_stress)
rhs = dfrx.ODETerm(rhs)
rhs_stress = dfrx.ODETerm(rhs_stress)

################################################
# Full scale case with large number of samples #
################################################
# generate samples using the Sobol sampling method
nsamps = 1024
param_vals_sobol_MM = sobol_samp.sample(bounds_MM, nsamps, calc_second_order=True, seed=seed)

param_vals_sobol_MM_np = np.array(param_vals_sobol_MM)

np.save(savedir + 'param_vals_sobol_MM_corr.npy', param_vals_sobol_MM_np)

# Run simulations
print('Reshaping input parameters...')
# qoi_fn = lambda params: single_model_eval(params, rhs, rhs_stress, y0)
# qoi_fn_vmap = jax.vmap(single_model_eval, in_axes=(0,None,None,None))
qoi_fn_pmap = jax.pmap(single_model_eval_nansafe, in_axes=(0,None,None,None))

params_shape = param_vals_sobol_MM_np.shape # get shape of parameter vectors (n_sampls, n_params)
# the parameter vector needs to be shape (n_loops, n_devices, n_params)
# n_loops needs to be the integer which is larger than n_sampls//n_devices
# thus we need to pad any extra entries added with nans
# for example if we have 120 parameter set, then we need to add 8 row of nans
pad = int((np.ceil(params_shape[0]/n_devices)*n_devices)-params_shape[0]) #params_shape[0] % n_devices
if pad:
    pad_mat = np.empty((pad, params_shape[1]))
    pad_mat[:] = np.nan
    param_vals_sobol_MM_np = np.vstack((param_vals_sobol_MM_np, pad_mat))

# now we can reshape the parameter vector accordingly
n_loops = int(np.ceil(params_shape[0]/n_devices))
new_params = jnp.array(param_vals_sobol_MM_np).reshape((n_loops,n_devices,params_shape[1]))

# we are now ready to run simulations
print('Running simulations...')
sols_sobol_MM =[]
tnow = time.time()
for i in range(n_loops):
    sol = qoi_fn_pmap(new_params[i,:,:], rhs, rhs_stress, y0)
    sols_sobol_MM.append(sol)
tend = time.time()

print('Simulations took {} seconds'.format(tend-tnow))
print('Saving results...')
np.save(savedir + 'sols_sobol_corr.npy', jnp.array(sols_sobol_MM))

print('Complete!')
quit()