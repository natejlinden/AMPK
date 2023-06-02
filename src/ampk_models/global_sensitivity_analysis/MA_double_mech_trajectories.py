import numpy as np
import os
import sys

import jax  
from jax import lax
import equinox as eqx
import diffrax as dfrx

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

jax.config.update('jax_enable_x64', True)
sys.path.insert(0, '../odes')
import ampk_MA_double_mech_diffrax as model
import ampk_MA_double_mech as model1
from gsa_utils import *

plt.style.use('~/.matplotlib/custom.mplstyle')
mpl.rcParams['figure.autolayout'] = True

savedir = 'MA_double_mech/'
nominals_file = 'nominal_params_MA.csv'

param_vals_sobol_corr = np.load('./' + savedir + 'param_vals_sobol_MA_corr.npy')
sobol_sols_corr = np.load('./' + savedir + 'sols_sobol_corr.npy')

# reshape into 2D array
sobol_sols_corr = sobol_sols_corr.reshape(sobol_sols_corr.shape[0]*sobol_sols_corr.shape[1], sobol_sols_corr.shape[2])
nsols, nqoi = sobol_sols_corr.shape

###### MODEL SETUP #######
state_names = ['AMP', 'ADP', 'ATP', 'AMPK', 'pAMPK', 'AMP_AMPK', 
               'ADP_AMPK', 'ATP_AMPK', 'AMP_pAMPK', 'ADP_pAMPK', 
               'ATP_pAMPK', 'AMP_AMP_AMPK', 'AMP_ADP_AMPK', 'AMP_ATP_AMPK', 
               'ADP_ADP_AMPK', 'ADP_ATP_AMPK', 'ATP_ATP_AMPK', 'AMP_AMP_pAMPK', 
               'AMP_ADP_pAMPK', 'AMP_ATP_pAMPK', 'ADP_ADP_pAMPK', 'ADP_ATP_pAMPK', 
               'ATP_ATP_pAMPK', 'CaMKK', 'CaMKK_AMPK', 'CaMKK_AMP_AMPK', 'CaMKK_ADP_AMPK', 
               'CaMKK_ATP_AMPK', 'CaMKK_AMP_AMP_AMPK', 'CaMKK_AMP_ADP_AMPK', 
               'CaMKK_AMP_ATP_AMPK', 'CaMKK_ADP_ADP_AMPK', 'CaMKK_ADP_ATP_AMPK', 
               'CaMKK_ATP_ATP_AMPK', 'LKB1', 'LKB1_AMP_AMPK', 'LKB1_ADP_AMPK', 
               'LKB1_AMP_AMP_AMPK', 'LKB1_AMP_ADP_AMPK', 'LKB1_ADP_ADP_AMPK', 'PP', 
               'PP_pAMPK', 'PP_ATP_pAMPK', 'PP_AMP_ATP_pAMPK', 'PP_ADP_ATP_pAMPK', 
               'PP_ATP_ATP_pAMPK', 'AMPKAR', 'pAMPKAR', 'AMPKAR_AMP_pAMPK', 
               'AMPKAR_AMP_AMP_pAMPK', 'AMPKAR_AMP_ADP_pAMPK', 'PP1', 'PP1_pAMPKAR']
ampkar_idx = state_names.index('AMPKAR')
# initial conditions
to_set = ['AMP', 'ADP', 'ATP', 'AMPK', 'CaMKK', 'LKB1', 'PP', 'AMPKAR', 'PP1']
idxs = [state_names.index(item) for item in to_set]

# print(idxs)
y0 = np.zeros((len(state_names),))
y0[idxs[0]] = 2e-5   # 'AMP' mM
y0[idxs[1]] = 1.3e-1 # 'ADP mM
y0[idxs[2]] = 8.2   # 'ATP mM
y0[idxs[3]] = 0.6   # 'AMPK mM
y0[idxs[4]] = 10.0   # CaMKK mM
y0[idxs[5]] = 10.0   # LKB1 mM
y0[idxs[6]] = 10.0   # PP mM
y0[idxs[7]] = 0.1   # 'AMPKAR mM
y0[idxs[8]] = 10.0   # PP1 mM

y0 = np.array(y0)


# metabolism_params
metab_parms_basal = {'kGly': 1300.0,'kHydro':1.4e-3,'kForAK':40.44,
                     'kRevAK':1.1e-3,'VmaxOxPhos':0.5,'Kadp': 5.8e-2,'n': 2.568,}
metab_parms_stress = {'kGly': 4.5,'kHydro':1.4e-3,'kForAK':40.44,
                      'kRevAK':1.1e-3,'VmaxOxPhos':0.5,'Kadp': 5.8e-2,'n': 2.568,}

# load nominal vals and compute bounds
nominal_df = pd.read_csv(nominals_file)
nominal_vals = nominal_df['value'].to_list()

## Now test a solve to steady-state function
@jax.jit
def solve_model(params, rhs, y0, t1, times):
    y0 = y0.at[ampkar_idx].set(params[-1])
    params = compute_MA_params(params)
    solver=dfrx.Kvaerno5()
    stepsize_controller = dfrx.PIDController(rtol=1e-10, atol=1e-10)
    t0 = 0.0
    # times = jnp.arange(t0, t1, 0.5)
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


# Model RHS #
rhs = model.ampk_MA_double_mech(**metab_parms_basal)
rhs_stress = model.ampk_MA_double_mech(**metab_parms_stress)
rhs = dfrx.ODETerm(rhs)
rhs_stress = dfrx.ODETerm(rhs_stress)

# run once to compile
times = jnp.arange(0.0, 1000.0, 0.5)
sol = solve_model(nominal_vals, rhs_stress, y0, 1000.0, times)
solve_model(nominal_vals, rhs_stress, y0, 1000.0, times)

###### SOLVE AND PLOT #######
# draw some random samples from the parameter vector
n_samples = 100
idxs = np.random.choice(np.arange(len(param_vals_sobol_corr)), size=(n_samples,), replace=False)

fig, ax = plt.subplots(1, 1, figsize=(4, 3))
fig1, ax1 = plt.subplots(1, 1, figsize=(4, 3))
for i in range(n_samples):
    # get the sample
    sample = param_vals_sobol_corr[idxs[i],:]
    sol = sobol_sols_corr[idxs[i],:]
    # run the model to initial steady-state
    times = jnp.arange(0.0, sol[1], sol[1]/1000)
    sol_1 = solve_model(sample, rhs, y0, sol[1], times)
    # now actually run to get a solution
    # note sol[0] will be the ic and sol[1] will be the time to steady-state
    times = jnp.arange(0.0, sol[2], sol[2]/1000)
    sol_2 = solve_model(sample, rhs_stress, sol_1.ys[-1,:], sol[2], times)
    # plot the solution
    ax.plot(sol_1.ts, sol_1.ys[:,ampkar_idx+1]/sample[-1], 'k', label=f'sample {idxs[i]}', alpha=0.5)
    ax.plot(sol_1.ts[-1]+sol_2.ts, sol_2.ys[:,ampkar_idx+1]/sample[-1], 'b', label=f'sample {idxs[i]}', alpha=0.5)
    ax1.plot(sol_2.ts, sol_2.ys[:,47]/sample[-1], 'b', label=f'sample {idxs[i]}', alpha=0.5)

ax.set_yscale('log')
ax.set_xlabel('time (s)')
ax.set_ylabel('pAMPKAR/AMPKARtot')
fig.savefig('../../../figures/' + savedir + 'stress_responses_init.pdf', bbox_inches='tight')

# ax1.set_yscale('log')
ax1.set_xlabel('time (s)')
ax1.set_ylabel('pAMPKAR/AMPKARtot')
fig1.savefig('../../../figures/' + savedir + 'stress_responses_only.pdf', bbox_inches='tight')