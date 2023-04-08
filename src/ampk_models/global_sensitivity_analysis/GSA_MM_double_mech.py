import sunode
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import sympy as sym
from SALib.sample import sobol as sobol_samp
from SALib.sample import morris as morris_samp
from SALib.analyze import sobol as sobol_analyze
from SALib.analyze import morris as morris_analyze
from SALib.analyze.hdmr import analyze as hdmr_analyze
import sys
import time
import multiprocessing
from tqdm import tqdm
from joblib import Parallel, delayed
sys.path.insert(0, '../odes')

import ampk_MM_double_mech as model1
from gsa_utils import *
#%matplotlib inline
plt.style.use('~/.matplotlib/custom.mplstyle')
mpl.rcParams['figure.autolayout'] = True

############################################
# Bounds and info for params #
############################################
# Michaelis-Menten parameters
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

bounds_MM_90 = [[0.1*param, 1.9*param] for param in nominal_vals_MM]

param_names_MM = ['KdAMP', 'KdADP', 'KdATP', 'kPhosCaMKK', 'KmCamKK', 'kPhosLKB1', 'KmLKB1', 'kDephosPP', 'KmPP', 'kPhosAMPK', 'KmAMPK', 
               'kDephosPP1', 'KmPP1', 'AMPKAR']
state_names = ['AMP', 'ADP', 'ATP', 'AMPK', 'pAMPK', 'AMP_AMPK', 'ADP_AMPK', 'ATP_AMPK', 'AMP_pAMPK', 'ADP_pAMPK', 'ATP_pAMPK', 'AMP_AMP_AMPK', 'AMP_ADP_AMPK', 'AMP_ATP_AMPK', 'ADP_ADP_AMPK', 'ADP_ATP_AMPK', 'ATP_ATP_AMPK', 'AMP_AMP_pAMPK', 'AMP_ADP_pAMPK', 'AMP_ATP_pAMPK', 'ADP_ADP_pAMPK', 'ADP_ATP_pAMPK', 'ATP_ATP_pAMPK', 'AMPKAR', 'pAMPKAR']

# dictionary of the problem for SALib
problem_MM_90 = {'num_vars':14, 'names':param_names_MM, 'bounds': bounds_MM_90,}

seed = np.random.seed(seed=2048)

############################################
# SET UP SOLVER AND MODEL #
############################################
# load the necessary dictionaries
params = model1.ampk_MM_double_mech_get_params()
states = model1.ampk_MM_double_mech_get_states()
nparams = len(params)
nstates = len(states)

# create sunode problem
problem_ode = sunode.SympyProblem(
    params=params,
    states=states,
    rhs_sympy=model1.ampk_MM_double_mech_RHS,
    # specify variables to take gradients wrt
    derivative_params=(),
)

# solver
solver = sunode.solver.Solver(problem_ode, solver='BDF')
lib = sunode._cvodes.lib
lib.CVodeSetMaxNumSteps(solver._ode, 5000) # increase max steps

y0 = set_init_conds_MM(problem_ode, state_names)

 # evaluate the solution at these times
tvals = np.linspace(0, 1000, 2000)


############################################
# Test Case with small number of samples #
############################################
# generate samples using the Saltelli and Morris sampling methods
# START with a small number of samples
nsamps = 8
param_vals_sobol_MM = sobol_samp.sample(problem_MM_90, nsamps, calc_second_order=False, seed=seed)
param_vals_morris_MM = morris_samp.sample(problem_MM_90, nsamps, seed=seed)

# Run simulations
y0 = set_init_conds_MM(problem_ode, state_names)
num_jobs=12

with Parallel(n_jobs=num_jobs) as parallel:
    # sobol
    param_vals_sobol_MM = tqdm(param_vals_sobol_MM)
    sols_sobol = parallel(delayed(single_model_eval)(param, 
                            problem_ode, solver, y0, tvals, state_names) for 
                            param in param_vals_sobol_MM)
    np.save('./MM_double_mech/sols_sobol_short.npy', np.array(sols_sobol, dtype=object))
    del sols_sobol
    # morris
    param_vals_morris_MM = tqdm(param_vals_morris_MM)
    sols_morris = parallel(delayed(single_model_eval)(param, 
                            problem_ode, solver, y0, tvals, state_names) for 
                            param in param_vals_morris_MM)
    np.save('./MM_double_mech/sols_morris_corr_short.npy', np.array(sols_morris, dtype=object))
    del sols_morris

############################################
# Full scale case with many samples #
############################################
# generate samples using the Saltelli and Morris sampling methods
# START with a small number of samples
nsamps = 2048
param_vals_sobol_MM = sobol_samp.sample(problem_MM_90, nsamps, calc_second_order=True, seed=seed)
param_vals_morris_MM = morris_samp.sample(problem_MM_90, nsamps, seed=seed)

np.save('./MM_double_mech/param_vals_sobol.npy', np.array(param_vals_sobol_MM))
np.save('./MM_double_mech/param_vals_morris.npy', np.array(param_vals_morris_MM))

# Run simulations
y0 = set_init_conds_MM(problem_ode, state_names)
num_jobs=12

with Parallel(n_jobs=num_jobs) as parallel:
    # Sobol
    param_vals_sobol_MM = tqdm(param_vals_sobol_MM)
    sols_sobol = parallel(delayed(single_model_eval)(param, 
                            problem_ode, solver, y0, tvals, state_names, full_output=False) for 
                            param in param_vals_sobol_MM)
    np.save('./MM_double_mech/sols_sobol.npy', np.array(sols_sobol))
    del sols_sobol
    # Morris
    param_vals_morris_MM = tqdm(param_vals_morris_MM)
    sols_morris = parallel(delayed(single_model_eval)(param, 
                            problem_ode, solver, y0, tvals, state_names, full_output=False) for 
                            param in param_vals_morris_MM)
    np.save('./MM_double_mech/sols_morris.npy', np.array(sols_morris))
  