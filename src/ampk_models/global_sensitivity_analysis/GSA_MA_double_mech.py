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

import ampk_MA_double_mech as model1
from gsa_utils import *
%matplotlib inline
plt.style.use('~/.matplotlib/custom.mplstyle')
mpl.rcParams['figure.autolayout'] = True

############################################
# Bounds and info for params #
############################################

# define the bounds for the parameters
# we use plus or minus on order of magnitude of any known values and then make reasonable assumptions for unknowns
# Note we fix all off rates to 1.0 and dont bother sampling these or computing sensitivities
############################################
# First for Mass action parameters
nominal_vals_MA = [
   2.5e-3, # KdAMP
   1.5e-3, # KdADP
   1.7e-3, # KdATP
   90.467, # kOnCaMKK = (koff+kphos/Km)
   0.357, # kPhosCaMKK
   0.742, # kOnLKB1
   3.92e-2, # kPhosLKB1
   16.56, # kOnPP
   1.1e-1, # kDephosPP
   1569.59, # kOnAMPK
   6.33, # kPhosAMPK
   16.56, # kOnPP1
   1.1e-1, # kDephosPP1
   1e-3, # AMPKAR
]

bounds_MA_90 = [[0.1*param, 1.9*param] for param in nominal_vals_MA]

param_names_MA = ['KdAMP', 'KdADP', 'KdATP', 'kOnCaMKK', 'kPhosCaMKK', 
               'kOnLKB1','kPhosLKB1','kOnPP','kDephosPP', 
               'kOnAMPK','kPhosAMPK', 'kOnPP1','kDephosPP1', 'AMPKAR']
state_names = ['AMP', 'ADP', 'ATP', 'AMPK', 'pAMPK', 'AMP_AMPK', 'ADP_AMPK', 'ATP_AMPK', 'AMP_pAMPK', 'ADP_pAMPK', 'ATP_pAMPK', 'AMP_AMP_AMPK', 'AMP_ADP_AMPK', 'AMP_ATP_AMPK', 'ADP_ADP_AMPK', 'ADP_ATP_AMPK', 'ATP_ATP_AMPK', 'AMP_AMP_pAMPK', 'AMP_ADP_pAMPK', 'AMP_ATP_pAMPK', 'ADP_ADP_pAMPK', 'ADP_ATP_pAMPK', 'ATP_ATP_pAMPK', 'CaMKK', 'CaMKK_AMPK', 'CaMKK_AMP_AMPK', 'CaMKK_ADP_AMPK', 'CaMKK_ATP_AMPK', 'CaMKK_AMP_AMP_AMPK', 'CaMKK_AMP_ADP_AMPK', 'CaMKK_AMP_ATP_AMPK', 'CaMKK_ADP_ADP_AMPK', 'CaMKK_ADP_ATP_AMPK', 'CaMKK_ATP_ATP_AMPK', 'LKB1', 'LKB1_AMP_AMPK', 'LKB1_ADP_AMPK', 'LKB1_AMP_AMP_AMPK', 'LKB1_AMP_ADP_AMPK', 'LKB1_ADP_ADP_AMPK', 'PP', 'PP_pAMPK', 'PP_ATP_pAMPK', 'PP_AMP_ATP_pAMPK', 'PP_ADP_ATP_pAMPK', 'PP_ATP_ATP_pAMPK', 'AMPKAR', 'pAMPKAR', 'AMPKAR_AMP_pAMPK', 'AMPKAR_AMP_AMP_pAMPK', 'AMPKAR_AMP_ADP_pAMPK', 'PP1', 'PP1_pAMPKAR']

############################################
# We also want to sample Michaelis-Menten parameters to generate correlated samples
# of the mass action parameters
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

# dictionary of the problem for SALib
problem_MA_90 = {'num_vars':14, 'names':param_names_MA, 'bounds': bounds_MA_90,}
problem_MM_90 = {'num_vars':14, 'names':param_names_MM, 'bounds': bounds_MM_90,}


seed = np.random.seed(seed=2048)


############################################
# SET UP SOLVER AND MODEL #
############################################
# load the necessary dictionaries
params = model1.ampk_MA_double_mech_get_params()
states = model1.ampk_MA_double_mech_get_states()
nparams = len(params)
nstates = len(states)

# create sunode problem
problem_ode = sunode.SympyProblem(
    params=params,
    states=states,
    rhs_sympy=model1.ampk_MA_double_mech_RHS,
    # specify variables to take gradients wrt
    derivative_params=(),
)

# solver
solver = sunode.solver.Solver(problem_ode, solver='BDF')
lib = sunode._cvodes.lib
lib.CVodeSetMaxNumSteps(solver._ode, 5000) # increase max steps

y0 = set_init_conds(problem_ode, state_names)

 # evaluate the solution at these times
tvals = np.linspace(0, 1000, 2000)


############################################
# Test Case with small number of samples #
############################################
# generate samples using the Saltelli and Morris sampling methods
# START with a small number of samples
nsamps = 8
param_vals_sobol_MA = sobol_samp.sample(problem_MA_90, nsamps, calc_second_order=False, seed=seed)
param_vals_morris_MA = morris_samp.sample(problem_MA_90, nsamps, seed=seed)
param_vals_sobol_MM = sobol_samp.sample(problem_MM_90, nsamps, calc_second_order=False, seed=seed)
param_vals_morris_MM = morris_samp.sample(problem_MM_90, nsamps, seed=seed)

# copy MA to np arrays
param_vals_sobol_MA_corr = np.array(param_vals_sobol_MA)
param_vals_morris_MA_corr = np.array(param_vals_morris_MA)
param_vals_sobol_MM_np = np.array(param_vals_sobol_MM)
param_vals_morris_MM_np = np.array(param_vals_morris_MM)

# now compute all kons using samples of Km and from the MM model and Kcat from MA model
kcat_idxs_MA = [4,6,8,10,12]
kon_idxs_MA = [3,5,7,9,11]
km_idxs_MM = [4,6,8,10,12]
for kcat_i, kon_i, km_i in zip(kcat_idxs_MA, kon_idxs_MA, km_idxs_MM):
    param_vals_sobol_MA_corr[:,kon_i] = (1+param_vals_sobol_MA_corr[:,kcat_i])/param_vals_sobol_MM_np[:,km_i]
for kcat_i, kon_i, km_i in zip(kcat_idxs_MA, kon_idxs_MA, km_idxs_MM):  
    param_vals_morris_MA_corr[:,kon_i] = (1+param_vals_morris_MA_corr[:,kcat_i])/param_vals_morris_MM_np[:,km_i]

# Run simulations
y0 = set_init_conds(problem_ode, state_names)
num_jobs=6


# uncorrelated
param_vals_sobol_MA= tqdm(param_vals_sobol_MA)
param_vals_morris_MA= tqdm(param_vals_morris_MA)
sols_sobol = Parallel(n_jobs=num_jobs)(delayed(single_model_eval)(param, 
                        problem_ode, solver, y0, tvals, state_names) for 
                        param in param_vals_sobol_MA)
np.save('./MA_double_mech/sols_sobol.npy', np.array(sols_sobol))
sols_morris = Parallel(n_jobs=num_jobs)(delayed(single_model_eval)(param, 
                        problem_ode, solver, y0, tvals, state_names) for 
                        param in param_vals_morris_MA)
np.save('./MA_double_mech/sols_morris.npy', np.array(sols_morris))
# correlated
param_vals_sobol_MA_corr= tqdm(param_vals_sobol_MA_corr)
param_vals_morris_MA_corr= tqdm(param_vals_morris_MA_corr)
sols_sobol_corr = Parallel(n_jobs=num_jobs)(delayed(single_model_eval)(param, 
                        problem_ode, solver, y0, tvals, state_names) for 
                        param in param_vals_sobol_MA_corr)
np.save('./MA_double_mech/sols_sobol_corr.npy', np.array(sols_sobol_corr))
sols_morris_corr = Parallel(n_jobs=num_jobs)(delayed(single_model_eval)(param, 
                        problem_ode, solver, y0, tvals, state_names) for 
                        param in param_vals_morris_MA_corr)
np.save('./MA_double_mech/sols_morris_corr.npy', np.array(sols_morris_corr))