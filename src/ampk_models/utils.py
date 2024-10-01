# Nathaniel Linden 2023
# Utilities to help with the project
import json
import os
import sys

import numpy as np
import jax.numpy as jnp
import pandas as pd
import jax

import pymc as pm
from pymc.sampling.jax import sample_numpyro_nuts
import pytensor
import pytensor.tensor as pt
from pytensor.graph import Apply, Op
from pytensor.link.jax.dispatch import jax_funcify
import arviz as az
import preliz as pz
import diffrax as dfrx
from optimistix import root_find, Newton, two_norm
import lineax as lx
#from tqdm import tqdm
import time

jax.config.update("jax_enable_x64", True)
rng = np.random.default_rng(seed=1234)

# load DIFFRAX PYTENSOR OP for JAX ODE
from pymc_jax_ode import *

###############################################################################
#### General Utilities ####
###############################################################################
def load_smc_samples_to_idata(samples_json, sample_time=False):
    """ Load SMC samples from json file to arviz InferenceData object """
    with open(samples_json, 'r') as f:
        data = json.load(f)
    
    # create idata object from dictionary
    # ignore sample stats because that changes with each SMC chain
    idata = az.from_dict(
        posterior =  data['posterior'],
        posterior_attrs = data['posterior_attrs'],
        # sample_stats = data['sample_stats'],
        observed_data = data['observed_data'],
        observed_data_attrs = data['observed_data_attrs'],
        log_likelihood = data['log_likelihood'],
        log_likelihood_attrs = data['log_likelihood_attrs'],
        constant_data = data['constant_data'],
        constant_data_attrs = data['constant_data_attrs'],
        attrs = data['attrs'],
    )

    sample_stats = data['sample_stats']

    if sample_time:
        return idata, sample_stats, data['sample_stats_attrs']['_t_sampling']
    else:
        return idata, sample_stats
 
def get_color_pallette(n_colors=11, append_colors=['#363737','#929591','#d8dcd6']):
    """Function to get standard colors for the project.
    
    Uses a desaturated version of the seaborn colorblind pallette.
    """
    colors = sns.color_palette("colorblind", n_colors, desat=0.65)
    return colors + append_colors

def load_data(data_file, to_seconds=False, constant_std=False):
    """ Loads the data from the specified file.
    """
    data = np.load(data_file) # read data npz file

    # handle time, convert to seconds if specified
    times = data['times']
    zero_idx = int(np.where(times==0.0)[0]) # we only want values after the 2-DG stimulus
    if to_seconds:
        mult = 60
    else:
        mult = 1
    times = mult*times[zero_idx:]

    # handle data
    mean_data = data['mean'][zero_idx:]

    if constant_std:
        std_data = data['std_constant']*np.ones_like(mean_data)
    else:
        std_data = data['std'][zero_idx:]

    return mean_data, std_data, times
###############################################################################
#### Solving ODEs ####
###############################################################################
@jax.jit
def solve_traj(rhs, rhs_stress, y0, params, times, rtol=1e-6, atol=1e-6, 
               evnt_rtol = 1e-12, evnt_atol = 1e-12, tmax_init = 1e3, 
               pcoeff=0, icoeff=1, dcoeff=0, solver = dfrx.Kvaerno5()):
    """ simulates a model over the specified time interval and returns the 
    calculated values.
    Returns an array of shape (n_species, 1) 
    TODO add way to specify autodiff method
    """
    dt0=1e-3
    stepsize_controller=dfrx.PIDController(rtol, atol, pcoeff=pcoeff, icoeff=icoeff, dcoeff=dcoeff)
    event = dfrx.SteadyStateEvent(rtol=evnt_rtol, atol=evnt_atol)
    t0 = 0.0
    t1 = times[-1]
    saveat=dfrx.SaveAt(ts=times)

    # first solve the basal model to SS
    sol = dfrx.diffeqsolve(
        rhs, solver, 
        t0, tmax_init, dt0, 
        y0, 
        args=params,
        stepsize_controller=stepsize_controller,
        discrete_terminating_event=event,
        max_steps=60000, throw=True)
    
    # then use that solution as the initial condition for the stressed setting
    sol_stressed = dfrx.diffeqsolve(
        rhs_stress, solver, 
        t0, t1, dt0, 
        jnp.squeeze(sol.ys), # use basal SS at IC
        args=params, saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=60000, throw=True)
    
    return jnp.squeeze(sol_stressed.ys), jnp.squeeze(sol.ys)

@jax.jit
def solve_SS(rhs, rhs_stress, y0, params, rtol=1e-6, atol=1e-6, 
             evnt_rtol = 1e-12, evnt_atol = 1e-12, tmax = 1e3,
             pcoeff=0, icoeff=1, dcoeff=0, solver = dfrx.Kvaerno5()):
    """ simulates a model over the specified time interval and returns the 
    calculated values.
    Returns an array of shape (n_species, 1) 
    TODO add way to specify autodiff method
    """
    dt0=1e-3
    solver = dfrx.Kvaerno5()
    event = dfrx.SteadyStateEvent(rtol=evnt_rtol, atol=evnt_atol)
    stepsize_controller=dfrx.PIDController(rtol, atol, pcoeff=pcoeff, icoeff=icoeff, dcoeff=dcoeff)
    t0 = 0.0

    # first solve the basal model
    sol = dfrx.diffeqsolve(
        rhs, solver, 
        t0, tmax, dt0, 
        y0, args=params,
        stepsize_controller=stepsize_controller,
        discrete_terminating_event=event,
        max_steps=60000, throw=True)
    
    # then use that solution as the initial condition for the stressed setting
    sol_stressed = dfrx.diffeqsolve(
        rhs_stress, solver, 
        t0, tmax, dt0, 
        jnp.squeeze(sol.ys), args=params,
        stepsize_controller=stepsize_controller,
        discrete_terminating_event=event,
        max_steps=60000, throw=True)
    
    return jnp.squeeze(sol_stressed.ys), jnp.squeeze(sol.ys)


###############################################################################
#### PyMC Inference Utils ####
###############################################################################
def set_prior_params(param_names, nominal_params, free_param_idxs, prior_family=[['Gamma()',['alpha', 'beta']]], upper_mult=1.9, lower_mult=0.1, prob_mass_bounds=0.95):
    """ Sets the prior parameters by finding parameters of the specified prior such that the specified probability mass is between the upper and lower bound.

    Uses the preliz maximum entropy function.
        Inputs:
            - param_names (list): list of parameter names
            - nominal_params (np.ndarray): array of nominal parameter values
            - free_param_idxs (list): list of indices of the free parameters
            - prior_family (str): prior family to use for the parameters. If a string will use that family for all free parameters, otherwise should be a list of strings of the same length as free_param_idxs. Each string should correspond to a pm.Distribution and pz.Distribution object, e.g., Gamma which is the default familly.
            - upper_mult (float): multiplier for the upper bound of the prior
            - lower_mult (float): multiplier for the lower bound of the prior
        Returns:
            - prior_param_dict (dict): dictionary of prior parameters for the model in syntax to use exec to set them in a pymc model object
    """

    # determine if a string or list of strings was passed for the prior family
    prior_family = eval(prior_family)
    if len(prior_family) == 1:
        prior_family_list = prior_family*len(free_param_idxs)
    else:
        prior_family_list = prior_family
    
    # set the prior parameters
    prior_param_dict = {}
    for i, param in enumerate(param_names):
        if i in free_param_idxs: # check if we are dealing with a free parameter
            # get the nominal value
            nominal_val = nominal_params[i]
            if nominal_val == 0:
                upper = 1.0
                lower = 1e-4
            else:
                # get the upper and lower bounds
                upper = nominal_val*upper_mult
                lower = nominal_val*lower_mult
                
                print(param, upper, lower)

            # use preliz.maxent to find the prior parameters for the specified family
            prior_fam = prior_family_list[free_param_idxs.index(i)]
            
            dist_family = eval('pz.' + prior_fam[0])
            ax, results = pz.maxent(dist_family, lower, upper, prob_mass_bounds, plot=False) # for some reason the [0] element is None

            # set the prior parameters
            prior_fam_name = prior_fam[0].strip(')').split('(')[0]
            fixed_params = prior_fam[0].strip(')').split('(')[1].split(',')
            
            tmp = 'pm.' + prior_fam_name + '("' + param + '",'
            for i, hyper_param in enumerate(prior_fam[1]):
                tmp += hyper_param + '=' + str(results.x[i]) + ', '
            
            for fixed_param in fixed_params:
                if len(fixed_param) > 0:
                    tmp += (fixed_param + ', ')
                        
            prior_param_dict[param] = tmp + ')'
            print(prior_param_dict[param])
        else: # fixed parameter
            # set the prior parameters to the nominal value
            prior_param_dict[param] = 'pm.ConstantData("' + param + '", ' + str(nominal_params[i]) + ')'

    return prior_param_dict