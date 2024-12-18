import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import diffrax as dfrx
import equinox as eqx
import pymc as pm
from pymc.sampling.jax import sample_numpyro_nuts, sample_blackjax_nuts, get_jaxified_logp
from pytensor.link.jax.dispatch import jax_funcify

import numpyro
import numpyro.distributions as dist
from jax import random
from numpyro.infer import MCMC, NUTS, AIES, init_to_sample
import arviz as az
from numpyro.infer import Predictive
import sys, argparse, json, os

sys.path.append("../")
from utils import *
from pymc_jax_ode import *

sys.path.append("../models/")

# tell jax to use 64bit floats
jax.config.update("jax_enable_x64", True)

##############################
# def arg parsers to take inputs from the command line
##############################
def parse_args(raw_args=None):
    """ function to parse command line arguments
    """
    parser=argparse.ArgumentParser(description="Run MCMC for AMPK models.")
    # model info and general setup
    parser.add_argument("-model", type=str, help="model to process.")
    parser.add_argument("-compartment", type=str, help="compartment to which data belongs.")
    parser.add_argument("-free_params", type=str, help="parameters to estimate")
    parser.add_argument("-data_file", type=str, help="path to the data file. Should be a NPZ with the following objects: \
                        'times', 'mean', 'std_constant', and 'std'.")
    parser.add_argument("-model_info_file", type=str, help="JSON file with relevant info. Model params, initial conditions, and AMPKAR states.")
    parser.add_argument("-metab_params_file", type=str, help="Metabolism model parameters. Should be a JSON")
    parser.add_argument("-savedir", type=str, help="Path to save results. Defaults to current directory.")
    # MCMC sampling
    parser.add_argument("-prior_family", type=str, default="[['Gamma()',['alpha', 'beta']]]", help="Family of priors to use. Defaults to 'lognormal'.")
    parser.add_argument("-lower_mult", type=float, default=0.1, help="Lower bound multiplier for uniform priors. Defaults to 0.1.")
    parser.add_argument("-upper_mult", type=float, default=2.0, help="Upper bound multiplier for uniform priors. Defaults to 2.0.")
    parser.add_argument("-nwarmup", type=int, default=1000, help="Number of MCMC tuning samples. Defaults to 1000.")
    parser.add_argument("-nsamples", type=int, default=1000, help="Number of posterior samples to draw per MCMC chain. Defaults to 1000.")
    parser.add_argument("-nchains", type=int, default=1, help="Number of chains to run. Defaults to 1.")
    parser.add_argument("-sampler", type=str, default='NUTS', help="Name of the MCMC sampler to use ['NUTS', 'NUTS-ADVI', 'NumpyroNUTS', 'BlackJaxNUTS']. Defaults to 'NUTS'")
    # simulation parameters
    parser.add_argument("-tmax_init", type=float, default=1e3, help="Maximum time to run the simulation. Defaults to 1e3.")
    parser.add_argument("-rtol", type=float,default=1e-6)
    parser.add_argument("-atol", type=float,default=1e-6)
    parser.add_argument("-evnt_rtol", type=float,default=1e-12)
    parser.add_argument("-evnt_atol", type=float,default=1e-12)
    parser.add_argument('-pcoeff', type=float, default=0, help='pcoeff for PID time stepper')
    parser.add_argument('-dcoeff', type=float, default=0, help='dcoeff for PID time stepper')
    parser.add_argument('-icoeff', type=float, default=1.0, help='icoeff for PID time stepper')
    # other
    parser.add_argument("-seed", type=int, default=0, help="Random seed to use. Defaults to 0.")
    parser.add_argument("-prior_only", type=bool, default=False, help="Boolean to only sample from the prior.")
    parser.add_argument("-n_advi_iter", type=int, default=1000, help="Number of iterations for ADVI. Defaults to 1000.")
    
    args=parser.parse_args(raw_args)
    return args


def main(raw_args=None):
    # jax.config.update("jax_enable_x64", True)
    """ Main function to execute command line script functionality. See the args parser for arguments
    """
    args = parse_args(raw_args) # parse the arguments
    print('Processing model {}.'.format(args.model))

    # add savedir if it does not exist
    if not os.path.isdir(args.savedir):
        os.makedirs(args.savedir)
    ####################################################
    # set up model info and priors #
    ####################################################
    # import the model
    try:
        exec('from ' + args.model + '_diffrax import *')
    except:
        print('Warning Model {} not found. Quitting.'.format(args.model))
        quit()

    # Load JSON files with param, state, and initial condition info
    # states and initial conditions
    with open(args.model_info_file, 'r') as file:
           model_info = json.load(file)

    # unpack loaded model data dictionary
    state_names = list(model_info["init_conds"].keys())
    ampkar_states = model_info['ampkar_states']
    pampkar_states = model_info['pampkar_states']
    y0 = list(model_info["init_conds"].values())

    # process the free parameters
    free_params = args.free_params.split(',')

    # get the indices of the states
    ampkar_idxs = [state_names.index(item) for item in ampkar_states]
    pampkar_idxs = [state_names.index(item) for item in pampkar_states]

    # parameters for the metabolic model
    with open(args.metab_params_file, 'r') as file:
           metab_params = json.load(file)

    basal_params = list(metab_params["metab_params_basal"].values())
    stress_params = list(metab_params["metab_params_stress"].values())
    ###############################################
    #                   Model RHS                  #
    ################################################
    try:
        rhs = eval(args.model + '(' + ','.join(str(elm) for elm in basal_params) \
            + ')')
        rhs_stress = eval(args.model + '(' + ','.join(str(elm) for elm in stress_params) \
             + ')')
        rhs = dfrx.ODETerm(rhs)
        rhs_stress = dfrx.ODETerm(rhs_stress)
    except:
        print('Warning Model {} not found. Quitting.'.format(args.model))
        quit()

    ############################################
    # Data #
    ############################################
    # load the data
    # converts from min to seconds
    data, data_std, times = load_data(args.data_file, to_seconds=True, constant_std=False)
    print(data.shape)
    data = data.reshape(1, len(data))
    data_std = data_std.reshape(1, len(data_std))
    # data_std = 1e-5*data_std

    print(y0)

    ############################################
    # Simulator func #
    ############################################
    # def simulation function that solves ODE and computes proper qoi
    # the solve_traj function first runs the model to SS in the basal energy state, and then 
    # runs the model in the stressed energy state using the SS from the basal state as the initial condition
    def simulator(params):
        # solve model
        sol_stressed, sol = solve_traj(rhs, rhs_stress, y0, params, times, tmax_init=args.tmax_init, rtol=args.rtol, atol=args.atol, evnt_atol=args.evnt_atol, evnt_rtol=args.evnt_rtol, pcoeff=args.pcoeff, icoeff=args.icoeff, dcoeff=args.dcoeff, dt0=1e-10)

        # compute delta pAMPKAR/AMPKAR_tot
        AMPKAR_stressed = sol_stressed[jnp.array(ampkar_idxs), :].sum(axis=0)
        pAMPKAR_stressed = sol_stressed[jnp.array(pampkar_idxs), :].sum(axis=0)
        result = pAMPKAR_stressed/AMPKAR_stressed
        
        return jnp.reshape(result, (1, len(result)))
    
    # construct PyTensor Op for simulator
    def sol_op_jax(*params):
        return simulator(params)
    
    sol_op_jax_jitted = eqx.filter_jit(sol_op_jax)
    
    def vjp_sol_op_jax(gz, *params):
        _, vjp_fn = jax.vjp(sol_op_jax, *params)
        return vjp_fn(gz)

    vjp_sol_op_jax_jitted = eqx.filter_jit(vjp_sol_op_jax)

    vjp_sol_op = VJPSolOp(vjp_sol_op_jax_jitted)
    sol_op = SolOp(sol_op_jax_jitted, vjp_sol_op)

    # register the ops with PyTensor
    @jax_funcify.register(SolOp)
    def sol_op_jax_funcify(op, **kwargs):
        return sol_op_jax

    @jax_funcify.register(VJPSolOp)
    def vjp_sol_op_jax_funcify(op, **kwargs):
        return vjp_sol_op_jax

    ####################################################
    # PyMC model #
    ####################################################
    prior_dict = set_prior_params(list(model_info["nominal_params"].keys()), 
                                  free_params, model_info["nominal_params"], 
                                  upper_mult=args.upper_mult, lower_mult=args.lower_mult, prior_family=args.prior_family)
    
    pm_model = build_pymc_model(prior_dict, data, sol_op, data_sigma=data_std)

    ###################################################
    # prior sampling #
    ###################################################
    with pm_model:
        prior_pred = pm.sample_prior_predictive(samples=2000, random_seed=args.seed)

    if args.prior_only:
        prior_pred.to_netcdf(os.path.join(args.savedir, args.model + '_' \
                                          + args.compartment + '_prior_samples.nc'))
    
    if not args.prior_only:
        #####################################################
        # MCMC (or other sampling) #
        #####################################################
        print('Running MCMC for model {}'.format(args.model))

        with pm_model:
            if args.sampler == 'NUTS':
                posterior = pm.sample(args.nsamples, tune=args.nwarmup, chains=args.nchains, 
                                    cores=1, random_seed=args.seed, 
                                    idata_kwargs={'log_likelihood': True})
                
            elif args.sampler == 'NUTS-ADVI':
                posterior = pm.sample(args.nsamples, tune=args.nwarmup, chains=args.nchains, 
                                    cores=1, init='ADVI', random_seed=args.seed, 
                                    idata_kwargs={'log_likelihood': True})
            elif args.sampler == 'BlackJaxNUTS':
                posterior = sample_blackjax_nuts(draws=args.nsamples, tune=args.nwarmup, 
                                                jitter=False, chains=args.nchains, 
                                                chain_method='vectorized', seed=args.seed, 
                                                idata_kwargs={'log_likelihood': True})
            elif args.sampler == 'NumpyroNUTS':
                posterior = sample_numpyro_nuts(draws=args.nsamples, tune=args.nwarmup, jitter=False,
                                                chains=args.nchains, random_seed=args.seed,
                                                idata_kwargs={'log_likelihood': True})
            elif args.sampler == "ADVI":
                mean_field = pm.fit(n=args.n_advi_iter)
                posterior = mean_field.sample(draws=args.nsamples)
            
        ####################################################
        # posterior predictive sampling #
        ####################################################
        print('Running posterior predictive sampling for model {}'.format(args.model))
        post_pred = pm.sample_posterior_predictive(posterior, model=pm_model)

        # ####################################################
        # # save the samples #
        # ####################################################
        posterior.extend(prior_pred)
        posterior.extend(post_pred)

        # save as netcdf file
        posterior.to_netcdf(os.path.join(args.savedir, args.model + '_' + \
                                        args.compartment + '_mcmc_samples_' + args.sampler + '.nc'))
                              
    print('Completed {}'.format(args.model))

if __name__ == '__main__':
    main()