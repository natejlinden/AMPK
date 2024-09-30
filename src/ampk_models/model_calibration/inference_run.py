import pdb
from os import environ
environ['OMP_NUM_THREADS'] = '1'

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import diffrax
import sys, argparse, json

# import models
sys.path.append("../odes/")
from ampk_MA_double_mech_diffrax import *
from ampk_MA_single_mech_diffrax import *
from ampk_MM_double_mech_diffrax import *
from ampk_MM_single_mech_diffrax import *
# from ampk_newmech_MA_diffrax import *

sys.path.append("../")
from utils import *
from pymc_jax_ode import *

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
    parser.add_argument("-free_params", type=str, help="parameters to estimate")
    parser.add_argument("-data_file", type=str, help="path to the data file. Should be a NPZ with the following objects: \
                        'times', 'mean', 'std_constant', and 'std'.")
    parser.add_argument("-model_info_file", type=str, help="JSON file with relevant info. Model params, initial conditions, and AMPKAR states.")
    parser.add_argument("-metab_params_file", type=str, help="Metabolism model parameters. Should be a JSON")
    parser.add_argument("-savedir", type=str, help="Path to save results. Defaults to current directory.")
    # priors
    parser.add_argument("-upper_prior_mult", type=float, default=10.0, help="Upper bound for the prior. Defaults to 10.0.")
    parser.add_argument("-lower_prior_mult", type=float, default=1.0, help="Lower bound for the prior. Defaults to 0.1.")
    parser.add_argument("-prior_family", type=str, default="[['Gamma()',['alpha', 'beta']]]", help="Prior family to use. Defaults to Gamma.")
    # MCMC sampling
    parser.add_argument("-ntune", type=int, default=1000, help="Number of MCMC tuning samples. Defaults to 1000.")
    parser.add_argument("-nsamples", type=int, default=1000, help="Number of posterior samples to draw per MCMC chain. Defaults to 1000.")
    parser.add_argument("-nchains", type=int, default=4, help="Number of chains to run. Defaults to 4.")
    parser.add_argument("-nuts_sampler", type=str, default='pymc', help="Name of the NUTS sampler to use ['pymc', 'nutpie', 'blackjax', 'numpyro']. Defaults to 'pymc'")
    parser.add_argument("-nuts_sampler_kwargs", type=str, default=None, help="Dictionary of keyword arguments to pass to the NUTS sampler")
    parser.add_argument("-ncores", type=int, default=1, help="Number of cores to use for multiprocessing. Defaults to None which will use all available cores. Note this is ignored if run in a environment with an NVIDIA GPU.")
    # flags to control what to run
    parser.add_argument("--prior_pred_sample", action='store_true',default=False, \
                        help="Flag to enable prior predictive sampling.") 
    parser.add_argument("--skip_sample", action='store_false',default=True, \
                        help="Flag to skip inference.")
    parser.add_argument("--skip_post_pred_sample", action='store_false',default=True, \
                        help="Flag to skip posterior predictive sampling.")
    # simulation parameters
    parser.add_argument("-tmax_init", type=float, default=1e3, help="Maximum time to run the simulation. Defaults to 1e3.")
    parser.add_argument("-rtol", type=float,default=1e-6)
    parser.add_argument("-atol", type=float,default=1e-6)
    parser.add_argument("-evnt_rtol", type=float,default=1e-12)
    parser.add_argument("-evnt_atol", type=float,default=1e-12)
    parser.add_argument('-pcoeff', type=float, default=0, help='pcoeff for PID time stepper')
    parser.add_argument('-dcoeff', type=float, default=0, help='dcoeff for PID time stepper')
    parser.add_argument('-icoeff', type=float, default=1.0, help='icoeff for PID time stepper')
    
    args=parser.parse_args(raw_args)
    return args


def main(raw_args=None):
    """ Main function to execute command line script functionality. See the args parser for arguments
    """
    args = parse_args(raw_args) # parse the arguments
    print('Processing model {}.'.format(args.model))
    
    seed = np.random.default_rng(12345)

    # add savedir if it does not exist
    if not os.path.isdir(args.savedir):
        os.makedirs(args.savedir)
    ####################################################
    # set up model info and priors #
    ####################################################
    # Load JSON files with param, state, and initial condition info
    # states and initial conditions
    with open(args.model_info_file, 'r') as file:
           model_info = json.load(file)

    # unpack loaded model data dictionary
    state_names = list(model_info["init_conds"].keys())
    ampkar_states = model_info['ampkar_states']
    pampkar_states = model_info['pampkar_states']
    n_states = len(state_names)
    y0 = jnp.array(list(model_info["init_conds"].values()))

    # get the indices of the states
    ampkar_idxs = [state_names.index(item) for item in ampkar_states]
    pampkar_idxs = [state_names.index(item) for item in pampkar_states]
    ampkar_idx = state_names.index('AMPKAR')
    pampkar_idx = state_names.index('pAMPKAR')

    # get the names of the fixed parameters
    param_names = list(model_info['nominal_params'].keys())
    nominal_params = model_info['nominal_params']
    
    free_params = args.free_params.split(',')
    free_param_idxs = [param_names.index(item) for item in free_params]
    fixed_params = list(set(param_names)  - set(free_params))
    fixed_param_idxs = [param_names.index(item) for item in fixed_params]

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
    
    # states and initial conditions
    with open(args.model_info_file, 'r') as file:
           model_info = json.load(file)

    # unpack loaded model data dictionary
    state_names = list(model_info["init_conds"].keys())
    ampkar_states = model_info['ampkar_states']
    pampkar_states = model_info['pampkar_states']
    n_states = len(state_names)
    y0 = jnp.array(list(model_info["init_conds"].values()))

    # get the indices of the states
    ampkar_idxs = [state_names.index(item) for item in ampkar_states]
    pampkar_idxs = [state_names.index(item) for item in pampkar_states]
    ampkar_idx = state_names.index('AMPKAR')
    pampkar_idx = state_names.index('pAMPKAR')

    ############################################
    # PRIORS #
    ############################################
    # perform maximum entropy prior elicitation to find prior params for all free params
    # this returns a dictionary of prior parameters for the model in syntax to use exec to set them in a pymc model object
    free_params = args.free_params.split(',')
    prior_param_dict = set_prior_params(param_names, np.array(list(nominal_params.values())), free_param_idxs, upper_mult=args.upper_prior_mult, lower_mult=args.lower_prior_mult, prior_family=args.prior_family)

    # def simulation function that solves ODE and computes proper qoi
    # the solve_traj function first runs the model to SS in the basal energy state, and then 
    # runs the model in the stressed energy state using the SS from the basal state as the initial condition
    def simulator(params):
        # solve model
        sol_stressed, sol = solve_traj(rhs, rhs_stress, y0, params, times, tmax_init=args.tmax_init, rtol=args.rtol, atol=args.atol, evnt_atol=args.evnt_atol, evnt_rtol=args.evnt_rtol, pcoeff=args.pcoeff, icoeff=args.icoeff, dcoeff=args.dcoeff)

        # compute delta pAMPKAR/AMPKAR_tot
        AMPKAR_stressed = sol_stressed[ampkar_idxs, :].sum(axis=0)
        AMPKAR_basal = sol[ampkar_idxs, :].sum(axis=0)
        pAMPKAR_stressed = sol_stressed[pampkar_idxs, :].sum(axis=0)
        pAMPKAR_basal = sol[pampkar_idxs, :].sum(axis=0)
        
        return (pAMPKAR_stressed/AMPKAR_stressed) - (pAMPKAR_basal/AMPKAR_basal)
    
    ####################################################
    # construct pymc model #
    ####################################################
    # jax functions to solve the model as a func of params
    def sol_op_jax(*params):
        """jax function to solve the model using diffrax"""
        return simulator(params)
    sol_op_jax_jitted = jax.jit(sol_op_jax)

    # gradient
    def vjp_sol_op_jax(gz, *params):
        _, vjp_fn = jax.vjp(sol_op_jax, *params)
        return vjp_fn(gz)
    vjp_sol_op_jax_jitted = jax.jit(vjp_sol_op_jax)

    sol_op = SolOp(sol_op_jax_jitted, vjp_sol_op_jax)
    vjp_sol_op = VJPSolOp(vjp_sol_op_jax_jitted)

    # register with jax
    @jax_funcify.register(sol_op)
    def sol_op_jax_funcify(op, **kwargs):
        return sol_op_jax

    @jax_funcify.register(vjp_sol_op)
    def vjp_sol_op_jax_funcify(op, **kwargs):
        return vjp_sol_op_jax

    # construct PyMC model
    pymc_model = pm.Model()
    with pymc_model:
        # loop over free params and construct the priors
        priors = []
        for key, value in prior_param_dict.items():
            # create PyMC variables for each parameters in the model (both free and fixed)
            prior = eval(value)
            priors.append(prior)

        # predict dose response
        prediction = sol_op(*priors)

        # assume a normal model for the data (ie. normal likelihood)
        llike = pm.Normal("llike", mu=prediction, sigma=data_std, observed=data)
        
    
    ####################################################
    # prior predictive sampling if desired #
    ####################################################
    if args.prior_pred_sample:
        with pymc_model:
            prior_pred = pm.sample_prior_predictive(samples=1000, random_seed = seed)
            # FIXME add input for # of samples
        # save the samples
        prior_pred.to_json(savedir + args.model + '_prior_pred_samples.json')

    ####################################################
    # MCMC (or other sampling) #
    ####################################################
    # SMC sampling
    if args.skip_sample:
        with pymc_model:
            posterior_idata = pm.sample(args.nsamples, tune=args.ntune, 
                                        nuts_sampler = args.nuts_sampler,
                                        nuts_sampler_kwargs=args.nuts_sampler_kwargs,
                                        cores=args.ncores, chains=args.nchains, 
                                        return_inferencedata=True, 
                                        random_seed = seed)
        # save the samples
        posterior_idata.to_json(args.savedir + args.model + '_samples.json')
    else:
        Warning('Skipping sampling. Will attempt to load previous samples.')
        try:
            posterior_idata = az.from_json(args.savedir + args.model + '_samples.json')
        except:
            print('No samples found. Skipping posterior predictive sampling and exiting.')
            sys.exit()

    ####################################################
    # posterior predictive sampling if desired #
    ####################################################
    if args.skip_post_pred_sample:
        with pymc_model:
            post_pred = pm.sample_posterior_predictive(posterior_idata, 
                                                              random_seed=seed)
        # save the samples
        post_pred.to_json(args.savedir + args.model + '_post_pred_samples.json')
    
    print('Completed {}'.format(args.model))

if __name__ == '__main__':
    main()