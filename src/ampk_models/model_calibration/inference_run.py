import pdb
from os import environ
environ['OMP_NUM_THREADS'] = '1'
import multiprocessing

environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count()
)

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import diffrax
import numpyro
import numpyro.distributions as dist
from jax import random
from numpyro.infer import MCMC, NUTS, AIES
from numpyro.infer import Predictive
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
    # MCMC sampling
    parser.add_argument("-nwarmup", type=int, default=1000, help="Number of MCMC tuning samples. Defaults to 1000.")
    parser.add_argument("-nsamples", type=int, default=1000, help="Number of posterior samples to draw per MCMC chain. Defaults to 1000.")
    parser.add_argument("-nchains", type=int, default=1, help="Number of chains to run. Defaults to 1.")
    parser.add_argument("-sampler", type=str, default='NUTS', help="Name of the MCMC sampler to use ['NUTS', 'AIES']. Defaults to 'NUTS'")
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
    
    args=parser.parse_args(raw_args)
    return args


def main(raw_args=None):
    """ Main function to execute command line script functionality. See the args parser for arguments
    """
    args = parse_args(raw_args) # parse the arguments
    print('Processing model {}.'.format(args.model))
    
    # Jax PRGN key
    key = random.PRNGKey(args.seed)

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
    y0 = jnp.array(list(model_info["init_conds"].values()))

    # get the indices of the states
    ampkar_idxs = [state_names.index(item) for item in ampkar_states]
    pampkar_idxs = [state_names.index(item) for item in pampkar_states]

    # # get the names of the fixed parameters
    # param_names = list(model_info['nominal_params'].keys())
    # # nominal_params = model_info['nominal_params']
    
    # free_params = args.free_params.split(',')
    # free_param_idxs = [param_names.index(item) for item in free_params]
    # fixed_params = list(set(param_names)  - set(free_params))
    # fixed_param_idxs = [param_names.index(item) for item in fixed_params]

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

    ###############################################
    #               NumPyro Model                 #
    ###############################################
    try:
        numpyro_model = eval(args.model + '_numpyro_model')
    except:
        print('Warning NumPyro Model {} not found. Quitting.'.format(args.model))
        quit()

    ############################################
    # Data #
    ############################################
    # load the data
    # converts from min to seconds
    data, data_std, times = load_data(args.data_file, to_seconds=True, constant_std=False)
    

    ############################################
    # Simulator func #
    ############################################
    # def simulation function that solves ODE and computes proper qoi
    # the solve_traj function first runs the model to SS in the basal energy state, and then 
    # runs the model in the stressed energy state using the SS from the basal state as the initial condition
    def simulator(params):
        # solve model
        sol_stressed, sol = solve_traj(rhs, rhs_stress, y0, params, times, tmax_init=args.tmax_init, rtol=args.rtol, atol=args.atol, evnt_atol=args.evnt_atol, evnt_rtol=args.evnt_rtol, pcoeff=args.pcoeff, icoeff=args.icoeff, dcoeff=args.dcoeff)

        # compute delta pAMPKAR/AMPKAR_tot
        AMPKAR_stressed = sol_stressed[jnp.array(ampkar_idxs), :].sum(axis=0)
        AMPKAR_basal = sol[jnp.array(ampkar_idxs)].sum(axis=0)
        pAMPKAR_stressed = sol_stressed[jnp.array(pampkar_idxs), :].sum(axis=0)
        pAMPKAR_basal = sol[jnp.array(pampkar_idxs)].sum(axis=0)
        
        return (pAMPKAR_stressed/AMPKAR_stressed) - (pAMPKAR_basal/AMPKAR_basal)


    ####################################################
    # Set up NumPyro sampling #
    ####################################################
    # sampler
    if args.sampler == 'NUTS':
        kernel = NUTS(numpyro_model)
        chain_method = 'parallel' # different chain method for NUTS
    elif args.sampler == 'AIES':
        if args.mvoes==None:
            moves = {AIES.DEMove() : 0.5, AIES.StretchMove() : 0.5}
        else:
            moves = args.moves
        kernel = AIES(numpyro_model, moves=moves)
        chain_method = 'vectorized' # AIES only works with the 'vectorized' chain method

    # MCMC set up
    mcmc = MCMC(kernel, num_warmup=args.nwarmup, num_samples=args.nsamples, 
                num_chains=args.nchains, chain_method=chain_method)
    
    ####################################################
    # prior sampling #
    ####################################################
    key, newkey = random.split(key)
    prior = Predictive(numpyro_model, num_samples=500)(newkey, data=data, data_std=data_std, solver=simulator)

    ####################################################
    # MCMC (or other sampling) #
    ####################################################
    print('Running MCMC for model {}'.format(args.model))
    key, newkey = random.split(key)
    mcmc.run(newkey, data=data, data_std=data_std, solver=simulator) # run the MCMC
    posterior_samples = mcmc.get_samples() # get the samples

    ####################################################
    # posterior predictive sampling #
    ####################################################
    key, newkey = random.split(key)
    print('Running posterior predictive sampling for model {}'.format(args.model))
    post_pred = Predictive(numpyro_model, posterior_samples)(newkey, data=data, data_std=data_std, solver=simulator)

    ####################################################
    # save the samples #
    ####################################################
    az_data = az.from_numpyro(
        mcmc,
        prior = prior,
        posterior_predictive=post_pred
    )

    # save as netcdf file
    az_data.to_netcdf(os.path.join(args.savedir, args.model + '_mcmc_samples.nc'))
                              
    
    print('Completed {}'.format(args.model))

if __name__ == '__main__':
    main()