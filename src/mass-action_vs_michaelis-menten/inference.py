import pdb
from os import environ

import numpyro.infer.util
environ['OMP_NUM_THREADS'] = '1'
import multiprocessing

environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count()
)

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import diffrax as dfrx
import equinox as eqx
import numpyro
import numpyro.distributions as dist
from jax import random
from numpyro.infer import MCMC, NUTS, AIES, init_to_sample
import arviz as az
from numpyro.infer import Predictive
import sys, argparse, json, os

sys.path.append("../ampk_models/")
from utils import *

# tell jax to use 64bit floats
jax.config.update("jax_enable_x64", True)
numpyro.enable_x64()

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
    parser.add_argument("-savedir", type=str, help="Path to save results. Defaults to current directory.")
    # MCMC sampling
    parser.add_argument("-nwarmup", type=int, default=1000, help="Number of MCMC tuning samples. Defaults to 1000.")
    parser.add_argument("-nsamples", type=int, default=1000, help="Number of posterior samples to draw per MCMC chain. Defaults to 1000.")
    parser.add_argument("-nchains", type=int, default=1, help="Number of chains to run. Defaults to 1.")
    parser.add_argument("-sampler", type=str, default='NUTS', help="Name of the MCMC sampler to use ['NUTS', 'AIES']. Defaults to 'NUTS'")
    # simulation parameters
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
    jax.config.update("jax_enable_x64", True)
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
    # import the model
    try:
        exec('from ' + args.model + '_diffrax import *')
        exec('from ' + args.model + '_numpyro import *')
    except:
        print('Warning Model {} not found. Quitting.'.format(args.model))
        quit()

    # Load JSON files with param, state, and initial condition info
    # states and initial conditions
    with open(args.model_info_file, 'r') as file:
           model_info = json.load(file)

    # unpack loaded model data dictionary
    state_names = list(model_info["init_conds"].keys())
    sub_prod_states = model_info["sub_prod_states"]
    prod_states = model_info["prod_states"]
    y0 = list(model_info["init_conds"].values())

    # get the indices of the states
    sub_prod_idxs = [state_names.index(item) for item in sub_prod_states]
    prod_idxs = [state_names.index(item) for item in prod_states]

    ###############################################
    #                   Model RHS                  #
    ################################################
    try:
        rhs = eval(args.model + '()')
        rhs = dfrx.ODETerm(rhs)
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
    dat = pd.read_csv(args.data_file)
    times = dat['time'].to_numpy()
    data = dat['conc (mM)'].to_numpy()

    # assume data std is 5% of the mean
    data_std = 0.05*np.mean(data)

    ############################################
    # Simulator func #
    ############################################
    # def simulation function that solves ODE and computes proper qoi
    # the solve_traj function first runs the model to SS in the basal energy state, and then 
    # runs the model in the stressed energy state using the SS from the basal state as the initial condition
    solver = dfrx.Kvaerno5()
    dt0=1e-5
    stepsize_controller=dfrx.PIDController(args.rtol, args.atol, pcoeff=args.pcoeff, icoeff=args.icoeff,dcoeff=args.dcoeff,
                                           dtmin=1e-10, force_dtmin=True)
    t0 = 0.0
    t1 = times[-1]
    saveat=dfrx.SaveAt(ts=times)


    def simulator(params):
        # solve model
        # first solve the basal model to SS
        sol = dfrx.diffeqsolve(rhs, solver, t0=t0, t1=t1, dt0=dt0, 
        y0=y0, args=params, saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=4096, throw=True)

        sol = jnp.squeeze(jnp.array(sol.ys))

        # compute ratio product/(total substrate + product)
        sub_prod = sol[jnp.array(sub_prod_idxs), :].sum(axis=0)
        prod = sol[jnp.array(prod_idxs), :].sum(axis=0)

        return prod #/sub_prod


    ####################################################
    # Set up NumPyro sampling #
    ####################################################
    # sampler
    if args.sampler == 'NUTS':
        kernel = NUTS(numpyro_model, init_strategy=numpyro.infer.init_to_mean)
        chain_method = 'parallel' # different chain method for NUTS
    elif args.sampler == 'AIES':
        moves = {AIES.DEMove() : 0.5, AIES.StretchMove() : 0.5}
        kernel = AIES(numpyro_model, moves=moves)
        chain_method = 'vectorized' # AIES only works with the 'vectorized' chain method

    # MCMC set up
    mcmc = MCMC(kernel, num_warmup=args.nwarmup, num_samples=args.nsamples, 
                num_chains=args.nchains, chain_method=chain_method)
    
    ####################################################
    # prior sampling #
    ####################################################
    key, newkey = random.split(key)
    prior = Predictive(numpyro_model, num_samples=500)(newkey, y_std=data_std, solver=simulator)

    # func = eqx.filter_jit(jax.value_and_grad(simulator))

    # def func(params):
    #     p_dict = {'k_f': params[0], 'k_r': params[1], 'k_cat': params[2]}
    #     return jnp.sum(numpyro.infer.util.log_likelihood(numpyro_model, p_dict, y=data, y_std=data_std, solver=simulator)['obs'])
    
    # grad_val_func = eqx.filter_jit(jax.value_and_grad(func))

    # for i in range(500):
    #     k_f = prior['k_f'][i]
    #     k_r = prior['k_r'][i]
    #     k_cat = prior['k_cat'][i]
    #     params = jnp.array((k_f, k_r, k_cat))
    #     print(grad_val_func(params))
    



    ####################################################
    # MCMC (or other sampling) #
    ####################################################
    print('Running MCMC for model {}'.format(args.model))
    key, newkey = random.split(key)
    eqx.filter_jit(mcmc.run(newkey, y=data, y_std=data_std, solver=simulator)) # run the MCMC
    posterior_samples = mcmc.get_samples() # get the samples

    # ####################################################
    # # posterior predictive sampling #
    # ####################################################
    # key, newkey = random.split(key)
    # print('Running posterior predictive sampling for model {}'.format(args.model))
    # post_pred = Predictive(numpyro_model, posterior_samples)(newkey, y_std=data_std, solver=simulator)

    # ####################################################
    # # save the samples #
    # ####################################################
    # az_data = az.from_numpyro(
    #     mcmc,
    #     prior = prior,
    #     posterior_predictive=post_pred
    # )

    # # save as netcdf file
    # az_data.to_netcdf(os.path.join(args.savedir, args.model + '_mcmc_samples.nc'))
                              
    # print('Completed {}'.format(args.model))

if __name__ == '__main__':
    main()