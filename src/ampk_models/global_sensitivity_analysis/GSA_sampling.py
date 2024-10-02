from os import environ
environ['OMP_NUM_THREADS'] = '1'

import numpy as np
from SALib.sample import sobol as sobol_samp
from SALib.sample import morris as morris_samp
from SALib.analyze import sobol as sobol_analyze
from SALib.analyze import morris as morris_analyze
from SALib.analyze.hdmr import analyze as hdmr_analyze
import os, sys, time, json, argparse
import pandas as pd
import argparse
from gsa_utils import * # GSA utility functions
import jax
import jax.numpy as jnp
from jax import lax
import equinox as eqx
import diffrax as dfrx

# import models
sys.path.append("../odes/")
from ampk_MA_double_mech_diffrax import *
from ampk_MA_single_mech_diffrax import *
from ampk_MM_double_mech_diffrax import *
from ampk_MM_single_mech_diffrax import *
from ampk_newmech_MA_single_diffrax import *

# import utils functions
sys.path.append("../")
from utils import *

# use 64 bit precision for Jax
jax.config.update('jax_enable_x64', True)

#############################
# def arg parsers to take inputs from the command line
##############################
def parse_args(raw_args=None):
    """ function to parse command line arguments
    """
    parser=argparse.ArgumentParser(description="Run GSA sampling and compute GSA indices.")
    parser.add_argument("-model", type=str, help="model to process.")
    parser.add_argument("-free_params", type=str, help="parameters to test")
    parser.add_argument("-model_info_file", type=str, help="JSON file with relevant info. Model params, initial conditions, and AMPKAR states.")
    parser.add_argument("-upper_mult", type=float, default=1e2, help="Multiplier for upper bound in GSA sampling. Defaults to 100")
    parser.add_argument("-lower_mult", type=float, default=1e-2, help="Multiplier for lower bound in GSA sampling. Defaults to 0.01.")
    parser.add_argument("-metab_params_file", type=str, help="Metabolism model parameters. Should be a JSON")
    parser.add_argument("-nsamples", type=int, default=256, help="Number of samples to draw in each parameter direction. Defaults to 256")
    parser.add_argument("-gsa_method", type=str, default="sobol", help="GSA method to use. Defaults to sobol. Options are sobol, morris, and hdmr.")
    parser.add_argument("-savedir", type=str, help="Path to save results. Defaults to current directory.", default="./")
    parser.add_argument("-tmax", type=float, default=1e3, help="Maximum time to run the simulation. Defaults to 1e3.")
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
    
    # random seed for reproducibility
    seed = np.random.default_rng(12345)

    # add savedir if it does not exist
    if not os.path.isdir(args.savedir):
        os.makedirs(args.savedir)

    ####################################################
    # set up model info and nominal parameters #
    ####################################################
    # Load JSON files with param, state, and initial condition info
    # states and initial conditions
    with open(args.model_info_file, 'r') as file:
           model_info = json.load(file)

    # unpack loaded model data dictionary
    y0 = jnp.array(list(model_info["init_conds"].values()))

    # get the names of the fixed parameters
    free_params = args.free_params.split(',')
    param_names = model_info['nominal_params'].keys()
    nominal_params = model_info['nominal_params']

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
    # Bounds and other info for the GSA #
    ############################################
    # define the bounds for the AMPK parameters
    bound_mults = np.array((args.lower_mult, args.upper_mult))
    bounds = [bound_mults*nominal_params[param] for param in free_params]

    # dictionary of the problem for SALib
    bounds = {'num_vars':len(free_params), 'names':free_params, 'bounds': bounds}


    ######################################################
    # # generate samples using specified method #
    ######################################################
    # use sobol sampling for hdmr since it is sampling agnostic
    if args.gsa_method in ['sobol', 'hdmr']:
        param_vals = sobol_samp.sample(bounds, args.nsamples, \
                                       calc_second_order=False, seed=seed)
    elif args.gsa_method == "morris":
        pass
        # TODO implement morris sampling

    np.save(args.savedir + args.model + '_param_vals_GSA.npy', np.array(param_vals))

    # Convert from parameter samples to full parameter sets, because we do not sample
    # all parameters in the model
    # ASSUME:
    # - all parameters are included in the nominals json
    # - the order of parameters in the nominals json is the order in 
    #       which parameters are expected by the RHS of the model
    # temp is a n_sample x n_params (free + fixed) matrix
    temp = np.empty(shape=(param_vals.shape[0], len(param_names))) # TODO check dims of paravals

    for i in range(param_vals.shape[0]):
        # copy the nominals dict
        temp_dict = nominal_params.copy()

        # loop over free params and get the ith sample of them
        # this should leave un-sampled params fixed in the dictionary
        for j, param in enumerate(free_params):
            temp_dict[param] = param_vals[i, j]

        # replace the ith row of temp with the values of the dictionary
        for j, key in enumerate(temp_dict.keys()):
            temp[i, j] = temp_dict[key]

    ######################################################
    # Set up solver
    ######################################################
    solve = jax.vmap(lambda params: solve_SS(rhs, rhs_stress, y0, params, tmax = args.tmax,
                                      rtol=args.rtol, atol=args.atol, 
                                      evnt_rtol=args.evnt_rtol, evnt_atol=args.evnt_atol, 
                                      pcoeff=args.pcoeff, icoeff=args.icoeff, dcoeff=args.dcoeff))


    # run the vmapped simulations
    tnow = time.time()
    sols = solve(temp)
    tend = time.time()

    # save model evals
    np.save(args.savedir + args.model + '_sols_GSA.npy', np.array(sols))

    print('Simulations took {} seconds'.format(tend-tnow))
    print('Completed {}'.format(args.model))

if __name__ == '__main__':
    main()
