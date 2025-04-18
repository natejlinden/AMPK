from os import environ
environ['OMP_NUM_THREADS'] = '1'

import numpy as np
from SALib.sample import sobol as sobol_samp
import os, sys, time, json, argparse
import argparse
from gsa_utils import * # GSA utility functions
import jax
import jax.numpy as jnp
import diffrax as dfrx

# import models
sys.path.append("../models/")
from MA_single_diffrax import *
from MM_single_diffrax import *
from MA_nonessential_diffrax import *
from MM_nonessential_diffrax import *
from MA_nonessential_all_diffrax import *
from MM_nonessential_all_diffrax import *

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
    # required parameters
    parser.add_argument("-model", type=str, help="Filename of model to process.")
    parser.add_argument("-free_params", type=str, help="Comma separated string of parameters to test.")
    parser.add_argument("-model_info_file", type=str, help="Path to JSON file with relevant info. Model params, initial conditions, and AMPKAR states.")
    # optional parameters
    parser.add_argument("-metab_params_file", type=str, help="Metabolism model parameters. Should be a JSON")
    parser.add_argument("-nsamples", type=int, default=256, help="Number of samples to draw in each parameter direction. Defaults to 256")
    parser.add_argument("-savedir", type=str, help="Path to save results. Defaults to current directory.", default="./")
    parser.add_argument("-tmax", type=float, default=1e3, help="Maximum time to run the simulation. Defaults to 1e3.")
    parser.add_argument("-rtol", type=float,default=1e-6)
    parser.add_argument("-atol", type=float,default=1e-6)
    parser.add_argument('-pcoeff', type=float, default=0, help='pcoeff for PID time stepper')
    parser.add_argument('-dcoeff', type=float, default=0, help='dcoeff for PID time stepper')
    parser.add_argument('-icoeff', type=float, default=1.0, help='icoeff for PID time stepper')
    parser.add_argument('-ca_stress', type=float, default=1.0, help='calcium stress level')
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
    y0 = list(model_info["init_conds"].values())
    
    # get the names of the fixed parameters
    free_params = args.free_params.split(',')
    param_names = list(model_info['nominal_params'].keys())
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
        print('Warning Model {} not loaded properly. Quitting.'.format(args.model))
        quit()

    ############################################
    # Bounds and other info for the GSA #
    ############################################
    # define the bounds for the AMPK parameters
    bound_dict = model_info['param_bounds']
    bounds = [bound_dict[param] for param in free_params]

    # dictionary of the problem for SALib
    bounds = {'num_vars':len(free_params), 'names':free_params, 'bounds': bounds}

    ######################################################
    # # generate samples using specified method #
    ######################################################
    # use sobol sampling
    param_vals = sobol_samp.sample(bounds, args.nsamples, \
                                    calc_second_order=False, seed=seed)

    np.save(args.savedir + args.model + '_param_vals_GSA.npy', np.array(param_vals))

    # Convert from parameter samples to full parameter sets, because we do not sample
    # all parameters in the model
    # ASSUME:
    # - all parameters are included in the nominals json
    # - the order of parameters in the nominals json is the order in 
    #       which parameters are expected by the RHS of the model
    # temp is a n_sample x n_params (free + fixed) matrix
    temp = np.empty(shape=(param_vals.shape[0], len(param_names)))

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
    # initial conditions without LKB1 or CaMKK
    ######################################################
    if 'MA' in args.model:
        y0_LKB1_KD = y0.copy()
        y0_CaMKK_KD = y0.copy()

        params_LKB1_KD = temp.copy() # do not need to change the params for MA models
        params_CaMKK_KD = temp.copy()
        params_LKB1_KD[:, param_names.index('kOnLKB1')] = 0
        params_LKB1_KD[:, param_names.index('kPhosLKB1')] = 0
        params_CaMKK_KD[:, param_names.index('kOnCaMKK')] = 0
        params_CaMKK_KD[:, param_names.index('kPhosCaMKK')] = 0
        
    elif "MM" in args.model:
        y0_LKB1_KD = y0.copy()
        y0_CaMKK_KD = y0.copy()

        # need to change the LKB1tot and CaMKKtot params for MM models
        params_LKB1_KD = temp.copy()
        params_CaMKK_KD = temp.copy()

        # knockdown LKB1 or CaMKK by setting total concentrations to 0
        params_LKB1_KD[:, param_names.index('kLKB1')] = 0
        params_LKB1_KD[:, param_names.index('LKB1tot')] = 0
        params_CaMKK_KD[:, param_names.index('kCaMKK')] = 0

    # get Ca index
    ca_index = list(model_info['init_conds'].keys()).index('Ca')

    ######################################################
    # Set up solver
    ######################################################
    times = np.linspace(0, args.tmax, 1000)
    solve = jax.vmap(lambda params: solve_traj_timeDepCaMKK(rhs, rhs_stress, y0, 
                                    jnp.array([args.ca_stress,]), ca_index, params, times,
                                    rtol=args.rtol, atol=args.atol, 
                                    pcoeff=args.pcoeff, icoeff=args.icoeff, dcoeff=args.dcoeff))
    
    solve_LKB1_KD = jax.vmap(lambda params: solve_traj_timeDepCaMKK(rhs, rhs_stress, y0_LKB1_KD, 
                                    jnp.array([args.ca_stress,]), ca_index, params, times,
                                    rtol=args.rtol, atol=args.atol, 
                                    pcoeff=args.pcoeff, icoeff=args.icoeff, dcoeff=args.dcoeff))
    
    solve_CaMKK_KD = jax.vmap(lambda params: solve_traj_timeDepCaMKK(rhs, rhs_stress, y0_CaMKK_KD, 
                                    jnp.array([args.ca_stress,]), ca_index, params, times,
                                    rtol=args.rtol, atol=args.atol,
                                    pcoeff=args.pcoeff, icoeff=args.icoeff, dcoeff=args.dcoeff))

    # run the vmapped simulations
    
    # If running on GPU, check if the expected array size is too large and if so, chunk the simulations
    # Check if JAX is running on CPU or GPU
    expected_array_size = len(times)*param_vals.shape[0]*8*3 # 8 bytes per float64
    # mult by 3 for the 3 sims

    if expected_array_size > 250000000:
        print('Warning: chunking simulations to avoid memory error.')

        # chunk the simulations
        n_chunks = int(np.ceil(expected_array_size/100000000))
        chunk_size = int(np.ceil(param_vals.shape[0]/n_chunks))

        # Initialize arrays to store the results
        all_sols_stressed = []
        all_sols_basal = []
        all_sols_stressed_LKB1_KD = []
        all_sols_stressed_CaMKK_KD = []

        # loop over the chunks
        for i in range(n_chunks):   
            tnow = time.time()
            if i == n_chunks-1:
                sols = solve(temp[i*chunk_size:])
                sols_LKB1_KD = solve_LKB1_KD(params_LKB1_KD[i*chunk_size:])
                sols_CaMKK_KD = solve_CaMKK_KD(params_CaMKK_KD[i*chunk_size:])
            else:
                sols = solve(temp[i*chunk_size:(i+1)*chunk_size])
                sols_LKB1_KD = solve_LKB1_KD(params_LKB1_KD[i*chunk_size:(i+1)*chunk_size])
                sols_CaMKK_KD = solve_CaMKK_KD(params_CaMKK_KD[i*chunk_size:(i+1)*chunk_size])
            tend = time.time()

            # append model evals to the lists
            all_sols_stressed.append(np.array(sols[0]))
            all_sols_basal.append(np.array(sols[1]))
            all_sols_stressed_LKB1_KD.append(np.array(sols_LKB1_KD[0]))
            all_sols_stressed_CaMKK_KD.append(np.array(sols_CaMKK_KD[0]))

            print('Simulations took {} seconds'.format(tend-tnow))
            print('Completed {} chunk {}'.format(args.model, i))

        # Concatenate all chunks into single arrays
        all_sols_stressed = np.concatenate(all_sols_stressed, axis=0)
        all_sols_basal = np.concatenate(all_sols_basal, axis=0)
        all_sols_stressed_LKB1_KD = np.concatenate(all_sols_stressed_LKB1_KD, axis=0)
        all_sols_stressed_CaMKK_KD = np.concatenate(all_sols_stressed_CaMKK_KD, axis=0)

        # Save the concatenated results
        np.save(args.savedir + args.model + '_sols_stressed_GSA.npy', all_sols_stressed)
        np.save(args.savedir + args.model + '_sols_basal_GSA.npy', all_sols_basal)
        np.save(args.savedir + args.model + '_sols_stressed_LKB1_KD_GSA.npy', all_sols_stressed_LKB1_KD)
        np.save(args.savedir + args.model + '_sols_stressed_CaMKK_KD_GSA.npy', all_sols_stressed_CaMKK_KD)
    else: # if not too big, just run all at once
        tnow = time.time()
        sols = solve(temp)
        sols_LKB1_KD = solve_LKB1_KD(params_LKB1_KD)
        sols_CaMKK_KD = solve_CaMKK_KD(params_CaMKK_KD)
        tend = time.time()

        # save model evals
        np.save(args.savedir + args.model + '_sols_stressed_GSA.npy', np.array(sols[0]))
        np.save(args.savedir + args.model + '_sols_basal_GSA.npy', np.array(sols[1]))
        np.save(args.savedir + args.model + '_sols_stressed_LKB1_KD_GSA.npy', np.array(sols_LKB1_KD[0]))
        np.save(args.savedir + args.model + '_sols_stressed_CaMKK_KD_GSA.npy', np.array(sols_CaMKK_KD[0]))

        print('Simulations took {} seconds'.format(tend-tnow))
        print('Completed {}'.format(args.model))

if __name__ == '__main__':
    main()