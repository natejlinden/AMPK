# Nathaniel Linden
# script to test mass conservation in AMPK models

import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import diffrax as dfrx
import pandas as pd
import sys, json
import met_brewer as mb
import equinox as eqx

sys.path.append('../')
from plotting_helper_funcs import *

# import utils functions
from utils import *
import os
import glob

# add odes folder
sys.path.append('../odes/')

# use 64 bit precision for Jax
jax.config.update('jax_enable_x64', True)

plt.style.use('~/.matplotlib/stylelib/custom.mplstyle')

def assess_conservations(model, model_info_file, conservations, metab_params_file):

    # read in the model info
    # import the model
    exec('from ' + model + '_diffrax import *')

    ####################################################
    # set up model info and nominal parameters #
    ####################################################
    # Load JSON files with param, state, and initial condition info
    # states and initial conditions
    with open(model_info_file, 'r') as file:
           model_info = json.load(file)

    # unpack initial conditions and nominal parameters
    y0 = list(model_info["init_conds"].values())
    nominal_params = list(model_info['nominal_params'].values())

    # parameters for the metabolic model
    with open(metab_params_file, 'r') as file:
           metab_params = json.load(file)

    basal_params = list(metab_params["metab_params_basal"].values())
    stress_params = list(metab_params["metab_params_stress"].values())

    ###############################################
    #                   Model RHS                 #
    ################################################
    try:
        rhs = eval(model + '(' + ','.join(str(elm) for elm in basal_params) \
            + ')')
        rhs_stress = eval(model + '(' + ','.join(str(elm) for elm in stress_params) \
             + ')')
        rhs = dfrx.ODETerm(rhs)
        rhs_stress = dfrx.ODETerm(rhs_stress)
    except:
        print('Warning Model {} not found. Quitting.'.format(model))
        quit()

    ###############################################
    # Run the model to SS with the nominal params #
    ################################################
    SS_stress, _ = solve_SS(rhs, rhs_stress, y0, nominal_params)

    ###############################################
    # Check conservations                         #
    ################################################
    # dictionary for returning results
    results = {}
    cons_flag = True
    for conservation in conservations:
        name, indices, expected = conservation # unpack conservation tuple

        # check the conservation
        total = jnp.sum(SS_stress[jnp.array(indices)])

        # check if the conservation is satisfied
        if np.isclose(total, expected, rtol=1e-10, atol=1e-10):
            results[name] = True
            cons_flag *= True
        else:
            results[name] = (False, np.abs(total - expected))
            cons_flag *= False
    
    return results, cons_flag


def plot_conservations(model, model_info_file, conservations, metab_params_file):

    # read in the model info
    # import the model
    exec('from ' + model + '_diffrax import *')

    ####################################################
    # set up model info and nominal parameters #
    ####################################################
    # Load JSON files with param, state, and initial condition info
    # states and initial conditions
    with open(model_info_file, 'r') as file:
           model_info = json.load(file)

    # unpack initial conditions and nominal parameters
    y0 = list(model_info["init_conds"].values())
    nominal_params = list(model_info['nominal_params'].values())

    # parameters for the metabolic model
    with open(metab_params_file, 'r') as file:
           metab_params = json.load(file)

    basal_params = list(metab_params["metab_params_basal"].values())
    stress_params = list(metab_params["metab_params_stress"].values())

    ###############################################
    #                   Model RHS                 #
    ################################################
    try:
        rhs = eval(model + '(' + ','.join(str(elm) for elm in basal_params) \
            + ')')
        rhs_stress = eval(model + '(' + ','.join(str(elm) for elm in stress_params) \
             + ')')
        rhs = dfrx.ODETerm(rhs)
        rhs_stress = dfrx.ODETerm(rhs_stress)
    except:
        print('Warning Model {} not found. Quitting.'.format(model))
        quit()

    ###############################################
    # Run the model to SS with the nominal params #
    ################################################
    times = np.linspace(0, 1000, 1000)
    traj_stress, _ = solve_traj(rhs, rhs_stress, y0, nominal_params, times)

    ###############################################
    # plot conservations                         #
    ################################################
    for conservation in conservations:
        name, indices, expected = conservation # unpack conservation tuple

        # check the conservation
        total = jnp.sum(traj_stress[jnp.array(indices),:], 0)

        fig, ax = plt.subplots()
        ax.plot(times, total, label='Total', color='black', linewidth=2)
        ax.axhline(expected, color='red', linestyle='--', label='Expected')

        for idx in indices:
            ax.plot(times, traj_stress[idx,:], label='Species {}'.format(idx), linewidth=1.0)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Concentration')
        ax.set_title(name)
        ax.legend()

        fig.savefig('../../../results/test_conservations/' + model + '_' + name + '_conservation.png')


# import models
sys.path.append("../odes/")

def main():

    # path to results
    results_path = '../../../results/test_conservations/'

    # remove all files in the results_path
    files = glob.glob(results_path + '*')
    for f in files:
        os.remove(f)

    # shared metabolism model parameters
    metab_params_file = '../odes/metabolism_params_Coccimiglio.json'

    # dictionary of models
    # each value is a list of tuples with the name, indices, and the expected value for each conservation
    with open('./conservations.json', 'r') as file:
        models_conservations = json.load(file)
    
    for model in models_conservations.keys():
        print(model)
        model_info_file = '../odes/' + model + '.json'
    
        conservations = models_conservations[model]

        results = assess_conservations(model, model_info_file, conservations, metab_params_file)

        # save the results as json
        with open(results_path + model + '_conservation_results.json', 'w') as file:
            json.dump(results, file)

        # if conservation is not satisfied plot trajectories of the conservations
        if not results[1]:
            # plot the trajectories
            plot_conservations(model, model_info_file, conservations, metab_params_file)

if __name__ == '__main__':
    main()