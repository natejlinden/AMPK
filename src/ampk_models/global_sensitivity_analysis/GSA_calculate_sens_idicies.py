import numpy as np
from SALib.sample import sobol as sobol_samp
from SALib.sample import morris as morris_samp
from SALib.analyze import sobol as sobol_analyze
from SALib.analyze import morris as morris_analyze
from SALib.analyze.hdmr import analyze as hdmr_analyze
import os, sys, json

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import met_brewer as mb
import argparse

sys.path.append('../')
sys.path.append('../models/')
from plotting_helper_funcs import *
from utils import *

plt.style.use('~/.matplotlib/stylelib/custom.mplstyle')

def parse_args(raw_args=None):
    """ function to parse command line arguments
    """
    parser=argparse.ArgumentParser(description="Run GSA plotting.")
    # required parameters
    parser.add_argument("-results_path", type=str, default="../../../results/GSA/", help="Path to load/save raw results.")
    parser.add_argument("-fig_path", type=str, default="../../../results/GSA/figs/", help="Path to save figs.")
    args=parser.parse_args(raw_args)
    return args

def main(raw_args=None):
    args = parse_args()

    colors = mb.met_brew(name="Veronese", n=7)

    # list of models 
    models_free_params = { 
       "ampk_Coccimiglio": {'free':["k6r","k7r","k8r","k9r","k10r","k11r","Km12",
                           "Km13","Km14","Km15","Km16","Km17","Km18","Km19",
                           "Vmaxkinase","VmaxkinaseATP","VmaxkinaseADP",
                           "VmaxkinaseAMP","Vmaxppase","VmaxppaseATP",
                           "VmaxppaseADP","VmaxppaseAMP","Km_pAMPK","k_pAMPK",
                           "Km_AMP_pAMPK","k_AMP_pAMPK","Km_ADP_pAMPK",
                           "k_ADP_pAMPK","Km_ATP_pAMPK","k_ATP_pAMPK"],
                   'names':[r'$k_{6r}$',r'$k_{7r}$',r'$k_{8r}$',r'$k_{9r}$',
                            r'$k_{10r}$',r'$k_{11r}$',r'$K_{m12}$',r'$K_{m13}$',
                            r'$K_{m14}$',r'$K_{m15}$',r'$K_{m16}$',r'$K_{m17}$',
                            r'$K_{m18}$',r'$K_{m19}$',r'$V_{max,kinase}$',
                           r'$V_{max,kinase,ATP}$',r'$V_{max,kinase,ADP}$',
                           r'$V_{max,kinase,AMP}$',r'$V_{max,ppase}$',
                           r'$V_{max,ppase,ATP}$',r'$V_{max,ppase,ADP}$',
                           r'$V_{max,ppase,AMP}$',r'$K_{m,pAMPK}$',
                           r'$k_{pAMPK}$',r'$K_{m,AMP,pAMPK}$',r'$k_{AMP,pAMPK}$',
                           r'$K_{m,ADP,pAMPK}$',r'$k_{ADP,pAMPK}$',r'$K_{m,ATP,pAMPK}$',
                           r'$k_{ATP,pAMPK}$'],
                   },
       "MA_single": {'free':["kOffAMP","kOffADP","kOffATP","kOffCaMKK","kPhosCaMKK",
                             "kOffLKB1","kPhosLKB1","kOffPP","kDephosPP","kOffAMPK",
                             "kPhosAMPK","kOffPP1","kDephosPP1"],
                   'names':[r'$k_{\text{OffAMP}}$',r'$k_{\text{OffADP}}$',
                            r'$k_{\text{OffATP}}$',r'$k_{\text{OffCaMKK}}$',
                            r'k_{\text{PhosCaMKK}}',r'k_{\text{OffLKB1}}',
                            r'k_{\text{PhosLKB1}}',r'k_{\text{OffPP}}',
                            r'$k_{\text{DephosPP}}$',r'$k_{\text{OffAMPK}}$',
                            r'$k_{\text{PhosAMPK}}$',r'$k_{\text{OffPP1}}$',
                            r'$k_{\text{Dephos,PP1}}$']}, 
        "MM_single":  {'free':["kOffAMP","kOffADP","kOffATP","kCaMKK","KmCaMKK",
                               "kLKB1","KmLKB1","kPP","KmPP"],
                    'names':[r'$k_{\text{OffAMP}}$',r'$k_{\text{OffADP}}$',
                             r'$k_{\text{OffATP}}$',r'$k_{\text{PhosCaMKK}}$',
                             r'$K_{m,\text{CaMKK}}$',r'$k_{\text{PhosLKB1}}$',
                             r'$K_{\text{m,LKB1}}$',r'$k_{\text{DephosPP}}$',
                             r'$K_{\text{M,PP}}$'],
                    },
        "MA_nonessential": {'free':["kOffAMP","kOffADP","kOffATP","kOffCaMKK",
                                    "kPhosCaMKK","kOffLKB1","kPhosLKB1","kOffPP",
                                    "kDephosPP","kOffAMPK","kPhosAMPK","kOffPP1",
                                    "kDephosPP1","alphaLKB1","alphaPP","betaAMP"],
                    'names':[r'$k_{\text{OffAMP}}$',r'$k_{\text{OffADP}}$',
                             r'$k_{\text{OffATP}}$',r'$k_{\text{OffCaMKK}}$',
                             r'$k_{\text{PhosCaMKK}}$',r'$k_{\text{OffLKB1}}$',
                             r'$k_{\text{PhosLKB1}}$',r'$k_{\text{OffPP}}$',
                             r'$k_{\text{DephosPP}}$',r'$k_{\text{OffAMPK}}$',
                             r'$k_{\text{PhosAMPK}}$',r'$k_{\text{OffPP1}}$',
                             r'$k_{\text{Dephos,PP1}}$',r'$\alpha_{\text{LKB1}}$',
                             r'$\alpha_{\text{PP}}$',r'$\beta_{\text{AMP}}$'],
                    }, 
         "MM_nonessential":  {'free':["kOffAMP","kOffADP","kOffATP","kPhosCaMKK","KmCaMKK",
                                      "kPhosLKB1","KmLKB1","kDephosPP","KmPP","alphaLKB1",
                                      "alphaPP","betaAMP"],
                              'names':[r'k_{\text{OffAMP}}',r'k_{\text{OffADP}}',
                              r'$k_{\text{OffATP}}$',r'$k_{\text{PhosCaMKK}}$',
                              r'$K_{m,\text{CaMKK}}$',r'$k_{\text{PhosLKB1}}$',
                              r'$K_{\text{m,LKB1}}$',r'$k_{\text{DephosPP}}$',
                              r'$K_{\text{M,PP}}$',r'$\alpha_{\text{LKB1}}$',
                             r'$\alpha_{\text{PP}}$',r'$\beta_{\text{AMP}}$'],
                     }
        }

    # loop through each model and analyze GSA results
    for i, model in enumerate(models_free_params.keys()):
        print(model)
        m_name = model.split('_mech')[0] # get the model name w/o _mech
        # we need mech in the model to load GSA sampling results correctly

        # load results
        sol_samples_basal = np.load(args.results_path  + model + '/' + model + '_sols_basal_GSA.npy')
        sol_samples_stressed = np.load(args.results_path + model + '/' + model + '_sols_stressed_GSA.npy')
        param_samples = np.load(args.results_path + model + '/' + model + '_param_vals_GSA.npy')

        # Load JSON files with param, state, and initial condition info
        # states and initial conditions
        info_file = '../models/' + m_name + '.json'
        with open(info_file, 'r') as file:
            model_info = json.load(file)

        # free parameters and nominal values
        free_params = models_free_params[model]['free']
        nominal_params = model_info['nominal_params']
        param_names = models_free_params[model]['names']

        # define the bounds for the AMPK parameters
        bounds = [model_info['param_bounds'] for param in free_params]

        # dictionary of the problem for SALib
        problem = {'num_vars':len(free_params), 'names':free_params, 'bounds': bounds}

        #### compute qois
        # use pAMPKAR_stressed/AMPKAR_stressed - pAMPKAR_basal/AMPKAR_basal
        # get relevant state indices
        state_names = list(model_info["init_conds"].keys())
        ampkar_idxs = [state_names.index(item) for item in model_info['ampkar_states']]
        pampkar_idxs = [state_names.index(item) for item in model_info['pampkar_states']]
        
        AMPKAR_stressed = sol_samples_stressed[: ,ampkar_idxs, :].sum(axis=1)
        AMPKAR_basal = sol_samples_basal[: ,ampkar_idxs].sum(axis=1)
        pAMPKAR_stressed = sol_samples_stressed[: ,pampkar_idxs, :].sum(axis=1)
        pAMPKAR_basal = sol_samples_basal[: ,pampkar_idxs].sum(axis=1)
        
        # calculate time to half max of pAMPKAR_stressed/AMPKAR_stressed
        def compute_half_max(arr):
            half_max = arr.max() / 2
            half_max_idx = np.argmin(np.abs(arr - half_max))
            return half_max_idx
        
        times = np.linspace(0, 1800, 1000)
        
        time_to_half_max_idx = np.apply_along_axis(compute_half_max, 1, pAMPKAR_stressed / AMPKAR_stressed)
        time_to_half_max = [times[idx] for idx in time_to_half_max_idx]
        time_to_half_max_delta_idx = np.apply_along_axis(compute_half_max, 1, (pAMPKAR_stressed / AMPKAR_stressed) - (pAMPKAR_basal / AMPKAR_basal).reshape((pAMPKAR_basal.shape[0],1)))
        time_to_half_max_delta = [times[idx] for idx in time_to_half_max_delta_idx]

        # initial pAMPKAR rate
        # load the model
        try:
            exec('from ' + model + '_diffrax import *')
        except:
            print('Warning Model {} not found. Quitting.'.format(model))
            quit()

        # load metabolic parameters
        # parameters for the metabolic model
        with open('../models/metabolism_params_Coccimiglio.json', 'r') as file:
            metab_params = json.load(file)

        stress_params = list(metab_params["metab_params_stress"].values())
        rhs_stress = eval(model + '(' + ','.join(str(elm) for elm in stress_params) \
            + ')')

        initial_rhs = []
        for sample_idx in range(sol_samples_basal.shape[0]):
            # get the initial conditions for the current sample
            y0 = sol_samples_basal[sample_idx, :]
            # create an instance of the model
            # compute the initial rate
            params = model_info['nominal_params'].copy()
            for i, param in enumerate(free_params):
                params[param] = param_samples[sample_idx, i]
            dydt = rhs_stress(0, y0, list(params.values()))
            dydt =  np.sum([dydt[idx] for idx in pampkar_idxs])
            initial_rhs.append(dydt)

        # fits to the data
        cyto_data, cyto_std, cyto_times = load_data('../../../Schmitt_et_al_2022_data/fig_2e_cyto.npz', to_seconds=True)
        lyso_data, lyso_std, lyso_times = load_data('../../../Schmitt_et_al_2022_data/fig_2b_lyso.npz', to_seconds=True)
        mito_data, mito_std, mito_times = load_data('../../../Schmitt_et_al_2022_data/fig_2c_mito.npz', to_seconds=True)

        # fit to data
        ratio = pAMPKAR_stressed/AMPKAR_stressed

        # interpolate predictions to times that data are evaluated at
        interp_fn = lambda x, data_times: np.interp(data_times, times, x)
        interp_cyto = np.apply_along_axis(lambda x: interp_fn(x, cyto_times), 1, ratio)
        interp_lyso = np.apply_along_axis(lambda x: interp_fn(x, lyso_times), 1, ratio)
        interp_mito = np.apply_along_axis(lambda x: interp_fn(x, mito_times), 1, ratio)

        # define the loss function
        loss_fn = lambda x, x_true, sigma_inv: (x - x_true).T@sigma_inv@(x - x_true)

        cyto_loss = np.apply_along_axis(lambda x: loss_fn(x, cyto_data, np.diag(1/cyto_std)), 1, interp_cyto)
        lyso_loss = np.apply_along_axis(lambda x: loss_fn(x, lyso_data, np.diag(1/lyso_std)), 1, interp_lyso)
        mito_loss = np.apply_along_axis(lambda x: loss_fn(x, mito_data, np.diag(1/mito_std)), 1, interp_mito)
        total_loss = cyto_loss + lyso_loss + mito_loss

        # define dict of the qoi's -- there are multiple, so we need to run sobol analysis for each
        # the items in the dict are tuples, where the first entry is the vector of qoi's
        # the second entry is the name of the qoi
        qois = {
            "ratio":(pAMPKAR_stressed/AMPKAR_stressed).max(axis=1), # raw ratio
            "delta_ratio":(pAMPKAR_stressed/AMPKAR_stressed).max(axis=1) - (pAMPKAR_basal/AMPKAR_basal), # delta ratio
            "t_half": np.array(time_to_half_max), # time to half max
            "t_half_delta": np.array(time_to_half_max_delta), # delta time to half max
            "ratio_basal":pAMPKAR_basal/AMPKAR_basal, # basal ratio
            "initial_dpAPKAR_dt":np.array(initial_rhs), # initial rate of pAMPAKR
            "cyto_loss":np.array(cyto_loss), # fits to data
            "lyso_loss":np.array(lyso_loss),
            "mito_loss":np.array(mito_loss),
            "total_loss":np.array(total_loss)
        }

        # save to npz file
        np.savez(args.results_path +  m_name + '/'+ m_name + '_qois.npz', **qois)

        qoi_names = ['ratio', 'delta_ratio', 't_half', 't_half_delta', 'ratio_basal', 
                     'initial_dpAPKAR_dt', 'cyto_loss', 'lyso_loss', 'mito_loss', 
                     'total_loss']

        # loop over QoIs and compute Sens idxs
        for qoi in qoi_names:
            # unpack qoi tuple
            qoi_vals = qois[qoi]

            # analyze GSA
            Si_sobol = sobol_analyze.analyze(problem, qoi_vals, calc_second_order=False)
            # Si_sobol = hdmr_analyze(problem, param_samples, qoi_vals)

            # covert to pandas dataframe for easier plotting
            sobol_df = pd.DataFrame({item:Si_sobol[item] for item in ['S1', 'S1_conf', 'ST', 'ST_conf']})
            # sobol_df = pd.DataFrame({item:Si_sobol[item] for item in ['Term', 'Sa', 'Sa_conf', 'Sb', 'Sb_conf', 'S', 'S_conf', 'ST', 'ST_conf']})
            sobol_df["param"] = free_params
            sobol_df["param_name"] = param_names
            print(sobol_df)
            sobol_df.to_csv(args.results_path  +  m_name + '/'+  m_name + '_' + qoi + '_sobol_GSA.csv')
    
if __name__ == "__main__":
    main()













