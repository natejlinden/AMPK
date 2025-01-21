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
    parser.add_argument("--replot", action='store_true', help='Flag to replot precomputed sensitivity indices without computation or loading raw data.')
    args=parser.parse_args(raw_args)
    return args

def main(raw_args=None):
    args = parse_args()

    colors = mb.met_brew(name="Veronese", n=7)

    # list of models 
    models_free_params = { 
        # "ampk_Coccimiglio": {'free':["k6r","k7r","k8r","k9r","k10r","k11r","Km12",
        #                     "Km13","Km14","Km15","Km16","Km17","Km18","Km19",
        #                     "Vmaxkinase","VmaxkinaseATP","VmaxkinaseADP",
        #                     "VmaxkinaseAMP","Vmaxppase","VmaxppaseATP",
        #                     "VmaxppaseADP","VmaxppaseAMP","Km_pAMPK","k_pAMPK",
        #                     "Km_AMP_pAMPK","k_AMP_pAMPK","Km_ADP_pAMPK",
        #                     "k_ADP_pAMPK","Km_ATP_pAMPK","k_ATP_pAMPK"],
        #             'names':[r'$k_{6r}$',r'$k_{7r}$',r'$k_{8r}$',r'$k_{9r}$',
        #                      r'$k_{10r}$',r'$k_{11r}$',r'$K_{m12}$',r'$K_{m13}$',
        #                      r'$K_{m14}$',r'$K_{m15}$',r'$K_{m16}$',r'$K_{m17}$',
        #                      r'$K_{m18}$',r'$K_{m19}$',r'$V_{max,kinase}$',
        #                     r'$V_{max,kinase,ATP}$',r'$V_{max,kinase,ADP}$',
        #                     r'$V_{max,kinase,AMP}$',r'$V_{max,ppase}$',
        #                     r'$V_{max,ppase,ATP}$',r'$V_{max,ppase,ADP}$',
        #                     r'$V_{max,ppase,AMP}$',r'$K_{m,pAMPK}$',
        #                     r'$k_{pAMPK}$',r'$K_{m,AMP,pAMPK}$',r'$k_{AMP,pAMPK}$',
        #                     r'$K_{m,ADP,pAMPK}$',r'$k_{ADP,pAMPK}$',r'$K_{m,ATP,pAMPK}$',
        #                     r'$k_{ATP,pAMPK}$'],
        #             },
        "MA_single": {'free':["kOffAMP","kOffADP","kOffATP","kOffCaMKK","kPhosCaMKK",
                              "kOffLKB1","kPhosLKB1","kOffPP","kDephosPP","kOffAMPK",
                              "kPhosAMPK","kOffPP1","kDephosPP1"],
                    'names':[r'$k_{\text{OffAMP}}$',r'$k_{\text{OffADP}}$',
                             r'$k_{\text{OffATP}}$',r'$k_{\text{OffCaMKK}}$',
                             r'$k_{\text{PhosCaMKK}}$',r'$k_{\text{OffLKB1}}$',
                             r'$k_{\text{PhosLKB1}}$',r'$k_{\text{OffPP}}$',
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
                     'names':[r'$k_{\text{OffAMP}}$',r'$k_{\text{OffADP}}$',
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
        if not args.replot:
            sol_samples_basal = np.load(args.results_path  + model + '/' + model + '_sols_basal_GSA.npy')
            sol_samples_stressed = np.load(args.results_path + model + '/' + model + '_sols_stressed_GSA.npy')

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

            # compute qoi
            # use pAMPKAR_stressed/AMPKAR_stressed - pAMPKAR_basal/AMPKAR_basal
            # get relevant state indices
            state_names = list(model_info["init_conds"].keys())
            ampkar_idxs = [state_names.index(item) for item in model_info['ampkar_states']]
            pampkar_idxs = [state_names.index(item) for item in model_info['pampkar_states']]

            pAMPK_states = ["pAMPK", "AMP_pAMPK" , "ADP_pAMPK" , "ATP_pAMPK" , "PP_pAMPK" , "PP_ATP_pAMPK", "AMPKAR_AMP_pAMPK"]
            ampk_states = ["AMPK", "AMP_AMPK" , "ADP_AMPK" , "ATP_AMPK", "CaMKK_AMPK" , "CaMKK_AMP_AMPK" , "CaMKK_ADP_AMPK" , "CaMKK_ATP_AMPK" , "LKB1_AMP_AMPK" , "LKB1_ADP_AMPK"] + pAMPK_states
            ampk_idxs = [list(model_info['init_conds'].keys()).index(state) for state in ampk_states]
            pampk_idxs = [list(model_info['init_conds'].keys()).index(state) for state in pAMPK_states]

            # load data
            cyto_data, cyto_std, cyto_times = load_data('../../../Schmitt_et_al_2022_data/fig_2e_cyto.npz', 
                                        to_seconds=False, constant_std=False)

            times = np.linspace(0, 1800, 1000)

            def fit_to_cyto(arr):
                arr_at_cyto_times = np.interp(cyto_times, times, arr)
                sigma_inv = np.diag(1/cyto_std)
                res = arr_at_cyto_times - cyto_data
                return np.exp(-0.5*(res.T*sigma_inv*res))
            
            AMPKAR_stressed = sol_samples_stressed[: ,ampkar_idxs, :].sum(axis=1)
            AMPKAR_basal = sol_samples_basal[: ,ampkar_idxs].sum(axis=1)
            pAMPKAR_stressed = sol_samples_stressed[: ,pampkar_idxs, :].sum(axis=1)
            pAMPKAR_basal = sol_samples_basal[: ,pampkar_idxs].sum(axis=1)
            
            # calculate time to half max of pAMPKAR_stressed/AMPKAR_stressed
            def compute_half_max(arr):
                half_max = arr.max() / 2
                half_max_idx = np.argmin(np.abs(arr - half_max))
                return half_max_idx
            
            time_to_half_max_idx = np.apply_along_axis(compute_half_max, 1, pAMPKAR_stressed / AMPKAR_stressed)
            time_to_half_max = [times[idx] for idx in time_to_half_max_idx]
            time_to_half_max_delta_idx = np.apply_along_axis(compute_half_max, 1, (pAMPKAR_stressed / AMPKAR_stressed) - (pAMPKAR_basal / AMPKAR_basal).reshape((pAMPKAR_basal.shape[0],1)))
            time_to_half_max_delta = [times[idx] for idx in time_to_half_max_delta_idx]

            # cyto_data_fit = np.apply_along_axis(fit_to_cyto, 1, pAMPKAR_stressed / AMPKAR_stressed)

            # define dict of the qoi's -- there are multiple, so we need to run sobol analysis for each
            # the items in the dict are tuples, where the first entry is the vector of qoi's
            # the second entry is the name of the qoi
            qois = {
                "ratio":((pAMPKAR_stressed/AMPKAR_stressed).max(axis=1), r'$\frac{[\rm pAMPKAR]}{[\rm AMPKAR]}$'), # raw ratio
                "delta_ratio":((pAMPKAR_stressed/AMPKAR_stressed).max(axis=1) - (pAMPKAR_basal/AMPKAR_basal), r'$\Delta\frac{[\rm pAMPKAR]}{[\rm AMPKAR]}$'), # delta ratio
                "t_half": (time_to_half_max, r'$t_{\frac{1}{2},{\rm max}}$'), # time to half max
                "t_half_delta": (time_to_half_max_delta, r'$t_{\frac{1}{2},{\rm max}}$'), # delta time to half max
                "ratio_basal":((pAMPKAR_basal/AMPKAR_basal), r'basal $\frac{[\rm pAMPKAR]}{[\rm AMPKAR]}$'),
                # "cyto_data_fit":(cyto_data_fit, 'fit to cyto data')
            }
        qoi_names = ['ratio', 'delta_ratio', 't_half', 't_half_delta', 'ratio_basal']

        for qoi in qoi_names:
            if not args.replot:
                # unpack qoi tuple
                qoi_vals, qoi_name = qois[qoi]
    
                # plot histogram of qoi
                fig, ax = get_sized_fig_ax(1.0, 1.0)
                sns.histplot(qoi_vals, ax=ax, kde=True, stat='density', bins=30, 
                            line_kws={'linewidth': 1.0, 'linestyle':'--'},
                            color=colors[i])
                ax.set_xlabel(qoi_name)
                ax.set_ylabel('density')
                fig.savefig(args.fig_path + m_name + '_'+ qoi + '_hist.pdf', bbox_inches='tight')

                # analyze GSA
                Si_sobol = sobol_analyze.analyze(problem, qoi_vals, calc_second_order=True)
                print(Si_sobol)

                # covert to pandas dataframe for easier plotting
                sobol_df = pd.DataFrame({item:Si_sobol[item] for item in ['S1', 'S1_conf', 'ST', 'ST_conf']})
                sobol_df["param"] = free_params
                sobol_df["param_name"] = param_names
                sobol_df.to_csv(args.results_path + m_name + '_' + qoi + '_sobol_GSA.csv')
            else:
                sobol_df = pd.read_csv(args.results_path + m_name + '_' + qoi + '_sobol_GSA.csv')

            # # plot sobol indices
            # S1
            if model == "ampk_Coccimiglio":
                fig_width = 3.25
            else:
                fig_width = 1.75
            fig_height = 1.0
            fig, ax = get_sized_fig_ax(fig_width, fig_height)
            sorted = sobol_df.sort_values(by='S1', ascending=False)
            order = list(sorted["param"])

            sns.barplot(x='param', y='S1', data=sobol_df, 
                        ax=ax, order=order, color=colors[i])
            
            # add error bars
            bar_width = ax.patches[0].get_width()
            #Calculate offsets for number of hues provided
            offset = np.linspace(-1/2, 1/2, 1)*bar_width/2 #
            x_dict = dict((x_val,x_pos) for x_pos,x_val in list(enumerate(order)))
            #Map the x-position and offset of each record in the dataset
            x_values = np.array([x_dict[x] for x in sorted["param"]]);
            #Overlay the error bars onto plot
            ax.errorbar(x = x_values, y = sorted["S1"], yerr=sorted["S1_conf"], fmt='none', c= 'black', capsize = 2)

            ax.set_ylabel(r'$S_1$  ' + qoi_name)
            ax.set_xlabel('')
            ax.set_xticklabels(sorted['param_name'], rotation=45, ha='right', fontsize=8)
            fig.savefig(args.fig_path + m_name + '_' + qoi + '_S1.pdf', bbox_inches='tight')

            # ST
            fig, ax = get_sized_fig_ax(fig_width, fig_height)
            sorted = sobol_df.sort_values(by='ST', ascending=False)
            order = list(sorted["param"])

            sns.barplot(x='param', y='ST', data=sobol_df, 
                        ax=ax, order=order, color=colors[i])
            
            # add error bars
            bar_width = ax.patches[0].get_width()
            #Calculate offsets for number of hues provided
            offset = np.linspace(-1/2, 1/2, 1)*bar_width/2 #
            x_dict = dict((x_val,x_pos) for x_pos,x_val in list(enumerate(order)))
            #Map the x-position and offset of each record in the dataset
            x_values = np.array([x_dict[x] for x in sorted["param"]]);
            #Overlay the error bars onto plot
            ax.errorbar(x = x_values, y = sorted["ST"], yerr=sorted["ST_conf"], fmt='none', c= 'black', capsize = 2)

            ax.set_ylabel(r'$S_T$  ' + qoi_name)
            ax.set_xlabel('')
            ax.set_xticklabels(sorted['param_name'], rotation=45, ha='right', fontsize=8)
            # if ST is greater than 0.01, change the color of the xtick labels
            idxs = np.arange(0,len(sorted['ST']),1)[sorted['ST'] >=0.01]  # specify the indices of the xticks to change color
            for tick_label in ax.get_xticklabels():
                if ax.get_xticklabels().index(tick_label) in idxs:
                    tick_label.set_color('red')
            fig.savefig(args.fig_path + m_name + '_' + qoi + '_ST.pdf', bbox_inches='tight')
    
if __name__ == "__main__":
    main()













