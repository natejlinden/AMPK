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

sys.path.append('../')
from plotting_helper_funcs import *

plt.style.use('~/.matplotlib/stylelib/custom.mplstyle')

def main():
    # path to results
    results_path = '../../../results/GSA/'

    # path to save figures
    fig_path = '../../../figures/GSA/'

    #  lower and upper bounds for GSA sampling
    lower_mult = 1e-2
    upper_mult = 1e2

    colors = mb.met_brew(name="Veronese", n=7)


    # list of models 
    models_free_params = {
        "MA_single_mech": {'free':["kOffAMP","kOffADP","kOffATP","kOffCaMKK","kPhosCaMKK","kOffLKB1","kPhosLKB1","kOffPP","kDephosPP","kOffAMPK","kPhosAMPK","kOffPP1","kDephosPP1"],
                           'names':[r'$k_{\text{OffAMP}}$',r'$k_{\text{OffADP}}$',r'$k_{\text{OffATP}}$',r'$k_{\text{OffCaMKK}}$',r'$k_{\text{PhosCaMKK}}$',r'$k_{\text{OffLKB1}}$',r'$k_{\text{PhosLKB1}}$',r'$k_{\text{OffPP}}$',r'$k_{\text{DephosPP}}$',r'$k_{\text{OffAMPK}}$',r'$k_{\text{PhosAMPK}}$',r'$k_{OffPP1}}$',r'$k_{\text{Dephos,PP1}}$']}, 
        "MM_single_mech": ["kOffAMP","kOffADP","kOffATP","kPhosCaMKK","KmCaMKK","kPhosLKB1","KmLKB1","kDephosPP","KmPP"], 
        "newmech_MA_single": ["kOffAMP","kOffCaMKK","kPhosCaMKK","kOffLKB1","kPhosLKB1","kOffPP","kDephosPP","kOffAMPK","kPhosAMPK","kOffPP1","kDephosPP1","alpha","beta"]}
    
    # models_free_params = {
    #     "MA_double_mech": ["kOffAMP","kOffADP","kOffATP","kOffCaMKK","kPhosCaMKK","kOffLKB1","kPhosLKB1","kOffPP","kDephosPP","kOffAMPK","kPhosAMPK","kOffPP1","kDephosPP1"], 
    #     "MA_single_mech": ["kOffAMP","kOffADP","kOffATP","kOffCaMKK","kPhosCaMKK","kOffLKB1","kPhosLKB1","kOffPP","kDephosPP","kOffAMPK","kPhosAMPK","kOffPP1","kDephosPP1"], 
    #     "MM_double_mech": ["kOffAMP","kOffADP","kOffATP","kPhosCaMKK","KmCaMKK","kPhosLKB1","KmLKB1","kDephosPP","KmPP"], 
    #     "MM_single_mech": ["kOffAMP","kOffADP","kOffATP","kPhosCaMKK","KmCaMKK","kPhosLKB1","KmLKB1","kDephosPP","KmPP"], 
    #     "newmech_MA_single": ["kOffAMP","kOffCaMKK","kPhosCaMKK","kOffLKB1","kPhosLKB1","kOffPP","kDephosPP","kOffAMPK","kPhosAMPK","kOffPP1","kDephosPP1","alpha","beta"]}
    

    # loop through each model and analyze GSA results
    for i, model in enumerate(models_free_params.keys()):
        m_name = model.split('_mech')[0] # get the model name w/o _mech
        # we need mech in the model to load GSA sampling results correctly

        # load results
        param_samples = np.load(results_path + model + '_param_vals_GSA.npy')
        sol_samples = np.load(results_path + model + '_sols_GSA.npy')

        # Load JSON files with param, state, and initial condition info
        # states and initial conditions
        info_file = '../odes/' + m_name + '.json'
        with open(info_file, 'r') as file:
            model_info = json.load(file)

        # free parameters and nominal values
        free_params = models_free_params[model]
        nominal_params = model_info['nominal_params']
        param_names = list(nominal_params.keys())

        # define the bounds for the AMPK parameters
        bound_mults = np.array((lower_mult, upper_mult))
        bounds = [bound_mults*nominal_params[param] for param in free_params]
        # dictionary of the problem for SALib
        problem = {'num_vars':len(free_params), 'names':free_params, 'bounds': bounds}

        # compute qoi
        # use pAMPKAR_stressed/AMPKAR_stressed - pAMPKAR_basal/AMPKAR_basal
        # get relevant state indices
        state_names = list(model_info["init_conds"].keys())
        ampkar_idxs = [state_names.index(item) for item in model_info['ampkar_states']]
        pampkar_idxs = [state_names.index(item) for item in model_info['pampkar_states']]
        
        AMPKAR_stressed = sol_samples[0, : ,ampkar_idxs].sum(axis=0)
        AMPKAR_basal = sol_samples[1, : ,ampkar_idxs].sum(axis=0)
        pAMPKAR_stressed = sol_samples[0, : ,pampkar_idxs].sum(axis=0)
        pAMPKAR_basal = sol_samples[1, : ,pampkar_idxs].sum(axis=0)
        qoi = (pAMPKAR_stressed/AMPKAR_stressed) - (pAMPKAR_basal/AMPKAR_basal)

        # plot histogram of qoi
        fig, ax = get_sized_fig_ax(2.5, 2.5)
        sns.histplot(qoi, ax=ax, kde=False, stat='density', bins=30, 
                     line_kws={'linewidth': 1.0, 'linestyle':'--'},
                     color=colors[i])
        ax.set_xlabel(r'$\Delta pAMPKAR/AMPKAR_{tot}$')
        ax.set_ylabel('density')
        fig.savefig(fig_path + m_name + '_qoi_hist.pdf', bbox_inches='tight')

        # analyze GSA
        Si_sobol = sobol_analyze.analyze(problem, qoi, calc_second_order=False)

        # covert to pandas dataframe for easier plotting
        sobol_df = pd.DataFrame(Si_sobol)
        sobol_df["param"] = free_params
        sobol_df.to_csv(results_path + m_name + '_sobol_delta_pampkar_GSA.csv')

        # # plot sobol indices
        # S1
        fig, ax = get_sized_fig_ax(2.5, 1.25)
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

        ax.set_ylabel('first-order index \n $S_1$')
        ax.set_xlabel('')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        fig.savefig(fig_path + m_name + '_S1_delta_pamkar.pdf', bbox_inches='tight')

        # ST
        fig, ax = get_sized_fig_ax(2.5, 1.25)
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

        ax.set_ylabel('total-order index \n $S_T$')
        ax.set_xlabel('')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        fig.savefig(fig_path + m_name + '_ST_delta_pamkar.pdf', bbox_inches='tight')
    
if __name__ == "__main__":
    main()













