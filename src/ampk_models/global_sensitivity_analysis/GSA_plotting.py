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
    parser.add_argument("-fig_path", type=str, default="../../../results/GSA/", help="Path to save figs.")
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
                    'names':[r'$k_{OffAMP}$',r'$k_{OffADP}$',
                             r'$k_{OffATP}$',r'$k_{OffCaMKK}$',
                             r'$k_{PhosCaMKK}$',r'$k_{OffLKB1}$',
                             r'$k_{PhosLKB1}$',r'$k_{OffPP}$',
                             r'$k_{DephosPP}$',r'$k_{OffAMPK}$',
                             r'$k_{PhosAMPK}$',r'$k_{OffPP1}$',
                             r'$k_{DephosPP1}$']}, 
         "MM_single":  {'free':["kOffAMP","kOffADP","kOffATP","kCaMKK","KmCaMKK",
                                "kLKB1","KmLKB1","kPP","KmPP"],
                     'names':[r'$k_{\text{OffAMP}}$',r'$k_{\text{OffADP}}$',
                              r'$k_{\text{OffATP}}$',r'$k_{\text{PhosCaMKK}}$',
                              r'$K_{m,\text{CaMKK}}$',r'$k_{\text{PhosLKB1}}$',
                              r'$K_{\text{m,LKB1}}$',r'$k_{\text{DephosPP}}$',
                              r'$K_{\text{m,PP}}$'],
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

    ST_dict = {}

    # loop through each model and analyze GSA results
    for i, model in enumerate(models_free_params.keys()):
        st_sub_dict = {}
        print(model)
        m_name = model.split('_mech')[0] # get the model name w/o _mech
        # we need mech in the model to load GSA sampling results correctly

        fig_path = args.fig_path + m_name + '/'

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

        # load qoi samples
        qoi_samples = np.load(args.results_path  +  m_name + '/'+  m_name + '_qois.npz')
        #  the second entry is the name of the qoi
        qoi_names = {
            "ratio":r'$\frac{[\rm pAMPKAR]}{[\rm AMPKAR]}$', # raw ratio
            "t_half": r'$t_{{\rm half-max}}$', # time to half max
            "delta_LKB1_KD": r'$\Delta \frac{[\rm pAMPKAR]}{[\rm AMPKAR]} LKB1 KO$',
            "delta_CaMKK_KD": 
        }

        for qoi in list(qoi_names.keys()):
            # unpack qoi tuple
            qoi_vals = qoi_samples[qoi]
            qoi_name = qoi_names[qoi]

            # plot histogram of qoi
            fig, ax = get_sized_fig_ax(1.0, 1.0)
            sns.histplot(qoi_vals, ax=ax, kde=True, stat='density', bins=30, 
                        line_kws={'linewidth': 1.0, 'linestyle':'--'},
                        color=colors[i])
            ax.set_xlabel(qoi_name)
            ax.set_ylabel('density')
            fig.savefig(fig_path  + m_name + '_'+ qoi + '_hist.pdf', bbox_inches='tight')

            # load the sesnitivity indices
            sobol_df = pd.read_csv(args.results_path  +  m_name + '/'+  m_name + '_' + qoi + '_sobol_GSA.csv')

            # fix param_names in the df
            for j, param in enumerate(sobol_df['param']):
                sobol_df.loc[j, 'param_name'] = param_names[free_params.index(param)]

            # # plot sobol indices
            # S1
            fig_width = 4.5
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
            ax.set_xticklabels(sorted['param_name'], rotation=90, fontsize=8)
            fig.savefig(fig_path + m_name + '_' + qoi + '_S1.pdf', bbox_inches='tight')
   
            # ST
            st_sub_dict[qoi] = dict(zip(sobol_df['param'], sobol_df['ST'])) #/sobol_df['ST'].sum()))
    
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
            ax.set_xticklabels(sorted['param_name'], rotation=90, fontsize=8)
            # if ST is greater than 0.01, change the color of the xtick labels
            idxs = np.arange(0,len(sorted['ST']),1)[sorted['ST'] >=0.01]  # specify the indices of the xticks to change color
            for tick_label in ax.get_xticklabels():
                if ax.get_xticklabels().index(tick_label) in idxs:
                    tick_label.set_color('red')
            fig.savefig(fig_path + m_name + '_' + qoi + '_ST.pdf', bbox_inches='tight')
        ST_dict[model] = st_sub_dict

    param_function_dict = {
        'AMP binding': ['kOffAMP', 'k8f', 'k8r', 'k11f', 'k11r'],
        'ADP binding': ['kOffADP', 'k7f', 'k7r', 'k10f', 'k10r'],
        'ATP binding':['kOffATP', 'k6f', 'k6r', 'k9f', 'k9r'],
        'AMPK phos.': ['kOffCaMK', 'kPhosCaMKK', 'KmCaMKK', 'kCaMKK','kOffLKB1', 
                  'kPhosLKB1', 'KmLKB1', 'kLKB1', 'Km12', 'Km14', 'Km16', 'Km18', 
                  'Vmaxkinase', 'VmaxkinaseATP', 'VmaxkinaseADP', 'VmaxkinaseAMP', 
                  'alphaLKB1'],
        'AMPK dephos.': ['kOffPP', 'kDephosPP', 'KmPP', 'kPP', 'Km13', 'Km15', 'Km17', 'Km19',
                'Vmaxppase', 'VmaxppaseATP', 'VmaxppaseADP', 'VmaxppaseAMP', 'alphaPP'],
        'AMPK kinase act.': ['kOffAMPK', 'kPhosAMPK', 'KmAMPK', 'kAMPK', 'Km_pAMPK','k_pAMPK', 
                 'Km_AMP_pAMPK', 'k_AMP_pAMPK', 'Km_ADP_pAMPK', 'k_ADP_pAMPK', 
                 'Km_ATP_pAMPK', 'k_ATP_pAMPK', 'betaAMP'],
        'AMKPAR dephos.': ['kOffPP1', 'kDephosPP1', 'KmPP1', 'kPP1', 'Km_AMPKAR_PP', 'Vmax_AMPKAR_PP']
    }
    # Make a heatmap of the ST values
    for qoi in list(qoi_names.keys()):
        tmp = {}
        data_to_plot = np.zeros((len(ST_dict.keys()), len(param_function_dict.keys())))
        for i, model in enumerate(ST_dict.keys()):
            tmp[model] = ST_dict[model][qoi]

            for j, param_func in enumerate(param_function_dict.keys()):
                for param in param_function_dict[param_func]:
                    if param in tmp[model].keys(): # if that parameter is in the model
                        data_to_plot[i,j] += tmp[model][param] # add ST value to the data_to_plot entry
        
        # anything that is still 0, set to np.nan
        data_to_plot[data_to_plot == 0] = np.nan
        
        # fig, ax = get_sized_fig_ax(5.0, 2.0)
        if np.any(data_to_plot > 1.5):
            vmax=1.5
        else:
            vmax=np.nanmax(data_to_plot)
        norm = mpl.colors.Normalize(vmin=0, vmax=vmax)

        fig, ax = plt.subplots(figsize=(5.0, 1.5))
        im, cbar = heatmap(data_to_plot, list(ST_dict.keys()), list(param_function_dict.keys()), 
                        ax=ax, cmap="Blues", cbar_kw={'location':'right', 'pad':0.02},
                        cbarlabel=r'$S_T$:  ' + qoi_names[qoi], aspect='auto', norm=norm)
        # annotate
        texts = annotate_heatmap(im, valfmt="{x:.2f}", fontsize=8.0)

        # tick labels
        ax.xaxis.set_ticks_position('bottom')  # Set ticks at the bottom
        ax.xaxis.set_label_position('bottom') 
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10.0)
        ax.set_yticklabels(['Coccimiglio et al. 2020', 'Mass action (MA)', 'Michealis Menten (MM)',  'MA - nonessential', 'MM - nonessential'], fontsize=10.0)

        fig.savefig(args.fig_path + 'ST_heatmap_' + qoi + '.pdf', bbox_inches='tight', transparent=True)
    
if __name__ == "__main__":
    main()













