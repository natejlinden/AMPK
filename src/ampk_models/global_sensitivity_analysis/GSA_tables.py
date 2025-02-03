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
                    'names':[r'$k_{\rm Off,AMP}$',r'$k_{\rm Off,ADP}$',
                             r'$k_{\rm Off,ATP}$',r'$k_{\rm Off,CaMKK}$',
                             r'$k_{\rm Phos,CaMKK}$',r'$k_{\rm Off,LKB1}$',
                             r'$k_{\rm Phos,LKB1}$',r'$k_{\rm Off,PP}$',
                             r'$k_{\rm Dephos,PP}$',r'$k_{\rm Off,AMPK}$',
                             r'$k_{\rm Phos,AMPK}$',r'$k_{\rm Off,PP1}$',
                             r'$k_{\rm Dephos,PP1}$']}, 
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

    # loop through each model and analyze GSA results
    for i, model in enumerate(models_free_params.keys()):
        print(model)
        m_name = model.split('_mech')[0] # get the model name w/o _mech
        # we need mech in the model to load GSA sampling results correctly

        #  the second entry is the name of the qoi
        qoi_names = {
            "ratio":r'$\frac{[\rm pAMPKAR]}{[\rm AMPKAR]}$', # raw ratio
            "t_half": r'$t_{1/2}$', # time to half max
        }

        free_params = models_free_params[model]['free']
        param_names = models_free_params[model]['names']

        #############################
        # load sobol results to df
        ratio_df = pd.read_csv(args.results_path  +  m_name + '/'+  m_name + '_ratio_sobol_GSA.csv')
        t_half_df = pd.read_csv(args.results_path  +  m_name + '/'+  m_name + '_t_half_sobol_GSA.csv')

        #############################
        # update param names
        # fix param_names in the df
        for j, param in enumerate(ratio_df['param']):
            ratio_df.loc[j, 'param_name'] = param_names[free_params.index(param)]
            t_half_df.loc[j, 'param_name'] = param_names[free_params.index(param)]

        #############################
        # drop unwanted columns
        drop_cols = ['Unnamed: 0', 'S1', 'S1_conf']

        ratio_df = ratio_df.drop(columns=drop_cols)
        t_half_df = t_half_df.drop(columns=drop_cols)

        #############################
        # add a parameter function column
        param_function_dict = {
            'AMP binding': ['kOffAMP', 'k8f', 'k8r', 'k11f', 'k11r'],
            'ADP binding': ['kOffADP', 'k7f', 'k7r', 'k10f', 'k10r'],
            'ATP binding':['kOffATP', 'k6f', 'k6r', 'k9f', 'k9r'],
            'AMPK phos.': ['kOffCaMKK', 'kPhosCaMKK', 'KmCaMKK', 'kCaMKK','kOffLKB1', 
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

        # create a reverse lookup dictionary
        reverse_param_function_dict = {}
        for key, values in param_function_dict.items():
            for value in values:
                reverse_param_function_dict[value] = key

        # add a new column to the dataframe
        ratio_df['Function'] = ratio_df['param'].map(reverse_param_function_dict)
        t_half_df['Function'] = t_half_df['param'].map(reverse_param_function_dict)

        # drop param col
        ratio_df = ratio_df.drop(columns=['param'])
        t_half_df = t_half_df.drop(columns=['param'])

        #############################
        # add a column that is formatted as Mean ± SD
        ratio_df[r'Ratio ($\pm$ SD)'] = ratio_df.apply(lambda x: f"${x['ST']:.3f} \pm {x['ST_conf']:.3f}$", axis=1)
        t_half_df[r'Time-to-half max ($\pm$ SD)'] = t_half_df.apply(lambda x: f"${x['ST']:.3f} \pm {x['ST_conf']:.3f}$", axis=1)

        #############################
        # drop the ST and ST_conf columns
        ratio_df = ratio_df.drop(columns=['ST', 'ST_conf'])
        t_half_df = t_half_df.drop(columns=['ST', 'ST_conf'])

        #############################
        # merge the two dfs
        merged_df = ratio_df.merge(t_half_df, on=['param_name', 'Function'])

        # Rename the param_name column to Parameter
        merged_df = merged_df.rename(columns={'param_name': 'Parameter'})

        # Generate the LaTeX table
        latex_table = merged_df.to_latex(index=False, 
                                        caption="", 
                                        label="tab:kinetic_params_sd",
                                        column_format="l|c|c|c",
                                        bold_rows=True)

        # Output the LaTeX table to a .tex file (optional)
        fname = args.results_path  +  m_name + '/'+  m_name + '_ST_table.tex'
        with open(fname, 'w') as f:
            f.write(latex_table)

        # Display the LaTeX table string
        print(latex_table)

if __name__ == "__main__":
    main()













