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
    parser=argparse.ArgumentParser(description="Create giant GSA table for supplement.")
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
        # "MA_single": {'free':["kOffAMP","kOffADP","kOffATP","kOffCaMKK","kPhosCaMKK",
        #                       "kOffLKB1","kPhosLKB1","kOffPP","kDephosPP","kOffAMPK",
        #                       "kPhosAMPK","kOffPP1","kDephosPP1"],
        #             'names':[r'$k_{OffAMP}$',r'$k_{OffADP}$',
        #                      r'$k_{OffATP}$',r'$k_{OffCaMKK}$',
        #                      r'$k_{PhosCaMKK}$',r'$k_{OffLKB1}$',
        #                      r'$k_{PhosLKB1}$',r'$k_{OffPP}$',
        #                      r'$k_{DephosPP}$',r'$k_{OffAMPK}$',
        #                      r'$k_{PhosAMPK}$',r'$k_{OffPP1}$',
        #                      r'$k_{DephosPP1}$']}, 
        #  "MM_single":  {'free':["kOffAMP","kOffADP","kOffATP","KmCaMKK",
        #                         "kLKB1","KmLKB1","kPP","KmPP"],
        #              'names':[r'$k_{\text{OffAMP}}$',r'$k_{\text{OffADP}}$',
        #                       r'$k_{\text{OffATP}}$',
        #                       r'$K_{m,\text{CaMKK}}$',r'$k_{\text{LKB1}}$',
        #                       r'$K_{\text{m,LKB1}}$',r'$k_{\text{PP}}$',
        #                       r'$K_{\text{m,PP}}$'],
        #              },
        #  "MA_nonessential": {'free':["kOffAMP","kOffADP","kOffATP","kOffCaMKK",
        #                              "kPhosCaMKK","kOffLKB1","kPhosLKB1","kOffPP",
        #                              "kDephosPP","kOffAMPK","kPhosAMPK","kOffPP1",
        #                              "kDephosPP1","alphaLKB1","alphaPP","betaAMP"],
        #              'names':[r'$k_{\text{OffAMP}}$',r'$k_{\text{OffADP}}$',
        #                       r'$k_{\text{OffATP}}$',r'$k_{\text{OffCaMKK}}$',
        #                       r'$k_{\text{PhosCaMKK}}$',r'$k_{\text{OffLKB1}}$',
        #                       r'$k_{\text{PhosLKB1}}$',r'$k_{\text{OffPP}}$',
        #                       r'$k_{\text{DephosPP}}$',r'$k_{\text{OffAMPK}}$',
        #                       r'$k_{\text{PhosAMPK}}$',r'$k_{\text{OffPP1}}$',
        #                       r'$k_{\text{Dephos,PP1}}$',r'$\alpha_{\text{LKB1}}$',
        #                       r'$\alpha_{\text{PP}}$',r'$\beta_{\text{AMP}}$'],
        #              }, 
        #  "MM_nonessential":  {'free':["kOffAMP","kOffADP","kOffATP","KmCaMKK",
        #                               "kLKB1","KmLKB1","kPP","KmPP","alphaLKB1",
        #                               "alphaPP","betaAMP"],
        #              'names':[r'$k_{\text{OffAMP}}$',r'$k_{\text{OffADP}}$',
        #                       r'$k_{\text{OffATP}}$',
        #                       r'$K_{m,\text{CaMKK}}$',r'$k_{\text{LKB1}}$',
        #                       r'$K_{\text{m,LKB1}}$',r'$k_{\text{PP}}$',
        #                       r'$K_{\text{M,PP}}$',r'$\alpha_{\text{LKB1}}$',
        #                       r'$\alpha_{\text{PP}}$',r'$\beta_{\text{AMP}}$'],
        #              },
        # "MA_nonessential_all": {'free':["kOffAMP","kOffADP","kOffATP","kOffCaMKK",
        #                         "kPhosCaMKK", "kOffLKB1","kPhosLKB1","kOffPP",
        #                         "kDephosPP","kOffAMPK", "kPhosAMPK","kOffPP1",
        #                         "kDephosPP1","alphaPP","betaAMP","betaLKB1","betaCaMKK"],
        #             'names':[r'$k_{OffAMP}$',r'$k_{OffADP}$',
        #                      r'$k_{OffATP}$',r'$k_{OffCaMKK}$',
        #                      r'$k_{PhosCaMKK}$',r'$k_{OffLKB1}$',
        #                      r'$k_{PhosLKB1}$',r'$k_{OffPP}$',
        #                      r'$k_{DephosPP}$',r'$k_{OffAMPK}$',
        #                      r'$k_{PhosAMPK}$',r'$k_{OffPP1}$',
        #                      r'$k_{DephosPP1}$', r'$\alpha_{PP}$',r'$\beta_{AMP}$',
        #                      r'$\beta_{LKB1}$',r'$\beta_{CaMKK}$']}, 
        "MM_nonessential_all":  {'free':["kOffAMP","kOffADP","kOffATP","KmCaMKK",
                                "kLKB1","KmLKB1","kPP","KmPP","alphaPP",
                                "betaAMPK","betaLKB1","betaCaMKK"],
                    'names':[r'$k_{\text{OffAMP}}$',r'$k_{\text{OffADP}}$',
                                r'$k_{\text{OffATP}}$',
                                r'$K_{m,\text{CaMKK}}$',r'$k_{\text{LKB1}}$',
                                r'$K_{\text{m,LKB1}}$',r'$k_{\text{PP}}$',
                                r'$K_{\text{M,PP}}$',r'$\alpha_{\text{PP}}$',
                                r'$\beta_{\text{AMP}}$',r'$\beta_{\text{LKB1}}$',
                                r'$\beta_{\text{CaMKK}}$']}
        }
    
    # param to function mapping
    param_function_dict = {
        'AMP binding': ['kOffAMP', 'k8f', 'k8r', 'k11f', 'k11r'],
        'ADP binding': ['kOffADP', 'k7f', 'k7r', 'k10f', 'k10r'],
        'ATP binding':['kOffATP', 'k6f', 'k6r', 'k9f', 'k9r'],
        'AMPK phos.': ['kOffCaMK', 'kPhosCaMKK', 'KmCaMKK', 'kCaMKK','kOffLKB1', 
                  'kPhosLKB1', 'KmLKB1', 'kLKB1', 'Km12', 'Km14', 'Km16', 'Km18', 
                  'Vmaxkinase', 'VmaxkinaseATP', 'VmaxkinaseADP', 'VmaxkinaseAMP', 
                  'alphaLKB1', 'betaLKB1', 'betaCaMKK'],
        'AMPK dephos.': ['kOffPP', 'kDephosPP', 'KmPP', 'kPP', 'Km13', 'Km15', 'Km17', 'Km19',
                'Vmaxppase', 'VmaxppaseATP', 'VmaxppaseADP', 'VmaxppaseAMP', 'alphaPP'],
        'AMPK kinase act.': ['kOffAMPK', 'kPhosAMPK', 'KmAMPK', 'kAMPK', 'Km_pAMPK','k_pAMPK', 
                 'Km_AMP_pAMPK', 'k_AMP_pAMPK', 'Km_ADP_pAMPK', 'k_ADP_pAMPK', 
                 'Km_ATP_pAMPK', 'k_ATP_pAMPK', 'betaAMP', 'betaAMPK'],
        'AMKPAR dephos.': ['kOffPP1', 'kDephosPP1', 'KmPP1', 'kPP1', 'Km_AMPKAR_PP', 'Vmax_AMPKAR_PP']
    } 

    # create a reverse lookup dictionary
    reverse_param_function_dict = {}
    for key, values in param_function_dict.items():
        for value in values:
            reverse_param_function_dict[value] = key

    #  the second entry is the name of the qoi
    qoi_names = {
        "ratio":r'$\frac{[\rm pAMPKAR]}{[\rm AMPKAR]}$', # raw ratio
        "t_half": r'$t_{{\rm 1/2}}$', # time to half max
        "ratio_LKB1_KD": r'$\frac{[\rm pAMPKAR]}{[\rm AMPKAR]} LKB1 KO$',
        "ratio_CaMKK_KD": r'$\frac{[\rm pAMPKAR]}{[\rm AMPKAR]} CaMKK KO$',
        "t_half_LKB1_KD": r'$t_{{\rm 1/2}}$ LKB1 KO',
        "t_half_CaMKK_KD": r'$t_{{\rm 1/2}}$ CaMKK KO',
        }

    # loop through each model and analyze GSA results
    for i, model in enumerate(models_free_params.keys()):
        print(model)

        # model specific info
        free_params = models_free_params[model]['free']
        param_names = models_free_params[model]['names']


        def preprocess_df(sobol_df, qoi):
            """ function to preprocess the sobol df
            Note this is defined in the loop so that it can access model specific info """

            # # add qoi column
            # sobol_df['qoi'] = qoi_names[qoi]

            # create a new column that is formatted as Mean ± SD
            sobol_df[qoi_names[qoi]] = sobol_df.apply(lambda x: f"${x['ST']:.3f} \pm {x['ST_conf']:.3f}$", axis=1)

            # drop unwanted columns
            drop_cols = ['Unnamed: 0', 'S1', 'S1_conf', 'ST', 'ST_conf']
            sobol_df.drop(columns=drop_cols, inplace=True)

            # add a function column
            sobol_df['function'] = sobol_df['param'].map(reverse_param_function_dict)
            sobol_df.drop(columns=['param'], inplace=True)
                          
            return sobol_df
        
        # Load and preprocess the sobol results
        file_name_prefix = args.results_path  +  model + '/'+  model + '_'
        result_df  = None

        for i, qoi in enumerate(qoi_names.keys()):
            temp_df = pd.read_csv(file_name_prefix + qoi + '_sobol_GSA.csv')
            temp_df = preprocess_df(temp_df, qoi)

            if i == 0:
                result_df = temp_df.copy()
                result_df = result_df.reindex(columns=['param_name', \
                            'function', qoi_names[list(qoi_names.keys())[0]]])
            else:
                result_df[qoi_names[qoi]] = temp_df[qoi_names[qoi]]

        latex_table = result_df.to_latex(index=False, 
                                        caption="", 
                                        label="tab:kinetic_params_sd",
                                        column_format="l|c|c|c",
                                        bold_rows=True)

        # Output the LaTeX table to a .tex file (optional)
        fname = args.results_path  +  model + '/'+  model + '_ST_table.tex'
        with open(fname, 'w') as f:
            f.write(latex_table)

        # Display the LaTeX table string
        print(latex_table)

if __name__ == "__main__":
    main()













