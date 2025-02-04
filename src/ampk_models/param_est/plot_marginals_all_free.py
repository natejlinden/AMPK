import arviz as az
import pandas as pd
import json
import os

import numpy as np
import diffrax
import matplotlib.pyplot as plt
import matplotlib as mpl
import met_brewer as mb
import seaborn as sns
import jax
import sys

sys.path.append("../")
from utils import *
from plotting_helper_funcs import *

sys.path.append("../models/")

# tell jax to use 64bit floats
jax.config.update("jax_enable_x64", True)

plt.style.use('custom')
mpl.rcParams['figure.autolayout'] = True
colors = mb.met_brew(name="Egypt", n=3)

print(colors)

samplers = ["Nutpie"] # list of samplers used

cyto_color = colors[0]
lyso_color = colors[1]
mito_color = colors[2]

plot_settings = {
    "kOffAMP":{'name':r'$k_{\text{OffAMP}}$',
                'tick_labs':[1e-3, 1e-1],
                'xlim':[5e-4,2e-1]},
    "kOffADP":{'name':r'$k_{\text{OffADP}}$',
                'tick_labs':[1e-3, 1e-1],
                'xlim':[1e-4,1.0e1]},
    "kOffATP":{'name':r'$k_{\text{OffATP}}$',
                'tick_labs':[1e-3, 1e-1],
                'xlim':[1e-4,9.99e-2]},
    "kOffLKB1":{'name':r'$k_{\text{OffLKB1}}$',
                'tick_labs':[1e-3, 1e-1],
                'xlim':[9e-4,5e0]},
    "kPhosLKB1":{'name':r'$k_{\text{PhosLKB1}}$',
                'tick_labs':[1e-3, 1e-1],
                'xlim':[9e-4,5e0]},
    "kOffCaMKK":{'name':r'$k_{\text{OffCaMKK}}$',
                'tick_labs':[1e-3, 1e-1],
                'xlim':[9e-4,5e0]},
    "kPhosCaMKK":{'name':r'$k_{\text{PhosCaMKK}}$',
                'tick_labs':[1e-3, 1e-1],
                'xlim':[9e-4,5e0]},
    "kLKB1":{'name':r'$k_{\text{LKB1}}$', # TODO: update
                'tick_labs':[1e-3, 1e0],
                'xlim':[5e-3, 1e0]},
    "KmLKB1":{'name':r'$K_{m,\text{LKB1}}$', # TODO: update
                'tick_labs':[2e-2, 5e2],
                'xlim':[2e-2, 5e2]},
    "kCaMKK":{'name':r'$k_{\text{CaMKK}}$', # TODO: update
                'tick_labs':[1e-3, 1e0],
                'xlim':[5e-3, 1e0]},
    "KmCaMKK":{'name':r'$K_{m,\text{CaMKK}}$', # TODO: update
                'tick_labs':[2e-2, 5e2],
                'xlim':[2e-2, 5e2]},
    "kOffPP":{'name':r'$k_{\text{OffPP}}$',
                'tick_labs':[1e-3, 1e-1],
                'xlim':[1e-2,5e0]},
    "kDephosPP":{'name':r'$k_{\text{Dephos,PP}}$', # TODO: update or Remove?
                'tick_labs':[1e-3, 1e-1],
                'xlim':[5.8e-4,1.4e-1]},
    "alphaPP":{'name':r'$\alpha_{\text{PP}}$', # TODO: update
                'tick_labs':[9e-3, 1e1],
                'xlim':[9e-2,5e2]},
    "kPP":{'name':r'$k_{\text{PP}}$', # TODO: update
                'tick_labs':[1e-3, 1e-1],
                'xlim':[2e-3,4e0]},
    "KmPP":{'name':r'$K_{m,\text{PP}}$', # TODO: update
                'tick_labs':[1e-3, 1e-1],
                'xlim':[5e-2,6e5]},
    "kOffAMPK":{'name':r'$k_{\text{OffAMPK}}$',
                'tick_labs':[1e-3, 1e-1],
                'xlim':[1e-2,1e2]},
    "kPhosAMPK":{'name':r'$k_{\text{PhosAMPK}}$',
                 'tick_labs':[1e-3, 1e-1],
                    'xlim':[9e-4,5e0]},
    "betaAMP":{'name':r'$\beta_{\text{AMP}}$',
                'tick_labs':[1e-3, 1e-1],
                'xlim':[5e0,5e2]},
    "kOffPP1":{'name':r'$k_{\text{OffPP1}}$',
                'tick_labs':[1e-3, 1e-1],
                'xlim':[5e-3,2e2]},
    "kDephosPP1":{'name':r'$k_{\text{Dephos,PP1}}$',
                'tick_labs':[1e-3, 1e-1],
                'xlim':[1e-11,2e0]},
    }

models_free_params = { 
        "MA_single": {'free':["kOffAMP","kOffADP","kOffATP","kPhosLKB1",
                              "kOffLKB1","kPhosCaMKK","kOffCaMKK","kOffPP",
                              "kDephosPP","kPhosAMPK","kOffAMPK","kOffPP1",
                              "kDephosPP1"],
                      'info_file':'../models/MA_single.json'
                    },
        "MM_single":  {'free':["kOffAMP","kOffADP","kOffATP","kCaMKK","KmCaMKK",
                               "kLKB1","KmLKB1","kPP","KmPP"],
                       'info_file': '../models/MM_single.json'
                    },
        }

data_dir = '../../../results/param_est/all_free/'
save_dir_base = '../../../results/param_est/figs/all_free/marginals/'

## Load data
cyto_data, _, cyto_times = load_data('../../../Schmitt_et_al_2022_data/fig_2e_cyto.npz', 
                                     to_seconds=False, constant_std=False)
lyso_data, _, lyso_times = load_data('../../../Schmitt_et_al_2022_data/fig_2b_lyso.npz', 
                                     to_seconds=False, constant_std=False)
mito_data, _, mito_times = load_data('../../../Schmitt_et_al_2022_data/fig_2c_mito.npz', 
                                     to_seconds=False, constant_std=False)

for i, model in enumerate(models_free_params.keys()):
    
    print(f"Processing model {i+1}: {model}")

    for sampler in samplers:

        # load the idata
        idata_cyto = az.from_netcdf(data_dir + model + '_cyto_mcmc_samples_' + sampler + '.nc')
        idata_lyso = az.from_netcdf(data_dir + model + '_lyso_mcmc_samples_' + sampler + '.nc')
        idata_mito = az.from_netcdf(data_dir + model + '_mito_mcmc_samples_' + sampler + '.nc')

        ############ plot 1D marginals for each model param colored by compartment
        posterior_idata_cyto = idata_cyto.posterior.to_dataframe()
        posterior_idata_cyto['compartment'] = 'cyto'
        posterior_idata_lyso = idata_lyso.posterior.to_dataframe()
        posterior_idata_lyso['compartment'] = 'lyso'
        posterior_idata_mito = idata_mito.posterior.to_dataframe()
        posterior_idata_mito['compartment'] = 'mito'

        # Combine the three dataframes into one
        combined_posterior = pd.concat([posterior_idata_cyto, posterior_idata_lyso, posterior_idata_mito])
        # Reset index to avoid duplicate indices
        combined_posterior.reset_index(drop=True, inplace=True)
    
        # make the plots
        for i, param in enumerate(models_free_params[model]['free']): #combined_posterior.columns:
            if param != 'compartment':

                save_dir = save_dir_base + param + '/'
                if not os.path.exists(save_dir): # if the dir for the parameter doesn't exist, make it
                    os.makedirs(save_dir)

                fig, ax = get_sized_fig_ax(0.6, 0.4)
                sns.kdeplot(data=combined_posterior, x=param, hue='compartment', ax=ax,
                            fill=True, palette={'cyto':cyto_color, 'lyso':lyso_color, 'mito':mito_color},
                            legend=False, log_scale=(True, False), linewidth=1.0)
                ax.set_xlabel("", fontsize=8.0)
                ax.set_ylabel("", fontsize=8.0)
                # ax.set_xscale("log")
                # for label in ax.get_xticklabels() + ax.get_yticklabels():
                #     label.set_fontsize(8)
                ax.tick_params(axis='both', which='major', labelsize=8)

                leg = ax.legend(['Mitochondria', 'Lysosome', 'Cytoplasm'], loc='upper right', fontsize=8.0, bbox_to_anchor=(3, 1))
                leg.set_title("")
                export_legend(leg, save_dir_base + 'dist_legend.pdf')
                leg.remove()

                print(ax.get_xlim())

                ax.set_xlim(plot_settings[param]['xlim'])

                # print(param, ax.get_xlim())

                # we want at most two xtick labels
                # xticks = ax.get_xticks()
                # for idx, lab in enumerate(ax.get_xticklabels()):
                #     if xticks[idx] not in models_free_params[model]['free'][param]['tick_labs']:
                #         lab.set_visible(False)

                # ax.set_xlim(models_free_params[model]['free'][param]['xlim'])
    
                # if len(ax.get_xticklabels()) > 2:
                #     for idx, label in enumerate(ax.get_xticklabels()):
                #         if idx not in [0, len(ax.get_xticklabels())-1]:
                            # label.set_visible(False)

                # ax.ticklabel_format(style='plain', axis='x')
                # ax.ticklabel_format(style='plain', axis='y')

                ax.set_title(plot_settings[param]['name'], fontsize=10.0,
                             fontweight='normal', pad=0)

                plt.savefig(save_dir + f'{model}_{param}_{sampler}.pdf', 
                            transparent=True,
                            bbox_inches='tight')
                plt.close()
