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
import preliz as pl

sys.path.append("../models/")

# tell jax to use 64bit floats
jax.config.update("jax_enable_x64", True)

plt.style.use('custom')
mpl.rcParams['figure.autolayout'] = True
colors = mb.met_brew(name="Egypt", n=3)


###############################################################################
# Settings and Info
###############################################################################
samplers = ["Pathfinder"] # list of samplers used

cyto_color = colors[0]

param_names = {
    "kOffAMP":r'$k_{\text{OffAMP}}$',
    "kOffADP":r'$k_{\text{OffADP}}$',
    "kOffATP":r'$k_{\text{OffATP}}$',
    "kOffCaMKK":r'$k_{\text{OffCaMKK}}$',
    "kPhosCaMKK":r'$k_{\text{PhosCaMKK}}$',
    "kOffLKB1":r'$k_{\text{OffLKB1}}$',
    "kPhosLKB1":r'$k_{\text{PhosLKB1}}$',
    "kCaMKK":r'$k_{\text{CaMKK}}$',
    "KmCaMKK":r'$K_{m,\text{CaMKK}}$',
    "kLKB1":r'$k_{\text{LKB1}}$',
    "KmLKB1":r'$K_{m,\text{LKB1}}$',
    "kOffPP":r'$k_{\text{OffPP}}$',
    "kDephosPP":r'$k_{\text{Dephos,PP}}$',
    "alphaPP":r'$\alpha_{\text{PP}}$',
    "kPP":r'$k_{\text{PP}}$',
    "KmPP":r'$K_{m,\text{PP}}$',
    "kOffAMPK":r'$k_{\text{OffAMPK}}$',
    "kPhosAMPK":r'$k_{\text{PhosAMPK}}$',
    "betaAMP":r'$\beta_{\text{AMP}}$',
    "betaAMPK":r'$\beta_{\text{AMPK}}$',
    "kOffPP1":r'$k_{\text{OffPP1}}$',
    "kDephosPP1":r'$k_{\text{Dephos,PP1}}$',
    "betaLKB1":r'$\beta_{\text{LKB1}}$',
    "betaCaMKK":r'$\beta_{\text{CaMKK}}$',
    "alphaLKB1":r'$\alpha_{\text{LKB1}}$'
}

models_free_params = { 
        "MA_single": {'free':["kOffAMP","kOffADP","kOffATP","kOffCaMKK",
                              "kPhosCaMKK","kOffLKB1","kPhosLKB1","kOffAMPK",
                              "kPhosAMPK"],
                      'info_file':'../models/MA_single.json'
                    },
        "MM_single":  {'free':["kOffAMP","kOffADP","kOffATP","KmCaMKK","kLKB1",
                               "KmLKB1","kPP"],
                       'info_file': '../models/MM_single.json'
                    },
        "MA_nonessential": {'free':["kOffAMP","kOffADP","kOffATP","kOffCaMKK",
                                    "kPhosCaMKK","kOffLKB1","kPhosLKB1","kOffPP",
                                    "kDephosPP","kOffAMPK","kPhosAMPK","kOffPP1",
                                    "kDephosPP1","alphaLKB1","betaAMP"],
                            'info_file': '../models/MA_nonessential.json',
                    },
        "MM_nonessential":  {'free':["kOffAMP","kOffADP","kOffATP","KmCaMKK","kLKB1",
                                     "KmLKB1","alphaLKB1","kPP","KmPP","alphaPP",
                                     "betaAMP"],
                    'info_file': '../models/MM_nonessential.json',
                   },
        "MA_nonessential_all" : {'free':["kOffAMP","kOffADP","kOffATP","kPhosCaMKK",
                                         "kOffLKB1","kPhosLKB1","kOffPP","kDephosPP"
                                         ,"kOffAMPK","kPhosAMPK","kOffPP1",
                                         "kDephosPP1","alphaPP","betaAMP",
                                         "betaLKB1"],
                                 'info_file': '../models/MA_nonessential_all.json',
                    },
        "MM_nonessential_all" : {'free':["kOffAMP","kOffADP","kOffATP","KmCaMKK",
                                         "kLKB1","KmLKB1","kPP","KmPP","betaAMPK",
                                         "betaLKB1","betaCaMKK","alphaPP"],
                                 'info_file': '../models/MM_nonessential_all.json',
                    },
        }

directories = {
    'WT_only':{
        'data_dir':'../../../results/param_est/WT_only/',
        'save_dir_base':'../../../results/param_est/WT_only/figs/marginals/'
    },
    'kinase_ko':{
        'data_dir':'../../../results/param_est/kinase_KO/std_dcr/',
        'save_dir_base':'../../../results/param_est/kinase_KO/std_dcr/figs/marginals/'
    }
}

plot_larger = {'MM_single':['kOffAMP','kOffADP','kOffATP'],}

# plot sizes
normal_width = 0.6
normal_height = 0.4
large_width = 1.35
large_height = 0.9

################################################################################
# plotting function
################################################################################
def plot_dist(param, idata, width, height, save_dir_base, save_dir, model, 
              sampler, model_info, name_suffix=''):
    """ func to plot posterior dist with prior overlayed """

    fig, ax = get_sized_fig_ax(width, height)
    sns.kdeplot(data=idata, x=param, ax=ax,
                fill=True, color=cyto_color,
                legend=False, log_scale=(False, False), 
                linewidth=1.0, clip=(0.0, np.inf))
    ax.set_xlabel("", fontsize=8.0)
    ax.set_ylabel("", fontsize=8.0)
    ax.tick_params(axis='both', which='major', labelsize=8)

    xlim = ax.get_xlim()

    #### add plot of the prior
    # vector of values along the x-axis to eval prior
    x = np.linspace(min(xlim[0], model_info['param_bounds'][param][0]*0.9), 
                    max(xlim[1], model_info['param_bounds'][param][1]*1.1), 1000)

    # Evaluate the lognormal PDF using preliz
    prior_pdf = pl.LogNormal(mu=model_info['prior_params'][param]['mu'], 
                            sigma=model_info['prior_params'][param]['sigma']).pdf(x)
    ax.plot(x, prior_pdf, linestyle='--', color='black', linewidth=1.0)
    ax.fill_between(x, prior_pdf, color='gray', alpha=0.3)

    # Ensure tick labels do not use scientific notation
    ax.ticklabel_format(style='plain', axis='both', useOffset=False)

    # Set the prior fill to be behind all other elements
    ax.collections[-1].set_zorder(0)

    # legend
    leg = ax.legend(['posterior', 'prior'], loc='upper right', 
                    fontsize=8.0, bbox_to_anchor=(3, 1))
    
    leg.set_title("")
    export_legend(leg, save_dir_base + 'dist_legend.pdf')
    leg.remove()

    ax.set_title(param_names[param], fontsize=10.0,
                fontweight='normal', pad=0)

    fig.savefig(save_dir + f'{model}_{param}_{sampler}{name_suffix}.pdf', 
                transparent=True,
                bbox_inches='tight')
    plt.close()

################################################################################
# main loop over conditions, models, and samplers
################################################################################
for condition in directories.keys():
    print(f"Plotting condition: {condition}")

    data_dir = directories[condition]['data_dir']
    save_dir_base = directories[condition]['save_dir_base']

    for i, model in enumerate(models_free_params.keys()):
        
        print(f"Processing model {i+1}: {model}")

        for sampler in samplers:

            # load the idata
            idata_cyto = az.from_netcdf(data_dir + model + '_cyto_mcmc_samples_' + sampler + '.nc')

            ############ plot 1D marginals for each model param colored by compartment
            posterior_idata_cyto = \
                idata_cyto.posterior.mean(dim='prediction_dim_1').squeeze('prediction_dim_0').to_dataframe()

            # load model info file
            with open('../models/' + model + '.json', 'r') as file:
                model_info = json.load(file)
        
            # make the plots
            for i, param in enumerate(models_free_params[model]['free']): 
                if param != 'compartment':

                    save_dir = save_dir_base + model + '/'
                    # if the dir for the parameter doesn't exist, make it
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)

                    # plot the dist
                    plot_dist(param, posterior_idata_cyto, normal_width, normal_height, 
                              save_dir_base, save_dir, model, sampler, model_info)
                    
                    # make a larger plot for some params
                    if model in plot_larger.keys() and param in plot_larger[model]:
                        plot_dist(param, posterior_idata_cyto, large_width, large_height, 
                                save_dir_base, save_dir, model, sampler, model_info,
                                name_suffix='_LARGER')

                    # if model in plot_larger.keys() and param in plot_larger[model]:
                    #     fig, ax = get_sized_fig_ax(1.35, 0.9)
                    #     sns.kdeplot(data=combined_posterior, x=param, hue='compartment', ax=ax,
                    #                 fill=True, palette={'cyto':cyto_color, 'lyso':lyso_color, 'mito':mito_color},
                    #                 legend=False, log_scale=(True, False), linewidth=1.0)
                    #     ax.set_xlabel("", fontsize=8.0)
                    #     ax.set_ylabel("", fontsize=8.0)
                    #     ax.tick_params(axis='both', which='major', labelsize=8)

                    #     leg = ax.legend(['Mitochondria', 'Lysosome', 'Cytoplasm'], loc='upper right', fontsize=8.0, bbox_to_anchor=(3, 1))
                    #     leg.set_title("")
                    #     export_legend(leg, save_dir_base + 'dist_legend.pdf')
                    #     leg.remove()
                    #     print(ax.get_xlim())

                    #     ax.set_xlim(plot_settings[param]['xlim'])

                    #     ax.set_title(plot_settings[param]['name'], fontsize=10.0,
                    #                 fontweight='normal', pad=0)

                    #     plt.savefig(save_dir + f'{model}_{param}_{sampler}_LARGER.pdf', 
                    #                 transparent=True,
                    #                 bbox_inches='tight')
                    #     plt.close()