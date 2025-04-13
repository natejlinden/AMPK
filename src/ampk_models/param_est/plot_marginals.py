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
from scipy.stats import kruskal


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

samplers = ["ADVI"]#,"Nutpie"] #, "NUTS"] # list of samplers used
n_trajectories = 10 # number of trajectories to add to PPC plot

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
    "kPhosCaMKK":{'name':r'$k_{\text{PhosCaMKK}}$',
                'tick_labs':[1e-3, 1e-1],
                'xlim':[9e-4,5e0]},
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
        # "ampk_Coccimiglio": {'free':["Km_ADP_pAMPK","k_ADP_pAMPK","Km16",
        #                              "Km_AMP_pAMPK","VmaxkinaseADP","k10r",
        #                              "Km_pAMPK","Km13","Km_ATP_pAMPK","k_AMP_pAMPK",
        #                              "Km17","Vmaxppase","k_ATP_pAMPK","Km15","k11r",
        #                              "k9r","Km14","VmaxppaseATP","Km18","k7r","Km12",
        #                              "k_pAMPK"],
        #                     'names':[r'$K_{m,ADP,pAMPK}$',r'$k_{ADP,pAMPK}$',r'$K_{m16}$',
        #                              r'$K_{m,AMP,pAMPK}$', r'$V_{max,kinase,ADP}$',
        #                              r'$k_{10r}$',r'$K_{m,pAMPK}$',r'$K_{m13}$',
        #                              r'$K_{m,ATP,pAMPK}$',r'$K_{m,AMP,pAMPK}$',r'$K_{m17}$',r'$V_{max,ppase}$',
        #                              r'$k_{ATP,pAMPK}$',r'$K_{m15}$',r'$k_{11r}$',r'$k_{9r}$',
        #                              r'$K_{m14}$',r'$V_{max,ppase,ATP}$',r'$K_{m18}$',
        #                              r'$k_{7r}$',r'$K_{m12}$',r'$k_{pAMPK}$'],
        #                     'info_file': '../models/ampk_Coccimiglio.json',
        #                     },
        # "MA_single": {'free':["kOffAMP", "kOffADP","kOffATP","kOffAMPK", 
        #                       "kPhosAMPK", "kOffPP1", "kDephosPP1"],
        #               'info_file':'../models/MA_single.json'
        #             },
        "MM_single":  {'free':["kOffAMP","kOffADP","kOffATP"],
                       'info_file': '../models/MM_single.json'
                    },
        # "MA_nonessential": {'free':["kOffAMP","kOffADP","kOffATP","kPhosCaMKK",
        #                             "kOffPP","alphaPP","kOffAMPK","kPhosAMPK",
        #                             "betaAMP","kOffPP1","kDephosPP1"],
        #                     'info_file': '../models/MA_nonessential.json',
        #             },
        # "MM_nonessential":  {'free':["kOffAMP","kOffADP","kOffATP","kCaMKK",
        #                              "KmCaMKK","kPP","KmPP","alphaPP","betaAMP"],
        #             'info_file': '../models/MM_nonessential.json',
        #            }
        }

data_dir = '../../../results/param_est/'
save_dir_base = '../../../results/param_est/figs/marginals/'

## Load data
cyto_data, _, cyto_times = load_data('../../../Schmitt_et_al_2022_data/fig_2e_cyto.npz', 
                                     to_seconds=False, constant_std=False)
lyso_data, _, lyso_times = load_data('../../../Schmitt_et_al_2022_data/fig_2b_lyso.npz', 
                                     to_seconds=False, constant_std=False)
mito_data, _, mito_times = load_data('../../../Schmitt_et_al_2022_data/fig_2c_mito.npz', 
                                     to_seconds=False, constant_std=False)

kruskal_wallis_results = {sampler:{model:{} for model in models_free_params.keys()} \
                        for sampler in samplers}
pariwise_KLs = {sampler:{model:{} for model in models_free_params.keys()} \
                        for sampler in samplers}

for i, model in enumerate(models_free_params.keys()):
    
    print(f"Processing model {i+1}: {model}")

    for sampler in samplers:

        # load the idata
        idata_cyto = az.from_netcdf(data_dir + model + '_cyto_mcmc_samples_' + sampler + '.nc')
        idata_lyso = az.from_netcdf(data_dir + model + '_lyso_mcmc_samples_' + sampler + '.nc')
        idata_mito = az.from_netcdf(data_dir + model + '_mito_mcmc_samples_' + sampler + '.nc')

        ############ plot 1D marginals for each model param colored by compartment
        posterior_idata_cyto = \
            idata_cyto.posterior.mean(dim='prediction_dim_1').squeeze('prediction_dim_0').to_dataframe()
        posterior_idata_cyto['compartment'] = 'cyto'
        posterior_idata_lyso = \
            idata_lyso.posterior.mean(dim='prediction_dim_1').squeeze('prediction_dim_0').to_dataframe()
        posterior_idata_lyso['compartment'] = 'lyso'
        posterior_idata_mito = \
            idata_mito.posterior.mean(dim='prediction_dim_1').squeeze('prediction_dim_0').to_dataframe()
        posterior_idata_mito['compartment'] = 'mito'

        # Combine the three dataframes into one
        combined_posterior = pd.concat([posterior_idata_cyto, posterior_idata_lyso, posterior_idata_mito])
        # Reset index to avoid duplicate indices
        combined_posterior.reset_index(drop=True, inplace=True)

        plot_larger = {'MM_single':['kOffAMP','kOffADP','kOffATP'],}
        plot_ratios = {'MM_single':[('kOffAMP','kOffADP'),('kOffAMP','kOffATP'), ('kOffADP','kOffATP')],}
    
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
                ax.tick_params(axis='both', which='major', labelsize=8)

                leg = ax.legend(['Mitochondria', 'Lysosome', 'Cytoplasm'], loc='upper right', fontsize=8.0, bbox_to_anchor=(3, 1))
                leg.set_title("")
                export_legend(leg, save_dir_base + 'dist_legend.pdf')
                leg.remove()
                print(ax.get_xlim())

                ax.set_xlim(plot_settings[param]['xlim'])

                ax.set_title(plot_settings[param]['name'], fontsize=10.0,
                             fontweight='normal', pad=0)

                plt.savefig(save_dir + f'{model}_{param}_{sampler}.pdf', 
                            transparent=True,
                            bbox_inches='tight')
                plt.close()

                if model in plot_larger.keys() and param in plot_larger[model]:
                    fig, ax = get_sized_fig_ax(1.35, 0.9)
                    sns.kdeplot(data=combined_posterior, x=param, hue='compartment', ax=ax,
                                fill=True, palette={'cyto':cyto_color, 'lyso':lyso_color, 'mito':mito_color},
                                legend=False, log_scale=(True, False), linewidth=1.0)
                    ax.set_xlabel("", fontsize=8.0)
                    ax.set_ylabel("", fontsize=8.0)
                    ax.tick_params(axis='both', which='major', labelsize=8)

                    leg = ax.legend(['Mitochondria', 'Lysosome', 'Cytoplasm'], loc='upper right', fontsize=8.0, bbox_to_anchor=(3, 1))
                    leg.set_title("")
                    export_legend(leg, save_dir_base + 'dist_legend.pdf')
                    leg.remove()
                    print(ax.get_xlim())

                    ax.set_xlim(plot_settings[param]['xlim'])

                    ax.set_title(plot_settings[param]['name'], fontsize=10.0,
                                fontweight='normal', pad=0)

                    plt.savefig(save_dir + f'{model}_{param}_{sampler}_LARGER.pdf', 
                                transparent=True,
                                bbox_inches='tight')
                    plt.close()

        if model in plot_ratios.keys():
            combined_posterior_ratio = combined_posterior.copy()
            for i, (param1, param2) in enumerate(plot_ratios[model]):
                # add column with ratio
                combined_posterior_ratio[param1 + '/' + param2] = combined_posterior_ratio[param1] / combined_posterior_ratio[param2]

            # drop the original columns
            combined_posterior_ratio = combined_posterior_ratio.drop(columns=models_free_params[model]['free']+['prediction', 'prediction_dim_0'])

    
            
            # fig, ax = get_sized_fig_ax(1.35, 0.9)
            g = sns.pairplot(combined_posterior_ratio, hue='compartment',
                         palette={'cyto':cyto_color, 'lyso':lyso_color, 'mito':mito_color})
            
            g.savefig(save_dir_base + f'{model}_ratio_{sampler}.pdf',
                        transparent=True,
                        bbox_inches='tight')
                # sns.kdeplot(data=combined_posterior, x=param1 + '/' + param2, hue='compartment', ax=ax,
                #             fill=True, palette={'cyto':cyto_color, 'lyso':lyso_color, 'mito':mito_color},
                #             legend=False, log_scale=(True, False), linewidth=1.0)
                # ax.set_xlabel("", fontsize=8.0)
                # ax.set_ylabel("", fontsize=8.0)
                # ax.tick_params(axis='both', which='major', labelsize=8)

                # leg = ax.legend(['Mitochondria', 'Lysosome', 'Cytoplasm'], loc='upper right', fontsize=8.0, bbox_to_anchor=(3, 1))
                # leg.set_title("")
                # export_legend(leg, save_dir_base + 'dist_legend.pdf')
                # leg.remove()
                # print(ax.get_xlim())


                # # ax.set_xlim(plot_settings[param]['xlim'])

                # # ax.set_title(plot_settings[param]['name'], fontsize=10.0,
                # #             fontweight='normal', pad=0)

                # plt.savefig(save_dir + f'{model}_{param1 + '_' + param2}_{sampler}.pdf', 
                #             transparent=True,
                #             bbox_inches='tight')
                # plt.close()
                


                

                ############## stat sig. diff. ############
                # Perform Kruskal-Wallis test to dermine if there are significant differences between compartments
                data = {
                    'cyto': posterior_idata_cyto[param].values.tolist(),
                    'lyso': posterior_idata_lyso[param].values.tolist(),
                    'mito': posterior_idata_mito[param].values.tolist()
                }

                # Convert data into DataFrame
                df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in data.items()]))

                # Perform Kruskal-Wallis Test
                groups = [df[col].dropna() for col in df.columns]  # Ensure no NaNs
                statistic, p_value = kruskal(*groups)

                kruskal_wallis_results[sampler][model][param] = (statistic, p_value)

                ############## Pairwise KL divergence analysis ############
                # Calculate KL divergence for each pair of compartments
                kl_cyto_lyso = kl_divergence_knn(
                    posterior_idata_cyto[param].values[:, np.newaxis],
                    posterior_idata_lyso[param].values[:, np.newaxis],
                    k=4 
                )

                kl_cyto_mito = kl_divergence_knn(
                    posterior_idata_cyto[param].values[:, np.newaxis],
                    posterior_idata_mito[param].values[:, np.newaxis],
                    k=4 
                )

                kl_lyso_mito = kl_divergence_knn(
                    posterior_idata_lyso[param].values[:, np.newaxis],
                    posterior_idata_mito[param].values[:, np.newaxis],
                    k=4 
                )

                pariwise_KLs[sampler][model][param] = {
                    'cyto_lyso': kl_cyto_lyso,      
                    'cyto_mito': kl_cyto_mito,
                    'lyso_mito': kl_lyso_mito
                }

row_order = ['kOffAMP', 'kOffADP', 'kOffATP', 'kPhosCaMKK','kCaMKK', 'KmCaMKK',
              'kOffPP', 'kPP', 'KmPP', 'alphaPP', 'kOffAMPK', 'kPhosAMPK',
              'betaAMP', 'kOffPP1', 'kDephosPP1']

kruskal_df = pd.DataFrame(kruskal_wallis_results['ADVI'])

def tup_second_item(tup):
    if isinstance(tup, tuple):
        return tup[1]
    return tup  # Return the original value if it's not a tuple

kruskal_df = kruskal_df.map(tup_second_item)
kruskal_df = kruskal_df.loc[row_order] # reorder rows
kruskal_df = kruskal_df.reset_index()
kruskal_df = kruskal_df.rename(columns={'index': 'Parameter'})

latex_table = kruskal_df.to_latex(index=False, 
                                        caption="", 
                                        label="tab:marginals_kruskal",
                                        column_format="c|c|c|c|c",
                                        bold_rows=True,
                                        na_rep="--",
                                        float_format="%.3f")

print(latex_table)


##### generate table of average pairwise KL divergences #####
kl_df = pd.DataFrame(pariwise_KLs['ADVI']) # data frame with NaN entires and dicts w/ pariwise KLs

# function to compute mean over values in the dicts
def dict_mean(d):
    if isinstance(d, dict):
        return np.mean(list(d.values()))
    return d  # Return the original value if it's not a dictionary

kl_df = kl_df.map(dict_mean)
kl_df = kl_df.loc[row_order] # reorder rows
kl_df = kl_df.reset_index()
kl_df = kl_df.rename(columns={'index': 'Parameter'})

latex_table = kl_df.to_latex(index=False, 
                                        caption="", 
                                        label="tab:marginals_KL_div",
                                        column_format="c|c|c|c|c",
                                        bold_rows=True,
                                        na_rep="--",
                                        float_format="%.3f")

print(latex_table)