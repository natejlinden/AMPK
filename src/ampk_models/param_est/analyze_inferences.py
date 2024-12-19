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
colors = mb.met_brew(name="Veronese", n=7)

sampler = "NUTS" # can be ADVI or NUTS


models_free_params = { 
        # "ampk_Coccimiglio": {'free':["Km_ADP_pAMPK","k_ADP_pAMPK","Km16",
        #                              "Km_AMP_pAMPK","VmaxkinaseADP","k10r",
        #                              "Km_pAMPK","Km13","Km_ATP_pAMPK","k_AMP_pAMPK",
        #                              "Km17","Vmaxppase","k_ATP_pAMPK","Km15","k11r",
        #                              "k9r","Km14","VmaxppaseATP","Km18","k7r","Km12",
        #                              "k_pAMPK"],
        #                     'names':[r'$K_{m,ADP,pAMPK}$',r'$k_{ADP,pAMPK}$',r'$K_{m16}$',
        #                              r'$K_{m,AMP,pAMPK}$',r'$V_{max,kinase,ADP}$',
        #                              r'$k_{10r}$',r'$K_{m,pAMPK}$',r'$K_{m13}$',
        #                              r'$K_{m,ATP,pAMPK}$',r'$K_{m17}$',r'$V_{max,ppase}$',
        #                              r'$k_{ATP,pAMPK}$',r'$K_{m15}$',r'$k_{11r}$',r'$k_{9r}$',
        #                              r'$K_{m14}$',r'$V_{max,ppase,ATP}$',r'$K_{m18}$',
        #                              r'$k_{7r}$',r'$K_{m12}$',r'$k_{pAMPK}$']
        #                     },
        # "MA_single": {'free':["kOffAMP","kOffADP","kOffATP","kOffAMPK",
        #                       "kPhosAMPK","kOffPP1","kDephosPP1"],
        #             'names':[r'$k_{\text{OffAMP}}$',r'$k_{\text{OffADP}}$',
        #                      r'$k_{\text{OffATP}}$',r'$k_{\text{OffAMPK}}$',
        #                      r'$k_{\text{PhosAMPK}}$',r'$k_{\text{OffPP1}}$',
        #                      r'$k_{\text{Dephos,PP1}}$']
        #             }, 
        "MM_single":  {'free':["kOffAMP","kOffADP","kOffATP","kCaMKK",
                               "kLKB1"],
                    'names':[r'$k_{\text{OffAMP}}$',r'$k_{\text{OffADP}}$',
                             r'$k_{\text{OffATP}}$',r'$k_{\text{PhosCaMKK}}$',
                             r'$k_{\text{PhosLKB1}}$',r'$K_{\text{m,LKB1}}$']
                      },
        # "MA_nonessential": {'free':["kOffAMP","kOffADP","kOffATP","kOffAMPK",
        #                             "kPhosAMPK","kOffPP1","kDephosPP1","betaAMP"],
        #             'names':[r'$k_{\text{OffAMP}}$',r'$k_{\text{OffADP}}$',
        #                      r'$k_{\text{OffATP}}$',r'$k_{\text{OffAMPK}}$',
        #                      r'$k_{\text{PhosAMPK}}$',r'$k_{\text{OffPP1}}$',
        #                      r'$k_{\text{Dephos,PP1}}$',r'$\beta_{\text{AMP}}$'],
        #                 }, 
        # "MM_nonessential":  {'free':["kOffAMP","kOffADP","kOffATP","betaAMP"],
        #             'names':[r'$k_{\text{OffAMP}}$',r'$k_{\text{OffADP}}$',
        #                      r'$k_{\text{OffATP}}$',r'$\beta_{\text{AMP}}$'],
        #            }
        }

data_dir = '../../../results/param_est/'
save_dir_base = '../../../results/param_est/figs/'

## Load data
cyto_data, _, cyto_times = load_data('../../../Schmitt_et_al_2022_data/fig_2e_cyto.npz', 
                                     to_seconds=False, constant_std=False)
lyso_data, _, lyso_times = load_data('../../../Schmitt_et_al_2022_data/fig_2b_lyso.npz', 
                                     to_seconds=False, constant_std=False)
mito_data, _, mito_times = load_data('../../../Schmitt_et_al_2022_data/fig_2c_mito.npz', 
                                     to_seconds=False, constant_std=False)

for i, model in enumerate(models_free_params.keys()):
    
    print(f"Processing model {i+1}: {model}")
    
    save_dir = save_dir_base + model + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # load the idata
    idata_cyto = az.from_netcdf(data_dir + model + '_cyto_mcmc_samples_' + sampler + '.nc')
    idata_lyso = az.from_netcdf(data_dir + model + '_lyso_mcmc_samples_' + sampler + '.nc')
    idata_mito = az.from_netcdf(data_dir + model + '_mito_mcmc_samples_' + sampler + '.nc')

    ########### plot traces
    az.plot_trace(idata_cyto)
    plt.savefig(save_dir + 'cyto_trace.png', dpi=500)
    az.plot_trace(idata_lyso)
    plt.savefig(save_dir + 'lyso_trace.png', dpi=500)
    az.plot_trace(idata_mito)
    plt.savefig(save_dir + 'mito_trace.png', dpi=500)

    ########### convergence metrics
    with open(model + '_convergence.txt', 'w') as file:
        summary = az.summary(idata_cyto)
        file.write("Cytosol:\n")
        file.write(summary.to_string())
        file.write("\n\n")

        summary = az.summary(idata_lyso)
        file.write("Lysosome:\n")
        file.write(summary.to_string())
        file.write("\n\n")

        summary = az.summary(idata_mito)
        file.write("Mitochondria:\n")
        file.write(summary.to_string())
        file.write("\n\n")

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
    for param in combined_posterior.columns:
        if param != 'compartment':
            fig, ax = get_sized_fig_ax(1.5, 1.0)
            sns.kdeplot(data=combined_posterior, x=param, hue='compartment', ax=ax,
                        fill=True)
            ax.set_xlabel("", fontsize=8.0)
            ax.set_ylabel("", fontsize=8.0)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontsize(8)
            plt.savefig(save_dir + f'{param}_dist.pdf', transparent=True,
                        bbox_inches='tight')
            plt.close()

    ############ plot posterior predictive for each model
    # cyto
    fig, ax, _ = plot_predictive(idata_cyto, cyto_data, cyto_times, plot_prior=False,
                    add_t_0=True, n_traces=0, figsize=(6, 4), prior_color='blue',
                    post_color='black', data_color='red', data_marker_size=10)
    fig, ax, _ = plot_predictive(idata_lyso, lyso_data, cyto_times, plot_prior=False,
                    add_t_0=True, n_traces=0, figsize=(6, 4), prior_color='blue',
                    post_color='blue', data_color='red', data_marker_size=10, fig_ax=(fig,ax))
    fig, ax, _ = plot_predictive(idata_mito, mito_data, mito_times, plot_prior=False,
                    add_t_0=True, n_traces=0, figsize=(6, 4), prior_color='blue',
                    post_color='green', data_color='red', data_marker_size=10, fig_ax=(fig,ax))
    
    fig.savefig(save_dir + 'ppc.pdf', transparent=True, bbox_inches='tight')

    # # Combine the three dataframes into one
    # combined_posterior = pd.concat([posterior_idata_cyto, posterior_idata_lyso, posterior_idata_mito])
    # # Reset index to avoid duplicate indices
    # combined_posterior.reset_index(drop=True, inplace=True)


    ############ plot response to LKB1 and CaMKK KD

# # load samples from all models
# # Model 1
# idata_1_cyto = az.from_netcdf("./test/MA_single_mcmc_samples_pm.nc")
# # idata_1_cyto = az.from_netcdf("./test/MA_single_cyto_prior_samples.nc")
# # idata_1_lyso = az.from_netcdf("./test/MA_single_lyso_mcmc_samples.nc")
# # # idata_1_mito = az.from_netcdf("./test/MA_single_mito_mcmc_samples.nc")

# # # Model 2
# # idata_2_cyto = az.from_netcdf("./test/MM_single_cyto_mcmc_samples.nc")
# # idata_2_lyso = az.from_netcdf("./test/MM_single_lyso_mcmc_samples.nc")
# # # idata_2_mito = az.from_netcdf("./test/MM_single_mito_mcmc_samples.nc")

# # dictionary to store all idata objects
# idata_dict = {
#     "model_1":{
#         "data":{"cyto": idata_1_cyto,
#         # "lyso": idata_1_lyso,
#         # "mito": idata_1_mito,
#         },
#         "free_params":['kOffAMP', 'kOffADP', 'kOffATP', 'kOffAMPK', 'kPhosAMPK', 'kOffPP1', 'kDephosPP1'],
#         "model_name": "MA_single",
#     },
#     # "model_2":{
#     #     "data":{"cyto": idata_2_cyto,
#     #     "lyso": idata_2_lyso,
#     #     # "mito": idata_2_mito,
#     #     },
#     #     "free_params":['kOffAMP', 'kOffADP', 'kOffATP', 'kPhosCaMKK', 'kPhosLKB1'],
#     #     "model_name": "MM_single",
#     # }
# }

# data_names = {"cyto":"../../../Schmitt_et_al_2022_data/fig_2e_cyto.npz",
#                 "lyso":"../../../Schmitt_et_al_2022_data/fig_2b_lyso.npz",
#                 "mito":"../../../Schmitt_et_al_2022_data/fig_2c_mito.npz"}

# save_dir = f"./test/figs/"
# # loop over models
# for model in idata_dict.keys():
#     # loop over compartments
#     for compartment in idata_dict[model]["data"].keys():
#         idata = idata_dict[model]["data"][compartment]
#         # plot the trace plot
#         # az.plot_trace(idata, var_names=idata_dict[model]["free_params"])
#         # plt.savefig(save_dir + f"{model}_{compartment}_trace_plot.png")
#         # plt.close()

#         # az.plot_pair(idata, var_names=idata_dict[model]["free_params"])
#         # plt.savefig(save_dir + f"{model}_{compartment}_pair_plot.png")
#         # plt.close()

#         # # plot the posterior plot
#         # az.plot_autocorr(idata, var_names=idata_dict[model]["free_params"])
#         # plt.savefig(save_dir + f"{model}_{compartment}_autocorr_plot.png")
#         # plt.close()

#         # # plot the posterior plot
#         # az.plot_ess(idata, var_names=idata_dict[model]["free_params"])
#         # plt.savefig(save_dir + f"{model}_{compartment}_ess_plot.png")
#         # plt.close()

#         # # plot ppc
#         # fig, ax = get_sized_fig_ax(5,2.5)
#         # n_samples = 100
#         # # chain_idxs = [0,4,7]
#         # for chain in chain_idxs:
#         #     for i in range(n_samples):
#         #         ax.plot(idata.prior_predictive["obs"][chain,i,:].values, color='gray', alpha=0.1)
#         # fig.savefig(save_dir + f"{model}_{compartment}_ppostpc_plot.png", bbox_inches='tight')
#         # ax.set_xlabel("Time index")
#         # ax.set_ylabel(r"$pAMPKAR/AMPKAR_{total$}")
#         # plt.close()

        

#         # plot prior pc
#         # fig, ax = get_sized_fig_ax(5,2.5)
#         # n_samples = 400
#         # # chain_idxs = [0,4,7]
#         # for i in range(n_samples):
#         #     ax.plot(idata.prior_predictive["obs"][0,i,:].values, color='gray', alpha=0.1)
#         # fig.savefig(save_dir + f"{model}_{compartment}_priorpc_plot.png", bbox_inches='tight')
#         # ax.set_xlabel("Time index")
#         # ax.set_ylabel(r"$pAMPKAR/AMPKAR_{total$}")
#         # plt.close()

#         # load data
#         data, data_std, times = load_data(data_names[compartment], to_seconds=True, constant_std=False)

#         # run prior and post sims and plot
#         # prior
#         # prior_samples = get_param_subsample(idata, 10, prior_or_post="prior")
#         # prior_sims = run_simulations(prior_samples, idata_dict[model]["model_name"], 
#         #                 "../models/" + idata_dict[model]["model_name"] + ".json", "../models/metabolism_params_Coccimiglio.json", times)
        
#         fig, ax = get_sized_fig_ax(5,2.5)

#         for i in range(20):
#             # print(idata.prior_predictive["llike"][0,i,:].values.shape)
#             ax.plot(times, jnp.squeeze(idata.posterior_predictive["llike"][0,i,:].values), color='gray', alpha=0.2)
#             # ax.plot(times, jnp.squeeze(idata.posterior_predictive["llike"][1,i,:].values), color='blue', alpha=0.2)
#             # ax.plot(times, jnp.squeeze(idata.posterior_predictive["llike"][2,i,:].values), color='red', alpha=0.2)
#             # ax.plot(times, jnp.squeeze(idata.posterior_predictive["llike"][3,i,:].values), color='black', alpha=0.2)

#         ax.plot(times, data, color='black', label='data')
    
#         # for sample in prior_sims:
#         #     ax.plot(times, sample, color='gray', alpha=0.1)
#         # ax.set_xlabel("Time index")
#         # ax.set_ylabel(r"$pAMPKAR/AMPKAR_{total}$")
#         fig.savefig(save_dir + f"{model}_{compartment}_prior_sim_.png", bbox_inches='tight')
#         plt.close()

#         az.plot_trace(idata)
#         plt.savefig(save_dir + f"{model}_{compartment}_trace_plot.png")


#         # # posterior
#         # post_samples = get_param_subsample(idata, 110)
#         # post_sims = run_simulations(post_samples, idata_dict[model]["model_name"], 
#         #                 "../models/" + idata_dict[model]["model_name"] + ".json", "../models/metabolism_params_Coccimiglio.json", times)
        
#         # fig, ax = get_sized_fig_ax(5,2.5)
    
#         # for sample in post_sims:
#         #     ax.plot(times, sample, color='gray', alpha=0.1)
#         # ax.plot(times, data, color='red', label='data')
#         # ax.errorbar(times, data, yerr=data_std, color='red', fmt='o')
#         # ax.set_xlabel("Time index")
#         # ax.set_ylabel(r"$pAMPKAR/AMPKAR_{total}$")
#         # fig.savefig(save_dir + f"{model}_{compartment}_post_sim_.png", bbox_inches='tight')
#         # plt.close()
