import arviz as az
import pandas as pd
import json
import os

import numpy as np
import diffrax
import matplotlib.pyplot as plt
import seaborn as sns
import jax
import sys

sys.path.append("../")
from utils import *
from plotting_helper_funcs import *

sys.path.append("../models/")

# tell jax to use 64bit floats
jax.config.update("jax_enable_x64", True)

# load samples from all models
# Model 1
idata_1_cyto = az.from_netcdf("./test/MA_single_cyto_mcmc_samples.nc")
idata_1_lyso = az.from_netcdf("./test/MA_single_lyso_mcmc_samples.nc")
# idata_1_mito = az.from_netcdf("./test/MA_single_mito_mcmc_samples.nc")

# Model 2
idata_2_cyto = az.from_netcdf("./test/MM_single_cyto_mcmc_samples.nc")
idata_2_lyso = az.from_netcdf("./test/MM_single_lyso_mcmc_samples.nc")
# idata_2_mito = az.from_netcdf("./test/MM_single_mito_mcmc_samples.nc")

# dictionary to store all idata objects
idata_dict = {
    "model_1":{
        "data":{"cyto": idata_1_cyto,
        "lyso": idata_1_lyso,
        # "mito": idata_1_mito,
        },
        "free_params":['kOffAMP', 'kOffADP', 'kOffATP', 'kOffAMPK', 'kPhosAMPK', 'kOffPP1', 'kDephosPP1'],
        "model_name": "MA_single",
    },
    "model_2":{
        "data":{"cyto": idata_2_cyto,
        "lyso": idata_2_lyso,
        # "mito": idata_2_mito,
        },
        "free_params":['kOffAMP', 'kOffADP', 'kOffATP', 'kPhosCaMKK', 'kPhosLKB1'],
        "model_name": "MM_single",
    }
}

data_names = {"cyto":"../../../Schmitt_et_al_2022_data/fig_2e_cyto.npz",
                "lyso":"../../../Schmitt_et_al_2022_data/fig_2b_lyso.npz",
                "mito":"../../../Schmitt_et_al_2022_data/fig_2c_mito.npz"}

save_dir = f"./test/figs/"
# loop over models
for model in idata_dict.keys():
    # loop over compartments
    for compartment in idata_dict[model]["data"].keys():
        idata = idata_dict[model]["data"][compartment]
        # plot the trace plot
        # az.plot_trace(idata, var_names=idata_dict[model]["free_params"])
        # plt.savefig(save_dir + f"{model}_{compartment}_trace_plot.png")
        # plt.close()

        # az.plot_pair(idata, var_names=idata_dict[model]["free_params"])
        # plt.savefig(save_dir + f"{model}_{compartment}_pair_plot.png")
        # plt.close()

        # # plot the posterior plot
        # az.plot_autocorr(idata, var_names=idata_dict[model]["free_params"])
        # plt.savefig(save_dir + f"{model}_{compartment}_autocorr_plot.png")
        # plt.close()

        # # plot the posterior plot
        # az.plot_ess(idata, var_names=idata_dict[model]["free_params"])
        # plt.savefig(save_dir + f"{model}_{compartment}_ess_plot.png")
        # plt.close()

        # # plot ppc
        # fig, ax = get_sized_fig_ax(5,2.5)
        # n_samples = 100
        # # chain_idxs = [0,4,7]
        # for chain in chain_idxs:
        #     for i in range(n_samples):
        #         ax.plot(idata.prior_predictive["obs"][chain,i,:].values, color='gray', alpha=0.1)
        # fig.savefig(save_dir + f"{model}_{compartment}_ppostpc_plot.png", bbox_inches='tight')
        # ax.set_xlabel("Time index")
        # ax.set_ylabel(r"$pAMPKAR/AMPKAR_{total$}")
        # plt.close()

        

        # plot prior pc
        # fig, ax = get_sized_fig_ax(5,2.5)
        # n_samples = 400
        # # chain_idxs = [0,4,7]
        # for i in range(n_samples):
        #     ax.plot(idata.prior_predictive["obs"][0,i,:].values, color='gray', alpha=0.1)
        # fig.savefig(save_dir + f"{model}_{compartment}_priorpc_plot.png", bbox_inches='tight')
        # ax.set_xlabel("Time index")
        # ax.set_ylabel(r"$pAMPKAR/AMPKAR_{total$}")
        # plt.close()

        # load data
        data, data_std, times = load_data(data_names[compartment], to_seconds=True, constant_std=False)

        # run prior and post sims and plot
        # prior
        prior_samples = get_param_subsample(idata, 10, prior_or_post="prior")
        prior_sims = run_simulations(prior_samples, idata_dict[model]["model_name"], 
                        "../models/" + idata_dict[model]["model_name"] + ".json", "../models/metabolism_params_Coccimiglio.json", times)
        
        fig, ax = get_sized_fig_ax(5,2.5)
    
        for sample in prior_sims:
            ax.plot(times, sample, color='gray', alpha=0.1)
        ax.set_xlabel("Time index")
        ax.set_ylabel(r"$pAMPKAR/AMPKAR_{total}$")
        fig.savefig(save_dir + f"{model}_{compartment}_prior_sim_.png", bbox_inches='tight')
        plt.close()

        # posterior
        post_samples = get_param_subsample(idata, 110)
        post_sims = run_simulations(post_samples, idata_dict[model]["model_name"], 
                        "../models/" + idata_dict[model]["model_name"] + ".json", "../models/metabolism_params_Coccimiglio.json", times)
        
        fig, ax = get_sized_fig_ax(5,2.5)
    
        for sample in post_sims:
            ax.plot(times, sample, color='gray', alpha=0.1)
        ax.plot(times, data, color='red', label='data')
        ax.errorbar(times, data, yerr=data_std, color='red', fmt='o')
        ax.set_xlabel("Time index")
        ax.set_ylabel(r"$pAMPKAR/AMPKAR_{total}$")
        fig.savefig(save_dir + f"{model}_{compartment}_post_sim_.png", bbox_inches='tight')
        plt.close()
