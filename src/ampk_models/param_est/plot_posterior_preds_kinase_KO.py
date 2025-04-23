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

colors = ['#2c6b67', '#67322e', '#122c43', '#99610a', '#c38f16', '#175449']

samplers = ["Pathfinder"] #,"Nutpie", "NUTS"] # list of samplers used
n_trajectories = 10 # number of trajectories to add to PPC plot

models_free_params = { 
        "MA_single": {'info_file':'../models/MA_single.json'},
        "MM_single":  {'info_file': '../models/MM_single.json'},
        "MA_nonessential": {'info_file': '../models/MA_nonessential.json'},
        "MM_nonessential":  {'info_file': '../models/MM_nonessential.json'},
        "MA_nonessential_all": {'info_file': '../models/MA_nonessential_all.json'},
        "MM_nonessential_all":  {'info_file': '../models/MM_nonessential_all.json'}
        }

data_dir = '../../../results/param_est/kinase_KO/std_dcr/'
save_dir_base = '../../../results/param_est/kinase_KO/std_dcr/figs/'

## Load data
cyto_data, _, cyto_times = load_data('../../../Schmitt_et_al_2022_data/fig_2e_cyto.npz', 
                                     to_seconds=False, constant_std=False)
lyso_data, _, lyso_times = load_data('../../../Schmitt_et_al_2022_data/fig_2b_lyso.npz', 
                                     to_seconds=False, constant_std=False)
mito_data, _, mito_times = load_data('../../../Schmitt_et_al_2022_data/fig_2c_mito.npz', 
                                     to_seconds=False, constant_std=False)

cyto_data_LKB1_KO, _, _ = load_data('../../../Schmitt_et_al_2022_data/fig_2e_cyto_LKB1_KD.npz', 
                                     to_seconds=False, constant_std=False)
lyso_data_LKB1_KO, _, _ = load_data('../../../Schmitt_et_al_2022_data/fig_2f_lyso_LKB1_KD.npz', 
                                     to_seconds=False, constant_std=False)
mito_data_LKB1_KO, _, _ = load_data('../../../Schmitt_et_al_2022_data/fig_2g_mito_LKB1_KD.npz', 
                                     to_seconds=False, constant_std=False)

cyto_data_CaMKK2_KO, _, _ = load_data('../../../Schmitt_et_al_2022_data/sup_fig_2g_cyto_CaMKK_KD.npz', 
                                     to_seconds=False, constant_std=False)
lyso_data_CaMKK2_KO, _, _ = load_data('../../../Schmitt_et_al_2022_data/sup_fig_2g_lyso_CaMKK_KD.npz', 
                                     to_seconds=False, constant_std=False)
mito_data_CaMKK2_KO, _, _ = load_data('../../../Schmitt_et_al_2022_data/sup_fig_2g_mito_CaMKK_KD.npz', 
                                     to_seconds=False, constant_std=False)

for j, model in enumerate(models_free_params.keys()):
    
    print(f"Processing model {j+1}: {model}")

    for sampler in samplers:
    
        save_dir = save_dir_base + model + '/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # load the idata
        idata_cyto = az.from_netcdf(data_dir + model + '_cyto_mcmc_samples_' + sampler + '.nc')

        ############ plot posterior predictive for each model
        dat = {
            'cyto':{'idata': idata_cyto, 'data': cyto_data, 'data_lkb1_ko':cyto_data_LKB1_KO,
                    'data_camkk2_ko':cyto_data_CaMKK2_KO, 'times': cyto_times},
        }
        # for comp in dat.keys():
        #     for llike in ['llike', 'llike_LKB1_KO', 'llike_CaMKK2_KO']:
        #         fig, ax = get_sized_fig_ax(2,1)

        #         if llike == 'llike':
        #             data = dat[comp]['data']
        #         elif llike == 'llike_LKB1_KO':
        #             data = dat[comp]['data_lkb1_ko']
        #         elif llike == 'llike_CaMKK2_KO':
        #             data = dat[comp]['data_camkk2_ko']
        
        #         fig, ax, leg = plot_predictive(dat[comp]['idata'], data, dat[comp]['times'], 
        #                         plot_prior=False,  n_traces=0, figsize=None, 
        #                         prior_color='', post_color=colors[j], data_color='k', 
        #                         data_linestyle='--, fig_ax = (fig, ax), llike_name=llike)
                
        #         # add n_trajectories to the plot if n_trajectories > 0
        #         if n_trajectories > 0:
        #             for i in range(n_trajectories):
        #                 ax.plot(dat[comp]['times'], 
        #                     jnp.squeeze(dat[comp]['idata'].posterior_predictive[llike][0,i,:].values), 
        #                     color=colors[j], alpha=0.2, linewidth=1.0)
                        
        #         export_legend(leg, save_dir + f'{comp}_ppc_legend_' + sampler + '.pdf')
        #         leg.remove()
                    
        #         ax.set_xlabel("")
        #         ax.set_ylabel("")
        #         ax.set_ylim(0, 1.5)

        #         plt.savefig(save_dir + f'{comp}_{llike}_ppc_' + sampler + '.pdf', transparent=True, bbox_inches='tight')


        ############ plot posterior for each model
        sims = {'cyto':{}}

        for comp in dat.keys():
            for pred in ['prediction', 'prediction_LKB1_KO', 'prediction_CaMKK2_KO']:
                fig_width, fig_height = 1.75, 0.6
                fig, ax = get_sized_fig_ax(fig_width, fig_height)

                if pred == 'prediction':
                    data = dat[comp]['data']
                elif pred == 'prediction_LKB1_KO':
                    data = dat[comp]['data_lkb1_ko']
                elif pred == 'prediction_CaMKK2_KO':
                    data = dat[comp]['data_camkk2_ko']

                trajectories = np.squeeze(dat[comp]['idata']['posterior'][pred].values)
        
                fig, ax, leg = plot_predictive(trajectories, data, dat[comp]['times'], 
                                plot_prior=False,  n_traces=0, figsize=None, 
                                prior_color='', post_color=colors[j], data_color='k', 
                                data_linestyle='--', fig_ax = (fig, ax), llike_name=pred)
                
                # add n_trajectories to the plot if n_trajectories > 0
                if n_trajectories > 0:
                    for i in range(n_trajectories):
                        ax.plot(dat[comp]['times'], 
                            trajectories[i,:], 
                            color=colors[j], alpha=0.2, linewidth=1.0)
                        
                export_legend(leg, save_dir + f'{comp}_ppc_legend_' + sampler + '.pdf')
                leg.remove()
                    
                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.set_ylim(0, 1.5)

                plt.savefig(save_dir + f'{comp}_{pred}_posterior_{sampler}.pdf', 
                            transparent=True, bbox_inches='tight')
                
                sims[comp][pred]=trajectories