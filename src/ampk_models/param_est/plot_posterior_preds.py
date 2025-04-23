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
from MA_single_diffrax import *
from MM_single_diffrax import *
from MA_nonessential_diffrax import *
from MM_nonessential_diffrax import *
from MA_nonessential_all_diffrax import *
from MM_nonessential_all_diffrax import *

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

data_dir = '../../../results/param_est/WT_only/'
save_dir_base = '../../../results/param_est/WT_only/figs/'

## Load data
cyto_data, _, cyto_times = load_data('../../../Schmitt_et_al_2022_data/fig_2e_cyto.npz', to_seconds=False, constant_std=False)
cyto_data_LKB1_KO, _, _ = load_data('../../../Schmitt_et_al_2022_data/fig_2e_cyto_LKB1_KD.npz', to_seconds=False, 
                        constant_std=False)
cyto_data_CaMKK2_KO, _, _ = load_data('../../../Schmitt_et_al_2022_data/sup_fig_2g_cyto_CaMKK_KD.npz', to_seconds=False,
                        constant_std=False)

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
            'cyto':{'idata': idata_cyto, 'data': cyto_data, 'times': cyto_times},
        }
        for comp in dat.keys():
            fig, ax = get_sized_fig_ax(2,1)
    
            fig, ax, leg = plot_predictive(dat[comp]['idata'], dat[comp]['data'], dat[comp]['times'], 
                            plot_prior=False, n_traces=0, figsize=None, 
                            prior_color='', post_color=colors[j], data_color='k', data_linestyle='--',
                            fig_ax = (fig, ax))
        
            # add n_trajectories to the plot if n_trajectories > 0
            if n_trajectories > 0:
                for i in range(n_trajectories):
                    ax.plot(dat[comp]['times'], 
                        jnp.squeeze(dat[comp]['idata'].posterior_predictive["llike"][0,i,:].values), 
                        color=colors[j], alpha=0.2, linewidth=1.0)
                    
            export_legend(leg, save_dir + f'{comp}_ppc_legend_' + sampler + '.pdf')
            leg.remove()
                   
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_ylim(0, 1.5)

            plt.savefig(save_dir + f'{comp}_ppc_' + sampler + '.pdf', transparent=True, bbox_inches='tight')


        ############ plot posterior for each model
        # run posterior simulations
        # cyto
        sims = {
            'cyto':None,
        }
        dat = {
            'cyto':{'data': cyto_data, 'times': cyto_times},
        }
        idata = {'cyto': idata_cyto}
        

        for comp in ['cyto']:
            # if simulations are already run, load them
            if os.path.isfile(save_dir + f'post_sims_{comp}_{sampler}.npy') & \
                os.path.isfile(save_dir + f'post_sims_{comp}_{sampler}_LKB1_KD.npy') & \
                os.path.isfile(save_dir + f'post_sims_{comp}_{sampler}_CaMKK_KD.npy'):
                print(f"Files already exists. Skipping simulations for {comp}.")
                
                post_sims = np.load(save_dir + f'post_sims_{comp}_{sampler}.npy')
                post_sims_LKB1_KD = np.load(save_dir + f'post_sims_{comp}_{sampler}_LKB1_KD.npy')
                post_sims_CaMKK_KD = np.load(save_dir + f'post_sims_{comp}_{sampler}_CaMKK_KD.npy')
            else: # otherwise run them
                model_info_file = json.load(open(models_free_params[model]['info_file']))         
                param_samples = get_param_subsample(model_info_file['params'], idata[comp], 400, prior_or_post="post", rng=np.random.default_rng(seed=1234))
                post_sims = run_simulations(param_samples, model, models_free_params[model]['info_file'], 
                                '../models/metabolism_params_Coccimiglio.json',
                                dat[comp]['times']*60, rtol=1e-6,atol=1e-6,pcoeff=0.3,
                                icoeff=0.4, ca_stress=0.25)
                np.save(save_dir + f'post_sims_{comp}_{sampler}.npy', np.array(post_sims))

                # LKB1 KD
                param_samples_LKB1_KD = param_samples.copy()
                if 'MA' in model:
                    k_on_lkb1_idx = model_info_file['params'].index('kOnLKB1')
                    k_cat_lkb1_idx = model_info_file['params'].index('kPhosLKB1')
                    param_samples_LKB1_KD[:,k_on_lkb1_idx] = 0.0
                    param_samples_LKB1_KD[:,k_cat_lkb1_idx] = 0.0
                elif 'MM' in model:
                    k_cat_lkb1_idx = model_info_file['params'].index('LKB1tot')
                    param_samples_LKB1_KD[:,k_cat_lkb1_idx] = 0.0

                post_sims_LKB1_KD = run_simulations(param_samples_LKB1_KD, model, 
                                                    models_free_params[model]['info_file'], 
                                '../models/metabolism_params_Coccimiglio.json',
                                dat[comp]['times']*60, rtol=1e-6,atol=1e-6,pcoeff=0.3,
                                icoeff=0.4, ca_stress=0.25)
                np.save(save_dir + f'post_sims_{comp}_{sampler}_LKB1_KD.npy',
                        np.array(post_sims_LKB1_KD))

                # CaMKK KD
                param_samples_CaMKK_KD = param_samples.copy()
                if 'MA' in model:
                    k_on_camkk_idx = model_info_file['params'].index('kOnCaMKK')
                    k_cat_camkk_idx = model_info_file['params'].index('kPhosCaMKK')
                    param_samples_CaMKK_KD[:,k_on_camkk_idx] = 0.0
                    param_samples_CaMKK_KD[:,k_cat_camkk_idx] = 0.0
                elif 'MM' in model:
                    k_cat_camkk_idx = model_info_file['params'].index('kCaMKK')
                    param_samples_CaMKK_KD[:,k_cat_camkk_idx] = 0.0

                post_sims_CaMKK_KD = run_simulations(param_samples_CaMKK_KD, model, 
                                                     models_free_params[model]['info_file'],
                                '../models/metabolism_params_Coccimiglio.json',
                                dat[comp]['times']*60, rtol=1e-6,atol=1e-6,pcoeff=0.3,
                                icoeff=0.4, ca_stress=0.25)
                np.save(save_dir + f'post_sims_{comp}_{sampler}_CaMKK_KD.npy', 
                        np.array(post_sims_CaMKK_KD))

            # store in dict
            sims[comp] = {
                'baseline':post_sims,
                'LKB1_KD':post_sims_LKB1_KD,
                'CaMKK_KD':post_sims_CaMKK_KD
            }

        ############ plots
        for comp in dat.keys():
            for perturb in ['baseline', 'LKB1_KD', 'CaMKK_KD']:
                ####### posterior #######
                fig_width, fig_height = 1.75, 0.6
                fig, ax = get_sized_fig_ax(fig_width, fig_height)

                if perturb == 'baseline':
                    data = dat[comp]['data']
                    linestyle = '-'
                else:
                    # # no data plotted for perturbed conditions
                    # data = np.nan*np.ones_like(dat[comp]['data'])
                    if perturb == 'LKB1_KD':
                        data = cyto_data_LKB1_KO
                        linsetstyle = '-'
                    else:
                        data = cyto_data_CaMKK2_KO
                        linestyle = '-'
        
                fig, ax, leg = plot_predictive(sims[comp][perturb], data, dat[comp]['times'], 
                                plot_prior=False, n_traces=0, figsize=None, 
                                prior_color='', post_color=colors[j], data_color='k', data_linestyle='--', 
                                fig_ax = (fig, ax))
            
                # add n_trajectories to the plot if n_trajectories > 0
                if n_trajectories > 0:
                    for i in range(n_trajectories):
                        ax.plot(dat[comp]['times'], 
                            jnp.squeeze(sims[comp][perturb][i,:]), 
                            color=colors[j], alpha=0.2, linewidth=1.0)
                        
                export_legend(leg, save_dir + f'{comp}_ppc_legend_' + sampler + '.pdf')
                leg.remove()
                    
                ax.set_xlabel("time (min)", fontsize=10.0)
                ax.set_ylabel("")
                ax.set_ylim(0, 1.5)

                if perturb == 'baseline':
                    plt.savefig(save_dir + f'{comp}_posterior_{sampler}.pdf', 
                                transparent=True, bbox_inches='tight')
                else:
                    plt.savefig(save_dir + f'{comp}_posterior_{perturb}_{sampler}.pdf', 
                                transparent=True, bbox_inches='tight')
                    

# Make a fake plot to get the legend with colors
fig, ax = get_sized_fig_ax(1.5, 1.0)
# Create a dummy plot to get the legend
for i, model in enumerate(models_free_params.keys()):
    ax.plot([], [], color=colors[i], label=f'Model {i+1}', linewidth=4.0)
# Create the legend
leg = ax.legend(title='', loc='upper left', bbox_to_anchor=(3, 1),
                fontsize=8.0, ncols=3)
export_legend(leg, save_dir_base + f'traj_legend.pdf')