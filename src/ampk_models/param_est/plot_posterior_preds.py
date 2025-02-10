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

samplers = ["ADVI"] #,"Nutpie", "NUTS"] # list of samplers used
n_trajectories = 10 # number of trajectories to add to PPC plot

cyto_color = colors[0]
lyso_color = colors[1]
mito_color = colors[2]


models_free_params = { 
        # "ampk_Coccimiglio": {'info_file': '../models/ampk_Coccimiglio.json'},
        "MA_single": {'info_file':'../models/MA_single.json'},
        "MM_single":  {'info_file': '../models/MM_single.json'},
        "MA_nonessential": {'info_file': '../models/MA_nonessential.json'},
        "MM_nonessential":  {'info_file': '../models/MM_nonessential.json'}
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

    for sampler in samplers:
    
        save_dir = save_dir_base + model + '/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # load the idata
        idata_cyto = az.from_netcdf(data_dir + model + '_cyto_mcmc_samples_' + sampler + '.nc')
        idata_lyso = az.from_netcdf(data_dir + model + '_lyso_mcmc_samples_' + sampler + '.nc')
        idata_mito = az.from_netcdf(data_dir + model + '_mito_mcmc_samples_' + sampler + '.nc')

        # # ########### plot traces
        # az.plot_trace(idata_cyto)
        # plt.savefig(save_dir + 'cyto_trace_' + sampler + '.png', dpi=500)
        # az.plot_trace(idata_lyso)
        # plt.savefig(save_dir + 'lyso_trace_' + sampler + '.png', dpi=500)
        # az.plot_trace(idata_mito)
        # plt.savefig(save_dir + 'mito_trace_' + sampler + '.png', dpi=500)

        # ########### convergence metrics
        # with open(model + '_convergence_' + sampler + '.txt', 'w') as file:
        #     summary = az.summary(idata_cyto)
        #     file.write("Cytosol:\n")
        #     file.write(summary.to_string())
        #     file.write("\n\n")

        #     summary = az.summary(idata_lyso)
        #     file.write("Lysosome:\n")
        #     file.write(summary.to_string())
        #     file.write("\n\n")

        #     summary = az.summary(idata_mito)
        #     file.write("Mitochondria:\n")
        #     file.write(summary.to_string())
        #     file.write("\n\n")

        # ############ plot posterior predictive for each model
        # dat = {
        #     'cyto':{'idata': idata_cyto, 'data': cyto_data, 'times': cyto_times, 'color': cyto_color},
        #     'lyso':{'idata': idata_lyso, 'data': lyso_data, 'times': lyso_times, 'color': lyso_color},
        #     'mito':{'idata': idata_mito, 'data': mito_data, 'times': mito_times, 'color': mito_color}
        # }
        # for comp in dat.keys():
        #     fig, ax = get_sized_fig_ax(2,1)
    
        #     fig, ax, leg = plot_predictive(dat[comp]['idata'], dat[comp]['data'], dat[comp]['times'], 
        #                     plot_prior=False, add_t_0=True, n_traces=0, figsize=None, 
        #                     prior_color='', post_color=dat[comp]['color'], data_color=dat[comp]['color'], 
        #                     data_marker_size=10, fig_ax = (fig, ax))
        
        #     # add n_trajectories to the plot if n_trajectories > 0
        #     if n_trajectories > 0:
        #         for i in range(n_trajectories):
        #             ax.plot(dat[comp]['times'], 
        #                 jnp.squeeze(dat[comp]['idata'].posterior_predictive["llike"][0,i,:].values), 
        #                 color=dat[comp]['color'], alpha=0.2, linewidth=1.0)
                    
        #     export_legend(leg, save_dir + f'{comp}_ppc_legend_' + sampler + '.pdf')
        #     leg.remove()
                   
        #     ax.set_xlabel("")
        #     ax.set_ylabel("")
        #     ax.set_ylim(0, 1.5)

        #     plt.savefig(save_dir + f'{comp}_ppc_' + sampler + '.pdf', transparent=True, bbox_inches='tight')


        ############ plot posterior for each model
        # run posterior simulations
        # cyto
        sims = {
            'cyto':None,
            'lyso':None,
            'mito':None
        }
        dat = {
            'cyto':{'data': cyto_data, 'times': cyto_times, 'color': cyto_color},
            'lyso':{'data': lyso_data, 'times': lyso_times, 'color': lyso_color},
            'mito':{'data': mito_data, 'times': mito_times, 'color': mito_color}
        }
        idata = {'cyto': idata_cyto, 'lyso': idata_lyso, 'mito': idata_mito}
        

        for comp in ['cyto', 'lyso', 'mito']:
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
                                icoeff=0.4)
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
                                icoeff=0.4)
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
                    k_cat_camkk_idx = model_info_file['params'].index('CaMKKtot')
                    param_samples_CaMKK_KD[:,k_cat_camkk_idx] = 0.0

                post_sims_CaMKK_KD = run_simulations(param_samples_CaMKK_KD, model, 
                                                     models_free_params[model]['info_file'],
                                '../models/metabolism_params_Coccimiglio.json',
                                dat[comp]['times']*60, rtol=1e-6,atol=1e-6,pcoeff=0.3,
                                icoeff=0.4)
                np.save(save_dir + f'post_sims_{comp}_{sampler}_CaMKK_KD.npy', 
                        np.array(post_sims_CaMKK_KD))

            # store in dict
            sims[comp] = {
                'baseline':post_sims,
                'LKB1_KD':post_sims_LKB1_KD,
                'CaMKK_KD':post_sims_CaMKK_KD
            }

        # ############ plots
        # for comp in dat.keys():
        #     for perturb in ['baseline', 'LKB1_KD', 'CaMKK_KD']:
        #         ####### posterior #######
        #         fig_width, fig_height = 1.25, 0.5
        #         fig, ax = get_sized_fig_ax(fig_width, fig_height)

        #         if perturb == 'baseline':
        #             data = dat[comp]['data']
        #             linestyle = '-'
        #         else:
        #             # no data plotted for perturbed conditions
        #             data = np.nan*np.ones_like(dat[comp]['data'])
        #             if perturb == 'LKB1_KD':
        #                 linsetstyle = '--'
        #             else:
        #                 linestyle = ':'
        
        #         fig, ax, leg = plot_predictive(sims[comp][perturb], data, dat[comp]['times'], 
        #                         plot_prior=False, add_t_0=True, n_traces=0, figsize=None, 
        #                         prior_color='', post_color=dat[comp]['color'], data_color=dat[comp]['color'], 
        #                         data_marker_size=10, fig_ax = (fig, ax))
            
        #         # add n_trajectories to the plot if n_trajectories > 0
        #         if n_trajectories > 0:
        #             for i in range(n_trajectories):
        #                 ax.plot(dat[comp]['times'], 
        #                     jnp.squeeze(sims[comp][perturb][i,:]), 
        #                     color=dat[comp]['color'], alpha=0.2, linewidth=1.0)
                        
        #         export_legend(leg, save_dir + f'{comp}_ppc_legend_' + sampler + '.pdf')
        #         leg.remove()
                    
        #         ax.set_xlabel("time (min)", fontsize=10.0)
        #         ax.set_ylabel("")
        #         ax.set_ylim(0, 1.5)

        #         if perturb == 'baseline':
        #             plt.savefig(save_dir + f'{comp}_posterior_{sampler}.pdf', 
        #                         transparent=True, bbox_inches='tight')
        #         else:
        #             plt.savefig(save_dir + f'{comp}_posterior_{perturb}_{sampler}.pdf', 
        #                         transparent=True, bbox_inches='tight')
                    
    ####### t-half-max and change in max activation #######
    # t-half-max
    # get t_half_max for each condition

    # calculate time to half max of pAMPKAR_stressed/AMPKAR_stressed
    def compute_half_max(arr):
        half_max = arr.max() / 2
        half_max_idx = np.argmin(np.abs(arr - half_max))
        return half_max_idx
    
    t_half_max = {
        'cyto':{'baseline':None,'LKB1_KD':None,'CaMKK_KD':None},
        'lyso':{'baseline':None,'LKB1_KD':None,'CaMKK_KD':None},
        'mito':{'baseline':None,'LKB1_KD':None,'CaMKK_KD':None}
    }

    max_dat = {
        'cyto':{'baseline':None,'LKB1_KD':None,'CaMKK_KD':None},
        'lyso':{'baseline':None,'LKB1_KD':None,'CaMKK_KD':None},
        'mito':{'baseline':None,'LKB1_KD':None,'CaMKK_KD':None}
    }

    for comp in ['cyto', 'lyso', 'mito']:
        t_half_idx = np.apply_along_axis(compute_half_max, 1, sims[comp]['baseline'])
        t_half_max[comp]['baseline'] = [dat[comp]['times'][idx] for idx in t_half_idx]
        t_half_idx = np.apply_along_axis(compute_half_max, 1, sims[comp]['LKB1_KD'])
        t_half_max[comp]['LKB1_KD'] = [dat[comp]['times'][idx] for idx in t_half_idx]
        t_half_idx = np.apply_along_axis(compute_half_max, 1, sims[comp]['CaMKK_KD'])
        t_half_max[comp]['CaMKK_KD'] = [dat[comp]['times'][idx] for idx in t_half_idx]

        max_dat[comp]['baseline'] = sims[comp]['baseline'][:,-1]
        max_dat[comp]['LKB1_KD'] = sims[comp]['LKB1_KD'][:,-1]
        max_dat[comp]['CaMKK_KD'] = sims[comp]['CaMKK_KD'][:,-1]

    # plot t_half_max
    fig, ax = get_sized_fig_ax(1.5, 1.0)

    thalf_df = pd.DataFrame(t_half_max)
    thalf_df = thalf_df.melt(var_name='compartment', value_name='t_half_max', ignore_index=False)
    thalf_df = thalf_df.explode('t_half_max')
    thalf_df = thalf_df.reset_index()
    thalf_df = thalf_df.rename(columns={'index':'perturbation'})

    thalf_df['compartment'] = thalf_df['compartment'].replace('cyto', 'cytoplasm')
    thalf_df['compartment'] = thalf_df['compartment'].replace('lyso', 'lysosome')
    thalf_df['compartment'] = thalf_df['compartment'].replace('mito', 'mitochondria')

    sns.barplot(data=thalf_df, x='perturbation', y='t_half_max', hue='compartment', 
                ax=ax, palette=colors, linewidth=0, edgecolor='black',width=0.8,
                dodge=True, saturation=1.0, gap=0.1)
    
    for patch in ax.patches:
        face_color = patch.get_facecolor()
        # Apply transparency to the face color only
        patch.set_facecolor(mpl.colors.to_rgba(face_color, alpha=0.5))  
        patch.set_edgecolor(face_color)  # Set edge color to match the fill

    leg = ax.legend(title='', loc='upper left', bbox_to_anchor=(3, 1), 
                    fontsize=8.0, ncols=2)
    export_legend(leg, save_dir + f't_half_max_legend.pdf')
    leg.remove()

    ax.set_xticklabels(['WT', 'LKB1 KD', 'CaMKK KD'], fontsize=8.0, rotation=45, 
                       )
    
    ax.set_yticks([0, 10])
    ax.set_yticklabels([0, 10], fontsize=8.0)

    ax.set_ylabel(r"$T_{1/2}$ (min)", fontsize=10.0)
    ax.set_xlabel("")

    plt.savefig(save_dir + f'{model}_t_half_max.pdf', transparent=True, bbox_inches='tight')

    # plot max activation
    fig, ax = get_sized_fig_ax(1.5, 1.0)

    max_df = pd.DataFrame(max_dat)
    max_df = max_df.melt(var_name='compartment', value_name='max', ignore_index=False)
    max_df = max_df.explode('max')
    max_df = max_df.reset_index()
    max_df = max_df.rename(columns={'index':'perturbation'})
    

    sns.barplot(data=max_df, x='perturbation', y='max', hue='compartment', 
                ax=ax, palette=colors, linewidth=0, edgecolor='black',width=0.8,
                dodge=True, saturation=1.0, gap=0.1)
    
    for patch in ax.patches:
        face_color = patch.get_facecolor()
        # Apply transparency to the face color only
        patch.set_facecolor(mpl.colors.to_rgba(face_color, alpha=0.5))  
        patch.set_edgecolor(mpl.colors.to_rgba(face_color, alpha=1.0))  # Set edge color to match the fill

    # leg = ax.legend(title='', loc='upper left', bbox_to_anchor=(3, 1), fontsize=8.0)
    # export_legend(leg, save_dir + f't_half_max_legend.pdf')
    ax.get_legend().remove()
    # leg.remove()


    ax.set_xticklabels(['WT', 'LKB1 KD', 'CaMKK KD'], fontsize=8.0, rotation=45, 
                       )
    
    ax.set_yticks([0, 1])
    ax.set_yticklabels([0, 1.0], fontsize=8.0)
    ax.set_ylim(0, 1.15)

    ax.set_ylabel("max. fraction\nactive sensor", fontsize=10.0)
    ax.set_xlabel("")

    plt.savefig(save_dir + f'{model}_max_act.pdf', transparent=True, bbox_inches='tight')
                