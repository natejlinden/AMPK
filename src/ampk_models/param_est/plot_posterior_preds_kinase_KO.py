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

samplers = ["Pathfinder"] #,"Nutpie", "NUTS"] # list of samplers used
n_trajectories = 10 # number of trajectories to add to PPC plot

cyto_color = colors[0]
lyso_color = colors[1]
mito_color = colors[2]


models_free_params = { 
        # "ampk_Coccimiglio": {'info_file': '../models/ampk_Coccimiglio.json'},
        # "MM_single":  {'info_file': '../models/MM_single.json'},
        # "MA_nonessential":  {'info_file': '../models/MA_nonessential.json'},
        # "MA_single":  {'info_file': '../models/MA_single.json'},
        "MA_timeDepCaMKK2":  {'info_file': '../models/MA_timeDepCaMKK2.json'},
        # "MM_nonessential":  {'info_file': '../models/MM_nonessential.json'},
        # "MA_amp_adp_dep":  {'info_file': '../models/MA_amp_adp_dep.json'},
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

for i, model in enumerate(models_free_params.keys()):
    
    print(f"Processing model {i+1}: {model}")

    for sampler in samplers:
    
        save_dir = save_dir_base + model + '/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # load the idata
        idata_cyto = az.from_netcdf(data_dir + model + '_cyto_mcmc_samples_' + sampler + '.nc')
        # idata_lyso = az.from_netcdf(data_dir + model + '_lyso_mcmc_samples_' + sampler + '.nc')
        # idata_mito = az.from_netcdf(data_dir + model + '_mito_mcmc_samples_' + sampler + '.nc')

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

        ############ plot posterior predictive for each model
        dat = {
            'cyto':{'idata': idata_cyto, 'data': cyto_data, 'data_lkb1_ko':cyto_data_LKB1_KO,
                    'data_camkk2_ko':cyto_data_CaMKK2_KO, 'times': cyto_times, 'color': cyto_color},
            #     'lyso':{'idata': idata_lyso, 'data': lyso_data, 'data_lkb1_ko':lyso_data_LKB1_KO,
            #             'data_camkk2_ko':lyso_data_CaMKK2_KO, 'times': lyso_times, 'color': lyso_color},
            # 'mito':{'idata': idata_mito, 'data': mito_data, 'data_lkb1_ko':mito_data_LKB1_KO,
            #         'data_camkk2_ko':mito_data_CaMKK2_KO, 'times': mito_times, 'color': mito_color},
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
        #                         plot_prior=False, add_t_0=True, n_traces=0, figsize=None, 
        #                         prior_color='', post_color=dat[comp]['color'], data_color=dat[comp]['color'], 
        #                         data_marker_size=10, fig_ax = (fig, ax), llike_name=llike)
                
        #         # add n_trajectories to the plot if n_trajectories > 0
        #         if n_trajectories > 0:
        #             for i in range(n_trajectories):
        #                 ax.plot(dat[comp]['times'], 
        #                     jnp.squeeze(dat[comp]['idata'].posterior_predictive[llike][0,i,:].values), 
        #                     color=dat[comp]['color'], alpha=0.2, linewidth=1.0)
                        
        #         export_legend(leg, save_dir + f'{comp}_ppc_legend_' + sampler + '.pdf')
        #         leg.remove()
                    
        #         ax.set_xlabel("")
        #         ax.set_ylabel("")
        #         ax.set_ylim(0, 1.5)

        #         plt.savefig(save_dir + f'{comp}_{llike}_ppc_' + sampler + '.pdf', transparent=True, bbox_inches='tight')


        ############ plot posterior for each model
        sims = {'cyto':{}, 'lyso':{}, 'mito':{}}

        for comp in dat.keys():
            for pred in ['prediction', 'prediction_LKB1_KO', 'prediction_CaMKK2_KO']:
                fig, ax = get_sized_fig_ax(2,1)

                if pred == 'prediction':
                    data = dat[comp]['data']
                elif pred == 'prediction_LKB1_KO':
                    data = dat[comp]['data_lkb1_ko']
                elif pred == 'prediction_CaMKK2_KO':
                    data = dat[comp]['data_camkk2_ko']

                trajectories = np.squeeze(dat[comp]['idata']['posterior'][pred].values)
        
                fig, ax, leg = plot_predictive(trajectories, data, dat[comp]['times'], 
                                plot_prior=False, add_t_0=True, n_traces=0, figsize=None, 
                                prior_color='', post_color=dat[comp]['color'], data_color=dat[comp]['color'], 
                                data_marker_size=10, fig_ax = (fig, ax), llike_name=pred)
                
                # add n_trajectories to the plot if n_trajectories > 0
                if n_trajectories > 0:
                    for i in range(n_trajectories):
                        ax.plot(dat[comp]['times'], 
                            trajectories[i,:], 
                            color=dat[comp]['color'], alpha=0.2, linewidth=1.0)
                        
                export_legend(leg, save_dir + f'{comp}_ppc_legend_' + sampler + '.pdf')
                leg.remove()
                    
                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.set_ylim(0, 1.5)

                plt.savefig(save_dir + f'{comp}_{pred}_posterior_{sampler}.pdf', 
                            transparent=True, bbox_inches='tight')
                
                sims[comp][pred]=trajectories

                    
    # ####### t-half-max and change in max activation #######
    # # t-half-max
    # # get t_half_max for each condition

    # # calculate time to half max of pAMPKAR_stressed/AMPKAR_stressed
    # def compute_half_max(arr):
    #     half_max = arr.max() / 2
    #     half_max_idx = np.argmin(np.abs(arr - half_max))
    #     return half_max_idx
    
    # t_half_max = {
    #     'cyto':{'baseline':None,'LKB1_KO':None,'CaMKK2_KO':None},
    #     'lyso':{'baseline':None,'LKB1_KO':None,'CaMKK2_KO':None},
    #     'mito':{'baseline':None,'LKB1_KO':None,'CaMKK2_KO':None}
    # }

    # max_dat = {
    #     'cyto':{'baseline':None,'LKB1_KO':None,'CaMKK2_KO':None},
    #     'lyso':{'baseline':None,'LKB1_KO':None,'CaMKK2_KO':None},
    #     'mito':{'baseline':None,'LKB1_KO':None,'CaMKK2_KO':None}
    # }

    # for comp in ['cyto', 'lyso', 'mito']:
    #     t_half_idx = np.apply_along_axis(compute_half_max, 1, sims[comp]['prediction'])
    #     t_half_max[comp]['baseline'] = [dat[comp]['times'][idx] for idx in t_half_idx]
    #     t_half_idx = np.apply_along_axis(compute_half_max, 1, sims[comp]['prediction_LKB1_KO'])
    #     t_half_max[comp]['LKB1_KO'] = [dat[comp]['times'][idx] for idx in t_half_idx]
    #     t_half_idx = np.apply_along_axis(compute_half_max, 1, sims[comp]['prediction_CaMKK2_KO'])
    #     t_half_max[comp]['CaMKK2_KO'] = [dat[comp]['times'][idx] for idx in t_half_idx]

    #     max_dat[comp]['baseline'] = sims[comp]['prediction'][:,-1]
    #     max_dat[comp]['LKB1_KO'] = sims[comp]['prediction_LKB1_KO'][:,-1]
    #     max_dat[comp]['CaMKK2_KO'] = sims[comp]['prediction_CaMKK2_KO'][:,-1]

    # # plot t_half_max
    # fig, ax = get_sized_fig_ax(1.5, 1.0)

    # thalf_df = pd.DataFrame(t_half_max)
    # thalf_df = thalf_df.melt(var_name='compartment', value_name='t_half_max', ignore_index=False)
    # thalf_df = thalf_df.explode('t_half_max')
    # thalf_df = thalf_df.reset_index()
    # thalf_df = thalf_df.rename(columns={'index':'perturbation'})

    # thalf_df['compartment'] = thalf_df['compartment'].replace('cyto', 'cytoplasm')
    # thalf_df['compartment'] = thalf_df['compartment'].replace('lyso', 'lysosome')
    # thalf_df['compartment'] = thalf_df['compartment'].replace('mito', 'mitochondria')

    # sns.barplot(data=thalf_df, x='perturbation', y='t_half_max', hue='compartment', 
    #             ax=ax, palette=colors, linewidth=0, edgecolor='black',width=0.8,
    #             dodge=True, saturation=1.0, gap=0.1)
    
    # for patch in ax.patches:
    #     face_color = patch.get_facecolor()
    #     # Apply transparency to the face color only
    #     patch.set_facecolor(mpl.colors.to_rgba(face_color, alpha=0.5))  
    #     patch.set_edgecolor(face_color)  # Set edge color to match the fill

    # leg = ax.legend(title='', loc='upper left', bbox_to_anchor=(3, 1), 
    #                 fontsize=8.0, ncols=2)
    # export_legend(leg, save_dir + f't_half_max_legend.pdf')
    # leg.remove()

    # ax.set_xticklabels(['WT', 'LKB1 KO', 'CaMKK2 KO'], fontsize=8.0, rotation=45, 
    #                    )
    
    # ax.set_yticks([0, 10])
    # ax.set_yticklabels([0, 10], fontsize=8.0)

    # ax.set_ylabel(r"$T_{1/2}$ (min)", fontsize=10.0)
    # ax.set_xlabel("")

    # plt.savefig(save_dir + f'{model}_t_half_max.pdf', transparent=True, bbox_inches='tight')

    # # plot max activation
    # fig, ax = get_sized_fig_ax(1.5, 1.0)

    # max_df = pd.DataFrame(max_dat)
    # max_df = max_df.melt(var_name='compartment', value_name='max', ignore_index=False)
    # max_df = max_df.explode('max')
    # max_df = max_df.reset_index()
    # max_df = max_df.rename(columns={'index':'perturbation'})
    

    # sns.barplot(data=max_df, x='perturbation', y='max', hue='compartment', 
    #             ax=ax, palette=colors, linewidth=0, edgecolor='black',width=0.8,
    #             dodge=True, saturation=1.0, gap=0.1)
    
    # for patch in ax.patches:
    #     face_color = patch.get_facecolor()
    #     # Apply transparency to the face color only
    #     patch.set_facecolor(mpl.colors.to_rgba(face_color, alpha=0.5))  
    #     patch.set_edgecolor(mpl.colors.to_rgba(face_color, alpha=1.0))  # Set edge color to match the fill

    # # leg = ax.legend(title='', loc='upper left', bbox_to_anchor=(3, 1), fontsize=8.0)
    # # export_legend(leg, save_dir + f't_half_max_legend.pdf')
    # ax.get_legend().remove()
    # # leg.remove()

    # ax.set_xticklabels(['WT', 'LKB1 KO', 'CaMKK KO'], fontsize=8.0, rotation=45, 
    #                    )
    
    # ax.set_yticks([0, 1])
    # ax.set_yticklabels([0, 1.0], fontsize=8.0)
    # ax.set_ylim(0, 1.15)

    # ax.set_ylabel("max. fraction\nactive sensor", fontsize=10.0)
    # ax.set_xlabel("")

    # plt.savefig(save_dir + f'{model}_max_act.pdf', transparent=True, bbox_inches='tight')
                