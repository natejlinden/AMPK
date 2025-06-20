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
colors = mb.met_brew(name="Egypt", n=4)

print(colors)

samplers = ["Pathfinder"] # list of samplers used

models_free_params = { 
        "MA_single": {'info_file':'../models/MA_single.json', 'name':'1'},
        "MM_single":  {'info_file': '../models/MM_single.json', 'name':'2'},
        "MA_nonessential": {'info_file': '../models/MA_nonessential.json', 'name':'3'},
        "MM_nonessential":  {'info_file': '../models/MM_nonessential.json', 'name':'4'},
        "MA_nonessential_all": {'info_file': '../models/MA_nonessential_all.json', 'name':'5'},
        "MM_nonessential_all":  {'info_file': '../models/MM_nonessential_all.json', 'name':'6'}
        }

data_dir = '../../../results/param_est/kinase_KO/std_dcr/'
save_dir_base = '../../../results/param_est/kinase_KO/std_dcr/figs/'

# dictionary to store idata
idata_dict_cyto = {model:{sampler:None for sampler in samplers} for model in models_free_params.keys()}
llike_dict_cyto = {model:{sampler:None for sampler in samplers} for model in models_free_params.keys()}

for i, model in enumerate(models_free_params.keys()):
    
    print(f"Processing model {i+1}: {model}")

    for sampler in samplers:
        save_dir = save_dir_base + model + '/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # load the llike idata
        idata_cyto = az.from_netcdf(data_dir + model + '_cyto_mcmc_samples_llike_' + sampler + '.nc')

        # store idata
        idata_dict_cyto[model][sampler] = idata_cyto
############ compare models using ELPD

print(idata_cyto['log_likelihood'])

llike_names = {'llike_WT':'Wild type', 
               'llike_LKB1_KO':'LKB1_KO', 
               'llike_CaMKK2_KO':'CaMKK2_KO'}

# repeat for each sampler
for sampler in samplers:
    elpds = []
    for llike_name in ['llike_WT', 'llike_LKB1_KO', 'llike_CaMKK2_KO']:
        cyto_dat = {model:idata_dict_cyto[model][sampler] for model in idata_dict_cyto.keys()}

        # run az.compare
        cyto_compare = az.compare(cyto_dat, var_name=llike_name)

        cyto_compare = cyto_compare[['elpd_loo']].reset_index()
        cyto_compare.rename(columns={'index': 'model'}, inplace=True)
        cyto_compare['llike'] = llike_names[llike_name]  # Add a new column for 'llike'

        # store elpds
        elpds.append(cyto_compare)    
    # combine into one dataframe
    elpd = pd.concat(elpds, ignore_index=True)

    # Calculate the sum of ELPD for each model over the unique values in llike
    elpd_sum = elpd.groupby('model')['elpd_loo'].sum().reset_index()
    elpd_sum['llike'] = 'Total ELPD'  # Add a new column for 'all'

    # Append the summed ELPD as a new row for each model
    elpd = pd.concat([elpd, elpd_sum], ignore_index=True)

    # Replace model names in the 'model' column with their corresponding 'name' from models_free_params
    elpd['model'] = elpd['model'].map(lambda x: models_free_params[x]['name'])

    print(elpd)

    # Ensure the models are ordered according to the keys in models_free_params
    elpd['model'] = pd.Categorical(elpd['model'], 
            categories=[models_free_params[mod]['name'] for mod in list(models_free_params.keys())], 
            ordered=True)


    # Create a clustered bar plot using seaborn
    fig, ax = get_sized_fig_ax(width=4, height=2.)

    sns.barplot(data=elpd, x='model', y='elpd_loo', hue='llike', ax=ax,
                palette=colors[:elpd['llike'].nunique()], edgecolor='black', alpha=0.8)

    # Customize the plot
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.set_ylabel('ELPD\n(Expected Log Predictive Density)', fontsize=10)
    ax.set_xlabel('Model', fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=8)

    leg = ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.25, 1),
                    ncols=4)

    export_legend(leg, save_dir_base + f'elpd_legend_{sampler}.pdf')
    leg.remove()
    ylim = ax.get_ylim()

    fig.savefig(save_dir_base + f'elpd_{sampler}.pdf', bbox_inches='tight', transparent=True)

    # Create separate plots for each unique value in the 'llike' column
    unique_llikes = elpd['llike'].unique()

    for i, llike in enumerate(unique_llikes):
        # Filter the dataframe for the current llike
        elpd_filtered = elpd[elpd['llike'] == llike]

        # Create a bar plot for the current llike
        fig, ax = get_sized_fig_ax(width=1., height=2.)

        sns.barplot(data=elpd_filtered, x='model', y='elpd_loo', ax=ax,
                    color=colors[i], edgecolor='black', alpha=0.8)

        # Customize the plot
        ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
        ax.set_ylabel('ELPD\n(Expected Log Predictive Density)', fontsize=10)
        ax.set_xlabel('Model', fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.set_ylim(ylim)  # Set the same y-axis limits as the main plot

        # Save the plot
        fig.savefig(save_dir_base + f'elpd_{sampler}_{llike}.pdf', bbox_inches='tight', transparent=True)
        plt.close(fig)

    # # Save the plot
    # fig.tight_layout()
    # fig.savefig(save_dir_base + f'clustered_bar_elpd_{sampler}.pdf', bbox_inches='tight', transparent=True)
    # plt.close(fig)

    # # Pivot the data to prepare for stacking
    # elpd_pivot = elpd.pivot(index='model', columns='llike', values='elpd_loo').fillna(0)


    #     # combine into one dataframe with compartment as a column
    #     cyto_compare['compartment'] = 'cyto'

    #     compare = pd.concat([cyto_compare]).loc[:, ['elpd_loo', 'compartment']].reset_index()
    #     compare = compare.rename(columns={'index': 'model'})

    #     elpd[llike_name] = compare

    #     # make a plot
    #     fig, ax = get_sized_fig_ax(1.5, 2)

    #     ax.axhline(0, color='black', linewidth=0.5, linestyle='--')

    #     sns.barplot(data=compare, x='compartment', y='elpd_loo', hue='model', ax=ax,
    #                 palette=colors, linewidth=1,
    #                 edgecolor='black')

    #     for patch in ax.patches:
    #         face_color = patch.get_facecolor()
    #         # Apply transparency to the face color only
    #         patch.set_facecolor(mpl.colors.to_rgba(face_color, alpha=0.8))  

    #         patch.set_edgecolor(face_color)  # Set edge color to match the fill

    #     # Create Custom Legend Handles to Match Bar Styles
    #     handles = []
    #     # unique_groups = ['Michaelis-Menten (MM)', 'MM nonessential', 'Mass action (MA)', 'MA nonessential']
    #     unique_groups = ['MM nonessential', 'Mass action (MA)', 'MA nonessential']
    #     palette = colors

    #     for color, group in zip(palette, unique_groups):
    #         handles.append(mpl.patches.Patch(facecolor=mpl.colors.to_rgba(color, alpha=0.8), 
    #                                         edgecolor=color, linewidth=1., label=group))

    #     # Apply the Custom Legend
    #     leg = ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.5, 1),
    #             fontsize=8, title_fontsize=10)
    #     export_legend(leg, save_dir_base + 'elpd_legend.pdf')
    #     leg.remove()

    #     ax.set_ylabel('expected log pointwise\npredictive density\n(predictive accuracy)', fontsize=10)
    #     ax.set_xlabel('sub-cellular compartment', fontsize=10)
    #     ax.tick_params(axis='both', which='major', labelsize=8)
    #     ax.set_xticklabels(['cytoplasm', 'lysosome', 'mitochondria'])

    #     fig.savefig(save_dir_base + 'elpd_' + sampler +'_' + llike_name + '.pdf', bbox_inches='tight',
    #                 transparent=True)
        
    # # average ELPD over three datasets
    # elpd_combined = pd.concat([df.assign(llike=key) for key, df in elpd.items()])

    # elpd_combined_avg = elpd_combined.groupby(['model', 'compartment'])['elpd_loo'].mean().reset_index()
    # elpd_combined_avg = elpd_combined_avg.rename(columns={'elpd_loo': 'avg_elpd_loo'})
    # print(elpd_combined_avg)

    # fig, ax = get_sized_fig_ax(2.5, 2)

    # ax.axhline(0, color='black', linewidth=0.5, linestyle='--')

    # sns.barplot(data=elpd_combined_avg, x='compartment', y='avg_elpd_loo', hue='model', ax=ax,
    #             palette=colors, linewidth=1,
    #             edgecolor='black')

    # for patch in ax.patches:
    #     face_color = patch.get_facecolor()
    #     patch.set_facecolor(mpl.colors.to_rgba(face_color, alpha=0.8))  
    #     patch.set_edgecolor(face_color)

    # handles = []
    # unique_groups = ['MM nonessential', 'Mass action (MA)', 'MA nonessential']
    # palette = colors

    # for color, group in zip(palette, unique_groups):
    #     handles.append(mpl.patches.Patch(facecolor=mpl.colors.to_rgba(color, alpha=0.8), 
    #                                     edgecolor=color, linewidth=1., label=group))

    # leg = ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.5, 1),
    #         fontsize=8, title_fontsize=10)
    # export_legend(leg, save_dir_base + 'elpd_combined_legend.pdf')
    # leg.remove()

    # ax.set_ylabel('expected log pointwise\npredictive density\n(predictive accuracy)', fontsize=10)
    # ax.set_xlabel('sub-cellular compartment', fontsize=10)
    # ax.tick_params(axis='both', which='major', labelsize=8)
    # ax.set_xticklabels(['cytoplasm', 'lysosome', 'mitochondria'])

    # fig.savefig(save_dir_base + 'elpd_combined_' + sampler + '.pdf', bbox_inches='tight',
    #             transparent=True)