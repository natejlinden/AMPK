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
colors = colors[0:4]

print(colors)

samplers = ["Pathfinder"] #,"Nutpie", "NUTS"] # list of samplers used
n_trajectories = 10 # number of trajectories to add to PPC plot

cyto_color = colors[0]
lyso_color = colors[1]
mito_color = colors[2]


models_free_params = { 
        # "ampk_Coccimiglio": {'info_file': '../models/ampk_Coccimiglio.json'},
        "MA_single": {'info_file':'../models/MA_single.json'},
        # "MM_single":  {'info_file': '../models/MM_single.json'},
        "MA_nonessential": {'info_file': '../models/MA_nonessential.json'},
        "MM_nonessential":  {'info_file': '../models/MM_nonessential.json'}
        }

data_dir = '../../../results/param_est/kinase_KO/std_dcr/'
save_dir_base = '../../../results/param_est/kinase_KO/std_dcr/figs/'

# dictionary to store idata
idata_dict = {
    'cyto':{model:{sampler:None for sampler in samplers} for model in models_free_params.keys()},
    'lyso':{model:{sampler:None for sampler in samplers} for model in models_free_params.keys()},
    'mito':{model:{sampler:None for sampler in samplers} for model in models_free_params.keys()}
}

llike_dict = {
    'cyto':{model:{sampler:None for sampler in samplers} for model in models_free_params.keys()},
    'lyso':{model:{sampler:None for sampler in samplers} for model in models_free_params.keys()},
    'mito':{model:{sampler:None for sampler in samplers} for model in models_free_params.keys()}
}

for i, model in enumerate(models_free_params.keys()):
    
    print(f"Processing model {i+1}: {model}")

    for sampler in samplers:
    
        save_dir = save_dir_base + model + '/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # load the llike idata
        idata_cyto = az.from_netcdf(data_dir + model + '_cyto_mcmc_samples_llike_' + sampler + '.nc')
        idata_lyso = az.from_netcdf(data_dir + model + '_lyso_mcmc_samples_llike_' + sampler + '.nc')
        idata_mito = az.from_netcdf(data_dir + model + '_mito_mcmc_samples_llike_' + sampler + '.nc')

        # store idata
        idata_dict['cyto'][model][sampler] = idata_cyto
        idata_dict['lyso'][model][sampler] = idata_lyso
        idata_dict['mito'][model][sampler] = idata_mito

############ compare models using ELPD

# repeat for each sampler
for sampler in samplers:
    elpd = {}
    for llike_name in ['llike', 'llike_LKB1_KO', 'llike_CaMKK2_KO']:
        cyto_dat = {model:idata_dict['cyto'][model][sampler] for model in idata_dict['cyto'].keys()}
        lyso_dat = {model:idata_dict['lyso'][model][sampler] for model in idata_dict['lyso'].keys()}
        mito_dat = {model:idata_dict['mito'][model][sampler] for model in idata_dict['mito'].keys()}

        # run az.compare
        cyto_compare = az.compare(cyto_dat, var_name=llike_name)
        lyso_compare = az.compare(lyso_dat, var_name=llike_name)
        mito_compare = az.compare(mito_dat, var_name=llike_name)

        # combine into one dataframe with compartment as a column
        cyto_compare['compartment'] = 'cyto'
        lyso_compare['compartment'] = 'lyso'
        mito_compare['compartment'] = 'mito'

        compare = pd.concat([cyto_compare, lyso_compare, mito_compare]).loc[:, ['elpd_loo', 'compartment']].reset_index()
        compare = compare.rename(columns={'index': 'model'})

        elpd[llike_name] = compare

        # make a plot
        fig, ax = get_sized_fig_ax(1.5, 2)

        ax.axhline(0, color='black', linewidth=0.5, linestyle='--')

        sns.barplot(data=compare, x='compartment', y='elpd_loo', hue='model', ax=ax,
                    palette=colors, linewidth=1,
                    edgecolor='black')

        for patch in ax.patches:
            face_color = patch.get_facecolor()
            # Apply transparency to the face color only
            patch.set_facecolor(mpl.colors.to_rgba(face_color, alpha=0.8))  

            patch.set_edgecolor(face_color)  # Set edge color to match the fill

        # Create Custom Legend Handles to Match Bar Styles
        handles = []
        # unique_groups = ['Michaelis-Menten (MM)', 'MM nonessential', 'Mass action (MA)', 'MA nonessential']
        unique_groups = ['MM nonessential', 'Mass action (MA)', 'MA nonessential']
        palette = colors

        for color, group in zip(palette, unique_groups):
            handles.append(mpl.patches.Patch(facecolor=mpl.colors.to_rgba(color, alpha=0.8), 
                                            edgecolor=color, linewidth=1., label=group))

        # Apply the Custom Legend
        leg = ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.5, 1),
                fontsize=8, title_fontsize=10)
        export_legend(leg, save_dir_base + 'elpd_legend.pdf')
        leg.remove()

        ax.set_ylabel('expected log pointwise\npredictive density\n(predictive accuracy)', fontsize=10)
        ax.set_xlabel('sub-cellular compartment', fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.set_xticklabels(['cytoplasm', 'lysosome', 'mitochondria'])

        fig.savefig(save_dir_base + 'elpd_' + sampler +'_' + llike_name + '.pdf', bbox_inches='tight',
                    transparent=True)
        
    # average ELPD over three datasets
    elpd_combined = pd.concat([df.assign(llike=key) for key, df in elpd.items()])

    elpd_combined_avg = elpd_combined.groupby(['model', 'compartment'])['elpd_loo'].mean().reset_index()
    elpd_combined_avg = elpd_combined_avg.rename(columns={'elpd_loo': 'avg_elpd_loo'})
    print(elpd_combined_avg)

    fig, ax = get_sized_fig_ax(2.5, 2)

    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')

    sns.barplot(data=elpd_combined_avg, x='compartment', y='avg_elpd_loo', hue='model', ax=ax,
                palette=colors, linewidth=1,
                edgecolor='black')

    for patch in ax.patches:
        face_color = patch.get_facecolor()
        patch.set_facecolor(mpl.colors.to_rgba(face_color, alpha=0.8))  
        patch.set_edgecolor(face_color)

    handles = []
    unique_groups = ['MM nonessential', 'Mass action (MA)', 'MA nonessential']
    palette = colors

    for color, group in zip(palette, unique_groups):
        handles.append(mpl.patches.Patch(facecolor=mpl.colors.to_rgba(color, alpha=0.8), 
                                        edgecolor=color, linewidth=1., label=group))

    leg = ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.5, 1),
            fontsize=8, title_fontsize=10)
    export_legend(leg, save_dir_base + 'elpd_combined_legend.pdf')
    leg.remove()

    ax.set_ylabel('expected log pointwise\npredictive density\n(predictive accuracy)', fontsize=10)
    ax.set_xlabel('sub-cellular compartment', fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.set_xticklabels(['cytoplasm', 'lysosome', 'mitochondria'])

    fig.savefig(save_dir_base + 'elpd_combined_' + sampler + '.pdf', bbox_inches='tight',
                transparent=True)