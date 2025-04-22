import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import met_brewer as mb
import argparse

sys.path.append('../')
sys.path.append('../models/')
from plotting_helper_funcs import *
from utils import *

plt.style.use('~/.matplotlib/stylelib/custom.mplstyle')

def parse_args(raw_args=None):
    """ function to parse command line arguments
    """
    parser=argparse.ArgumentParser(description="Run GSA plotting.")
    # required parameters
    parser.add_argument("-results_path", type=str, default="../../../results/GSA/", help="Path to load/save raw results.")
    parser.add_argument("-fig_path", type=str, default="../../../results/GSA/", help="Path to save figs.")
    args=parser.parse_args(raw_args)
    return args

def main(raw_args=None):
    args = parse_args()

    colors = mb.met_brew(name="Veronese", n=7)
    # colors = colors[0:5]

    # list of models 
    models_free_params = { 
        "MA_single": {'free':["kOffAMP","kOffADP","kOffATP","kOffCaMKK","kPhosCaMKK",
                              "kOffLKB1","kPhosLKB1","kOffPP","kDephosPP","kOffAMPK",
                              "kPhosAMPK","kOffPP1","kDephosPP1"],
                    'names':[r'$k_{OffAMP}$',r'$k_{OffADP}$',
                             r'$k_{OffATP}$',r'$k_{OffCaMKK}$',
                             r'$k_{PhosCaMKK}$',r'$k_{OffLKB1}$',
                             r'$k_{PhosLKB1}$',r'$k_{OffPP}$',
                             r'$k_{DephosPP}$',r'$k_{OffAMPK}$',
                             r'$k_{PhosAMPK}$',r'$k_{OffPP1}$',
                             r'$k_{DephosPP1}$']}, 
         "MM_single":  {'free':["kOffAMP","kOffADP","kOffATP","KmCaMKK",
                                "kLKB1","KmLKB1","kPP","KmPP"],
                     'names':[r'$k_{\text{OffAMP}}$',r'$k_{\text{OffADP}}$',
                              r'$k_{\text{OffATP}}$',
                              r'$K_{m,\text{CaMKK}}$',r'$k_{\text{LKB1}}$',
                              r'$K_{\text{m,LKB1}}$',r'$k_{\text{PP}}$',
                              r'$K_{\text{m,PP}}$'],
                     },
         "MA_nonessential": {'free':["kOffAMP","kOffADP","kOffATP","kOffCaMKK",
                                     "kPhosCaMKK","kOffLKB1","kPhosLKB1","kOffPP",
                                     "kDephosPP","kOffAMPK","kPhosAMPK","kOffPP1",
                                     "kDephosPP1","alphaLKB1","alphaPP","betaAMP"],
                     'names':[r'$k_{\text{OffAMP}}$',r'$k_{\text{OffADP}}$',
                              r'$k_{\text{OffATP}}$',r'$k_{\text{OffCaMKK}}$',
                              r'$k_{\text{PhosCaMKK}}$',r'$k_{\text{OffLKB1}}$',
                              r'$k_{\text{PhosLKB1}}$',r'$k_{\text{OffPP}}$',
                              r'$k_{\text{DephosPP}}$',r'$k_{\text{OffAMPK}}$',
                              r'$k_{\text{PhosAMPK}}$',r'$k_{\text{OffPP1}}$',
                              r'$k_{\text{Dephos,PP1}}$',r'$\alpha_{\text{LKB1}}$',
                              r'$\alpha_{\text{PP}}$',r'$\beta_{\text{AMP}}$'],
                     }, 
         "MM_nonessential":  {'free':["kOffAMP","kOffADP","kOffATP","KmCaMKK",
                                      "kLKB1","KmLKB1","kPP","KmPP","alphaLKB1",
                                      "alphaPP","betaAMP"],
                     'names':[r'$k_{\text{OffAMP}}$',r'$k_{\text{OffADP}}$',
                              r'$k_{\text{OffATP}}$',
                              r'$K_{m,\text{CaMKK}}$',r'$k_{\text{LKB1}}$',
                              r'$K_{\text{m,LKB1}}$',r'$k_{\text{PP}}$',
                              r'$K_{\text{M,PP}}$',r'$\alpha_{\text{LKB1}}$',
                              r'$\alpha_{\text{PP}}$',r'$\beta_{\text{AMP}}$'],
                     },
        "MA_nonessential_all": {'free':["kOffAMP","kOffADP","kOffATP","kOffCaMKK",
                                "kPhosCaMKK", "kOffLKB1","kPhosLKB1","kOffPP",
                                "kDephosPP","kOffAMPK", "kPhosAMPK","kOffPP1",
                                "kDephosPP1","alphaPP","betaAMP","betaLKB1","betaCaMKK"],
                    'names':[r'$k_{OffAMP}$',r'$k_{OffADP}$',
                             r'$k_{OffATP}$',r'$k_{OffCaMKK}$',
                             r'$k_{PhosCaMKK}$',r'$k_{OffLKB1}$',
                             r'$k_{PhosLKB1}$',r'$k_{OffPP}$',
                             r'$k_{DephosPP}$',r'$k_{OffAMPK}$',
                             r'$k_{PhosAMPK}$',r'$k_{OffPP1}$',
                             r'$k_{DephosPP1}$', r'$\alpha_{PP}$',r'$\beta_{AMP}$',
                             r'$\beta_{LKB1}$',r'$\beta_{CaMKK}$']}, 
        "MM_nonessential_all":  {'free':["kOffAMP","kOffADP","kOffATP","KmCaMKK",
                                "kLKB1","KmLKB1","kPP","KmPP","alphaPP",
                                "betaAMPK","betaLKB1","betaCaMKK"],
                    'names':[r'$k_{\text{OffAMP}}$',r'$k_{\text{OffADP}}$',
                                r'$k_{\text{OffATP}}$',
                                r'$K_{m,\text{CaMKK}}$',r'$k_{\text{LKB1}}$',
                                r'$K_{\text{m,LKB1}}$',r'$k_{\text{PP}}$',
                                r'$K_{\text{M,PP}}$',r'$\alpha_{\text{PP}}$',
                                r'$\beta_{\text{AMP}}$',r'$\beta_{\text{LKB1}}$',
                                r'$\beta_{\text{CaMKK}}$']}
        }

    ST_dict = {}

    # loop through each model and analyze GSA results
    for i, model in enumerate(models_free_params.keys()):
        st_sub_dict = {}
        print(model)
        m_name = model.split('_mech')[0] # get the model name w/o _mech
        # we need mech in the model to load GSA sampling results correctly

        fig_path = args.fig_path + m_name + '/'

        # free parameters and nominal values
        free_params = models_free_params[model]['free']
        param_names = models_free_params[model]['names']

        # define the bounds for the AMPK parameters
      
        # load qoi samples
        qoi_samples = np.load(args.results_path  +  m_name + '/'+  m_name + '_qois.npz')
        #  the second entry is the name of the qoi
        qoi_names = {
            "ratio":r'$\frac{[\rm pAMPKAR]}{[\rm AMPKAR]}$', # raw ratio
            "t_half": r'$t_{{\rm 1/2}}$', # time to half max
            "ratio_LKB1_KD": r'$\frac{[\rm pAMPKAR]}{[\rm AMPKAR]} LKB1 KO$',
            "ratio_CaMKK_KD": r'$\frac{[\rm pAMPKAR]}{[\rm AMPKAR]} CaMKK KO$',
            "t_half_LKB1_KD": r'$t_{{\rm 1/2}}$ LKB1 KO',
            "t_half_CaMKK_KD": r'$t_{{\rm 1/2}}$ CaMKK KO',
        }

        for qoi in list(qoi_names.keys()):
            # unpack qoi tuple
            qoi_vals = qoi_samples[qoi]
            qoi_name = qoi_names[qoi]

            # plot histogram of qoi
            fig, ax = get_sized_fig_ax(1.0, 1.0)
            sns.histplot(qoi_vals, ax=ax, kde=True, stat='density', bins=30, 
                        line_kws={'linewidth': 1.0, 'linestyle':'--'},
                        color=colors[i])
            ax.set_xlabel(qoi_name)
            ax.set_ylabel('density')
            fig.savefig(fig_path  + m_name + '_'+ qoi + '_hist.pdf', bbox_inches='tight')

            # load the sesnitivity indices
            sobol_df = pd.read_csv(args.results_path  +  m_name + '/'+  m_name + '_' + qoi + '_sobol_GSA.csv')

            print(sobol_df)

            # fix param_names in the df
            for j, param in enumerate(sobol_df['param']):
                sobol_df.loc[j, 'param_name'] = param_names[free_params.index(param)]

            # # plot sobol indices
            # S1
            fig_width = 4.5
            fig_height = 1.0
            fig, ax = get_sized_fig_ax(fig_width, fig_height)
            sorted = sobol_df.sort_values(by='S1', ascending=False)
            order = list(sorted["param"])

            sns.barplot(x='param', y='S1', data=sobol_df, 
                        ax=ax, order=order, color=colors[i])
            
            # add error bars
            bar_width = ax.patches[0].get_width()
            #Calculate offsets for number of hues provided
            offset = np.linspace(-1/2, 1/2, 1)*bar_width/2 #
            x_dict = dict((x_val,x_pos) for x_pos,x_val in list(enumerate(order)))
            #Map the x-position and offset of each record in the dataset
            x_values = np.array([x_dict[x] for x in sorted["param"]]);
            #Overlay the error bars onto plot
            ax.errorbar(x = x_values, y = sorted["S1"], yerr=sorted["S1_conf"], fmt='none', c= 'black', capsize = 2)

            ax.set_ylabel(r'$S_1$  ' + qoi_name)
            ax.set_xlabel('')
            ax.set_xticklabels(sorted['param_name'], rotation=90, fontsize=8)
            fig.savefig(fig_path + m_name + '_' + qoi + '_S1.pdf', bbox_inches='tight')
   
            # ST
            st_sub_dict[qoi] = dict(zip(sobol_df['param'], sobol_df['ST'])) #/sobol_df['ST'].sum()))
    
            fig, ax = get_sized_fig_ax(fig_width, fig_height)
            sorted = sobol_df.sort_values(by='ST', ascending=False)
            order = list(sorted["param"])
    
            sns.barplot(x='param', y='ST', data=sobol_df, 
                        ax=ax, order=order, color=colors[i])
            
            # add error bars
            bar_width = ax.patches[0].get_width()
            #Calculate offsets for number of hues provided
            offset = np.linspace(-1/2, 1/2, 1)*bar_width/2 #
            x_dict = dict((x_val,x_pos) for x_pos,x_val in list(enumerate(order)))
            #Map the x-position and offset of each record in the dataset
            x_values = np.array([x_dict[x] for x in sorted["param"]]);
            #Overlay the error bars onto plot
            ax.errorbar(x = x_values, y = sorted["ST"], yerr=sorted["ST_conf"], fmt='none', c= 'black', capsize = 2)

            ax.set_ylabel(r'$S_T$  ' + qoi_name)
            ax.set_xlabel('')
            ax.set_xticklabels(sorted['param_name'], rotation=90, fontsize=8)
            # if ST is greater than 0.01, change the color of the xtick labels
            idxs = np.arange(0,len(sorted['ST']),1)[sorted['ST'] >=0.01]  # specify the indices of the xticks to change color
            for tick_label in ax.get_xticklabels():
                if ax.get_xticklabels().index(tick_label) in idxs:
                    tick_label.set_color('red')
            fig.savefig(fig_path + m_name + '_' + qoi + '_ST.pdf', bbox_inches='tight')
        ST_dict[model] = st_sub_dict

    param_function_dict = {
        'AMP binding': ['kOffAMP', 'k8f', 'k8r', 'k11f', 'k11r'],
        'ADP binding': ['kOffADP', 'k7f', 'k7r', 'k10f', 'k10r'],
        'ATP binding':['kOffATP', 'k6f', 'k6r', 'k9f', 'k9r'],
        'AMPK phos.': ['kOffCaMK', 'kPhosCaMKK', 'KmCaMKK', 'kCaMKK','kOffLKB1', 
                  'kPhosLKB1', 'KmLKB1', 'kLKB1', 'Km12', 'Km14', 'Km16', 'Km18', 
                  'Vmaxkinase', 'VmaxkinaseATP', 'VmaxkinaseADP', 'VmaxkinaseAMP', 
                  'alphaLKB1', 'betaLKB1', 'betaCaMKK'],
        'AMPK dephos.': ['kOffPP', 'kDephosPP', 'KmPP', 'kPP', 'Km13', 'Km15', 'Km17', 'Km19',
                'Vmaxppase', 'VmaxppaseATP', 'VmaxppaseADP', 'VmaxppaseAMP', 'alphaPP'],
        'AMPK kinase act.': ['kOffAMPK', 'kPhosAMPK', 'KmAMPK', 'kAMPK', 'Km_pAMPK','k_pAMPK', 
                 'Km_AMP_pAMPK', 'k_AMP_pAMPK', 'Km_ADP_pAMPK', 'k_ADP_pAMPK', 
                 'Km_ATP_pAMPK', 'k_ATP_pAMPK', 'betaAMP', 'betaAMPK'],
        'AMKPAR dephos.': ['kOffPP1', 'kDephosPP1', 'KmPP1', 'kPP1', 'Km_AMPKAR_PP', 'Vmax_AMPKAR_PP']
    }

    model_names = {
        'MA_single': 'Model 1',
        'MM_single': 'Model 2',
        'MA_nonessential': 'Model 3',
        'MM_nonessential': 'Model 4',
        'MA_nonessential_all': 'Model 5',
        'MM_nonessential_all': 'Model 6'
    }

    # Make a heatmap of the ST values
    for qoi in list(qoi_names.keys()):
        tmp = {}

        data_to_plot = {}
        for i, model in enumerate(ST_dict.keys()):
            tmp[model] = ST_dict[model][qoi]

            tmp_dict = {param_func:0 for param_func in param_function_dict.keys()}

            for j, param_func in enumerate(param_function_dict.keys()):
                for param in param_function_dict[param_func]:
                    if param in tmp[model].keys(): # if that parameter is in the model
                        tmp_dict[param_func] += tmp[model][param] # add ST value to the data_to_plot entry

            data_to_plot[model_names[model]] = tmp_dict
        
        idxs_df = pd.DataFrame(data_to_plot)

        idxs_df_long = idxs_df.reset_index().melt(id_vars='index', var_name='model', value_name='idx')
        idxs_df_long.rename(columns={'index': 'param'}, inplace=True)
        print(idxs_df_long)

        fig, ax = get_sized_fig_ax(5.0, 1.5)
        sns.barplot(x='param', y='idx', hue='model', data=idxs_df_long, ax=ax, palette=colors)
        ax.set_ylabel(r'total sensitivity index', fontsize=10.0)
        # ax.set_ylabel('')
        ax.set_xlabel('')
        ax.set_xticklabels(list(param_function_dict.keys()), rotation=45, ha='right', fontsize=10.0)

        # update coloring
        for patch in ax.patches:
            face_color = patch.get_facecolor()
            # Apply transparency to the face color only
            patch.set_facecolor(mpl.colors.to_rgba(face_color, alpha=0.8))  

            patch.set_edgecolor(face_color)  # Set edge color to match the fill

        # legend and remove it
        leg = ax.legend(loc='upper right', bbox_to_anchor=(1.75, 1.0), fontsize=8.0, ncols=1, title='')
        export_legend(leg, args.fig_path + 'ST_legend.pdf')

        ax.set_ylim(0, 1.1)

        leg.remove()
        # save the figure
        fig.savefig(args.fig_path + 'ST_barplot_' + qoi + '.pdf', bbox_inches='tight', transparent=True)
    
if __name__ == "__main__":
    main()













