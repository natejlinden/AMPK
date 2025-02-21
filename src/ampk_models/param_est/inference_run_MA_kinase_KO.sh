#! /bin/zsh
# script to run inference for all models
# MA single, cytosol
# NUTS with default init
python inference_kinase_KO_pymc.py -model MA_single -compartment cyto -free_params kOffAMP,kOffADP,kOffATP,kOffPP,kOffCaMKK,kPhosCaMKK,kOffLKB1,kPhosLKB1,kPhosAMPK,kOffAMPK,kOffPP1,kDephosPP1 -data_file ../../../Schmitt_et_al_2022_data/fig_2e_cyto.npz -LKB1_KO_data_file ../../../Schmitt_et_al_2022_data/fig_2e_cyto_LKB1_KD.npz -CaMKK2_KO_data_file ../../../Schmitt_et_al_2022_data/sup_fig_2g_cyto_CaMKK_KD.npz -model_info_file ../models/MA_single.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/kinase_KO/ -nsamples 4000 -sampler Pathfinder -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 --sample_posterior

# MA single, lysosome
python inference_LKB1_knockdown_pymc.py -model MA_single -compartment lyso -free_params kOffAMP,kOffADP,kOffATP,kOffPP,kOffCaMKK,kPhosCaMKK,kOffLKB1,kPhosLKB1,kPhosAMPK,kOffAMPK,kOffPP1,kDephosPP1 -data_file ../../../Schmitt_et_al_2022_data/fig_2b_lyso.npz -LKB1_KD_data_file ../../../Schmitt_et_al_2022_data/fig_2f_lyso_LKB1_KD.npz -CaMKK2_KO_data_file ../../../Schmitt_et_al_2022_data/sup_fig_2g_lyso_CaMKK_KD.npz -model_info_file ../models/MA_single.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/kinase_KO/ -nsamples 4000 -sampler Pathfinder -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 --sample_posterior

# MA single, mitochondria
python inference_LKB1_knockdown_pymc.py -model MA_single -compartment mito -free_params kOffAMP,kOffADP,kOffATP,kOffPP,kOffCaMKK,kPhosCaMKK,kOffLKB1,kPhosLKB1,kPhosAMPK,kOffAMPK,kOffPP1,kDephosPP1 -data_file ../../../Schmitt_et_al_2022_data/fig_2c_mito.npz -LKB1_KD_data_file ../../../Schmitt_et_al_2022_data/fig_2g_mito_LKB1_KD.npz -CaMKK2_KO_data_file ../../../Schmitt_et_al_2022_data/sup_fig_2g_mito_CaMKK_KD.npz -model_info_file ../models/MA_single.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/kinase_KO/ -nsamples 4000 -sampler Pathfinder -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 --sample_posterior
