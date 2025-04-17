#! /bin/zsh
# script to run inference for all models
# MA single, cytosol
# NUTS with default init
python inference_pymc.py -model MA_single -compartment cyto -free_params kOffAMP,kOffADP,kOffATP,kOffPP,kPhosAMPK,kOffAMPK,kOffPP1,kDephosPP1 -data_file ../../../Schmitt_et_al_2022_data/fig_2e_cyto.npz -model_info_file ../models/MA_single.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nwarmup 300 -nsamples 1000 -nchains 1 -sampler Nutpie -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 --sample_posterior

python inference_pymc.py -model MA_single -compartment cyto -free_params kOffAMP,kOffADP,kOffATP,kOffPP,kPhosAMPK,kOffAMPK,kOffPP1,kDephosPP1 -data_file ../../../Schmitt_et_al_2022_data/fig_2e_cyto.npz -model_info_file ../models/MA_single.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nsamples 1000 -nchains 1 -sampler Pathfinder -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 --sample_posterior --compute_llike

# # ADVI with mean field approximation
# python inference_pymc.py -model MA_single -compartment cyto -free_params kOffAMP,kOffADP,kOffATP,kOffPP,kPhosAMPK,kOffAMPK,kOffPP1,kDephosPP1 -data_file ../../../Schmitt_et_al_2022_data/fig_2e_cyto.npz -model_info_file ../models/MA_single.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nsamples 4000 -nchains 4 -sampler ADVI -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 -n_advi_iter 150000 --sample_posterior

# MA single, lysosome
python inference_pymc.py -model MA_single -compartment lyso -free_params kOffAMP,kOffADP,kOffATP,kOffPP,kPhosAMPK,kOffAMPK,kOffPP1,kDephosPP1 -data_file ../../../Schmitt_et_al_2022_data/fig_2b_lyso.npz -model_info_file ../models/MA_single.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nwarmup 300 -nsamples 1000 -nchains 1 -sampler Nutpie -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 --sample_posterior

python inference_pymc.py -model MA_single -compartment lyso -free_params kOffAMP,kOffADP,kOffATP,kOffPP,kPhosAMPK,kOffAMPK,kOffPP1,kDephosPP1 -data_file ../../../Schmitt_et_al_2022_data/fig_2b_lyso.npz -model_info_file ../models/MA_single.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nsamples 1000 -nchains 1 -sampler Pathfinder -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 --sample_posterior --compute_llike

# # ADVI with mean field approximation
# python inference_pymc.py -model MA_single -compartment lyso -free_params kOffAMP,kOffADP,kOffATP,kOffPP,kPhosAMPK,kOffAMPK,kOffPP1,kDephosPP1 -data_file ../../../Schmitt_et_al_2022_data/fig_2b_lyso.npz -model_info_file ../models/MA_single.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nsamples 4000 -nchains 4 -sampler ADVI -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 -n_advi_iter 150000 --sample_posterior

# MA single, mitochondria
# NUTS with default init
# python inference_pymc.py -model MA_single -compartment mito -free_params kOffAMP,kOffADP,kOffATP,kOffPP,kPhosAMPK,kOffAMPK,kOffPP1,kDephosPP1 -data_file ../../../Schmitt_et_al_2022_data/fig_2c_mito.npz -model_info_file ../models/MA_single.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nwarmup 300 -nsamples 1000 -nchains 1 -sampler Nutpie -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 --sample_posterior

python inference_pymc.py -model MA_single -compartment mito -free_params kOffAMP,kOffADP,kOffATP,kOffPP,kPhosAMPK,kOffAMPK,kOffPP1,kDephosPP1 -data_file ../../../Schmitt_et_al_2022_data/fig_2c_mito.npz -model_info_file ../models/MA_single.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nwarmup 300 -nsamples 1000 -nchains 1 -sampler Nutpie -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 --sample_posterior

python inference_pymc.py -model MA_single -compartment mito -free_params kOffAMP,kOffADP,kOffATP,kOffPP,kPhosAMPK,kOffAMPK,kOffPP1,kDephosPP1 -data_file ../../../Schmitt_et_al_2022_data/fig_2c_mito.npz -model_info_file ../models/MA_single.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nsamples 1000 -nchains 1 -sampler Pathfinder -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 --sample_posterior

# ADVI with mean field approximation
python inference_pymc.py -model MA_single -compartment mito -free_params kOffAMP,kOffADP,kOffATP,kOffPP,kPhosAMPK,kOffAMPK,kOffPP1,kDephosPP1 -data_file ../../../Schmitt_et_al_2022_data/fig_2c_mito.npz -model_info_file ../models/MA_single.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nsamples 4000 -nchains 4 -sampler ADVI -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 -n_advi_iter 150000 --sample_posterior

##################################################
## GSA Uninformed -ie all parameters are free
# MA single, cytosol
# NUTS with default init
python inference_pymc.py -model MA_single -compartment cyto -free_params kOffAMP,kOffADP,kOffATP,kPhosLKB1,kOffLKB1,kPhosCaMKK,kOffCaMKK,kOffPP,kDephosPP,kPhosAMPK,kOffAMPK,kOffPP1,kDephosPP1 -data_file ../../../Schmitt_et_al_2022_data/fig_2e_cyto.npz -model_info_file ../models/MA_single.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/all_free/ -nwarmup 300 -nsamples 1000 -nchains 4 -sampler Nutpie -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 --sample_posterior

# ADVI with mean field approximation
python inference_pymc.py -model MA_single -compartment cyto -free_params kOffAMP,kOffADP,kOffATP,kPhosLKB1,kOffLKB1,kPhosCaMKK,kOffCaMKK,kOffPP,kDephosPP,kPhosAMPK,kOffAMPK,kOffPP1,kDephosPP1 -data_file ../../../Schmitt_et_al_2022_data/fig_2e_cyto.npz -model_info_file ../models/MA_single.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/all_free/ -nsamples 4000 -nchains 4 -sampler ADVI -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 -n_advi_iter 60000 --sample_posterior

# MA single, lysosome
# NUTS with default init
python inference_pymc.py -model MA_single -compartment lyso -free_params kOffAMP,kOffADP,kOffATP,kPhosLKB1,kOffLKB1,kPhosCaMKK,kOffCaMKK,kOffPP,kDephosPP,kPhosAMPK,kOffAMPK,kOffPP1,kDephosPP1 -data_file ../../../Schmitt_et_al_2022_data/fig_2b_lyso.npz -model_info_file ../models/MA_single.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/all_free/ -nwarmup 300 -nsamples 1000 -nchains 4 -sampler Nutpie -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 --sample_posterior

# ADVI with mean field approximation
python inference_pymc.py -model MA_single -compartment lyso -free_params kOffAMP,kOffADP,kOffATP,kPhosLKB1,kOffLKB1,kPhosCaMKK,kOffCaMKK,kOffPP,kDephosPP,kPhosAMPK,kOffAMPK,kOffPP1,kDephosPP1 -data_file ../../../Schmitt_et_al_2022_data/fig_2b_lyso.npz -model_info_file ../models/MA_single.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/all_free/ -nsamples 4000 -nchains 4 -sampler ADVI -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 -n_advi_iter 60000 --sample_posterior

# MA single, mitochondria
# NUTS with default init
python inference_pymc.py -model MA_single -compartment mito -free_params kOffAMP,kOffADP,kOffATP,kPhosLKB1,kOffLKB1,kPhosCaMKK,kOffCaMKK,kOffPP,kDephosPP,kPhosAMPK,kOffAMPK,kOffPP1,kDephosPP1 -data_file ../../../Schmitt_et_al_2022_data/fig_2c_mito.npz -model_info_file ../models/MA_single.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/all_free/ -nwarmup 300 -nsamples 1000 -nchains 4 -sampler Nutpie -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 --sample_posterior

# ADVI with mean field approximation
python inference_pymc.py -model MA_single -compartment mito -free_params kOffAMP,kOffADP,kOffATP,kPhosLKB1,kOffLKB1,kPhosCaMKK,kOffCaMKK,kOffPP,kDephosPP,kPhosAMPK,kOffAMPK,kOffPP1,kDephosPP1 -data_file ../../../Schmitt_et_al_2022_data/fig_2c_mito.npz -model_info_file ../models/MA_single.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/all_free/ -nsamples 4000 -nchains 4 -sampler ADVI -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 -n_advi_iter 60000 --sample_posterior