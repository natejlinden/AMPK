#! /bin/zsh
# script to run inference for all models

# MA single, cytosol
# NUTS with default init
XLA_FLAGS="--xla_force_host_platform_device_count=1" python inference_pymc.py -model MA_single -compartment cyto -free_params kOffAMP,kOffADP,kOffATP,kOffAMPK,kPhosAMPK,kOffPP1,kDephosPP1 -data_file ../../../Schmitt_et_al_2022_data/fig_2e_cyto.npz -model_info_file ../models/MA_single.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nwarmup 3000 -nsamples 3000 -nchains 4 -sampler NUTS -lower_mult 1e-2 -upper_mult 1e4 -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 --sample_posterior

# ADVI with mean field approximation
XLA_FLAGS="--xla_force_host_platform_device_count=1" python inference_pymc.py -model MA_single -compartment cyto -free_params kOffAMP,kOffADP,kOffATP,kOffAMPK,kPhosAMPK,kOffPP1,kDephosPP1 -data_file ../../../Schmitt_et_al_2022_data/fig_2e_cyto.npz -model_info_file ../models/MA_single.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nsamples 4000 -nchains 4 -sampler ADVI -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 -n_advi_iter 60000 --sample_posterior -log_transform_bounds True

XLA_FLAGS="--xla_force_host_platform_device_count=1" python inference_pymc.py -model MA_single -compartment cyto -free_params kOffAMP,kOffADP,kOffATP,kOffAMPK,kPhosAMPK,kOffPP1,kDephosPP1 -data_file ../../../Schmitt_et_al_2022_data/fig_2e_cyto.npz -model_info_file ../models/MA_single.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 nwarmup 300 -nsamples 1000 -nchains 1 -sampler NUTS --sample_posterior -log_transform_bounds True

# MA single, lysosome
# NUTS with default init
XLA_FLAGS="--xla_force_host_platform_device_count=1" python inference_pymc.py -model MA_single -compartment lyso -free_params kOffAMP,kOffADP,kOffATP,kOffAMPK,kPhosAMPK,kOffPP1,kDephosPP1 -data_file ../../../Schmitt_et_al_2022_data/fig_2b_lyso.npz -model_info_file ../models/MA_single.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nwarmup 3000 -nsamples 3000 -nchains 4 -sampler NUTS -lower_mult 1e-2 -upper_mult 1e4 -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 --sample_posterior

# ADVI with mean field approximation
XLA_FLAGS="--xla_force_host_platform_device_count=1" python inference_pymc.py -model MA_single -compartment lyso -free_params kOffAMP,kOffADP,kOffATP,kOffAMPK,kPhosAMPK,kOffPP1,kDephosPP1 -data_file ../../../Schmitt_et_al_2022_data/fig_2b_lyso.npz -model_info_file ../models/MA_single.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nsamples 4000 -nchains 4 -sampler ADVI -lower_mult 1e-2 -upper_mult 1e4 -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 -n_advi_iter 60000 --sample_posterior

# MA single, mitochondria
# NUTS with default init
XLA_FLAGS="--xla_force_host_platform_device_count=1" python inference_pymc.py -model MA_single -compartment mito -free_params kOffAMP,kOffADP,kOffATP,kOffAMPK,kPhosAMPK,kOffPP1,kDephosPP1 -data_file ../../../Schmitt_et_al_2022_data/fig_2c_mito.npz -model_info_file ../models/MA_single.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nwarmup 3000 -nsamples 3000 -nchains 4 -sampler NUTS -lower_mult 1e-2 -upper_mult 1e4 -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 --sample_posterior

# ADVI with mean field approximation
XLA_FLAGS="--xla_force_host_platform_device_count=1" python inference_pymc.py -model MA_single -compartment mito -free_params kOffAMP,kOffADP,kOffATP,kOffAMPK,kPhosAMPK,kOffPP1,kDephosPP1 -data_file ../../../Schmitt_et_al_2022_data/fig_2c_mito.npz -model_info_file ../models/MA_single.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nsamples 4000 -nchains 4 -sampler ADVI -lower_mult 1e-2 -upper_mult 1e4 -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 -n_advi_iter 60000 --sample_posterior

# MA single, cytosol with delta_ratio normalization
XLA_FLAGS="--xla_force_host_platform_device_count=1" python inference_pymc.py -model MA_single -compartment cyto -free_params kOffAMP,kOffADP,kOffATP,kOffAMPK,kPhosAMPK,kOffPP1,kDephosPP1 -data_file ../../../Schmitt_et_al_2022_data/fig_2e_cyto.npz -model_info_file ../models/MA_single.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/delta_norm_ -nsamples 4000 -nchains 4 -sampler ADVI -lower_mult 1e-2 -upper_mult 1e4 -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 -n_advi_iter 60000 --sample_posterior --sample_prior -normalization delta_ratio

XLA_FLAGS="--xla_force_host_platform_device_count=1" python inference_pymc.py -model MA_single -compartment cyto -free_params kOffAMP,kOffADP,kOffATP,kOffAMPK,kPhosAMPK,kOffPP1,kDephosPP1 -data_file ../../../Schmitt_et_al_2022_data/fig_2e_cyto.npz -model_info_file ../models/MA_single.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/delta_norm_ -nwarmup 3000 -nsamples 3000 -nchains 4 -sampler NUTS -lower_mult 1e-2 -upper_mult 1e4 -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 -n_advi_iter 60000 --sample_posterior --sample_prior -normalization delta_ratio

# tscc test
python inference_pymc.py -model MA_single -compartment cyto -free_params kOffAMP,kOffADP,kOffATP,kOffAMPK,kPhosAMPK,kOffPP1,kDephosPP1 -data_file ../../../Schmitt_et_al_2022_data/fig_2e_cyto.npz -model_info_file ../models/MA_single.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir /tscc/lustre/ddn/scratch/nlinden/ampk/results/param_est/ -nwarmup 3000 -nsamples 3000 -nchains 4 -sampler NUTS -lower_mult 1e-2 -upper_mult 1e4 -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 --sample_posterior


# MA single 
python inference_MA_single.py -compartment cyto -data_file ../../../Schmitt_et_al_2022_data/fig_2e_cyto.npz -model_info_file ../models/MA_single.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nwarmup 3000 -nsamples 3000 -nchains 4 -sampler NUTS -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 --sample_prior

XLA_FLAGS="--xla_force_host_platform_device_count=1" python inference_pymc.py -model MA_double -compartment cyto -free_params kOffAMP,kOffADP,kOffATP,kOffCaMKK,kPhosCaMKK,kOffLKB1,kPhosLKB1,kOffPP,kDephosPP,kOffAMPK,kPhosAMPK,kOffPP1,kDephosPP1 -data_file ../../../Schmitt_et_al_2022_data/fig_2e_cyto.npz -model_info_file ../models/MA_double.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 -nwarmup 300 -nsamples 1000 -nchains 1 -sampler NUTS --sample_posterior --sample_prior

XLA_FLAGS="--xla_force_host_platform_device_count=1" python inference_pymc.py -model MA_double -compartment cyto -free_params kOffAMP,kOffADP,kOffATP,kOffCaMKK,kPhosCaMKK,kOffLKB1,kPhosLKB1,kOffPP,kDephosPP,kOffAMPK,kPhosAMPK,kOffPP1,kDephosPP1 -data_file ../../../Schmitt_et_al_2022_data/fig_2e_cyto.npz -model_info_file ../models/MA_double.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/delta_ratio/ -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 -nwarmup 300 -nsamples 1000 -nchains 1 -sampler NUTS --sample_posterior --sample_prior -normalization delta_ratio