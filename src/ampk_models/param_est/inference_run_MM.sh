#! /bin/zsh
# script to run inference for all models

# MM single, cytosol
# NUTS with default init
XLA_FLAGS="--xla_force_host_platform_device_count=1" python inference_pymc.py -model MM_single -compartment cyto -free_params kOffAMP,kOffADP,kOffATP,kLKB1,kCaMKK -data_file ../../../Schmitt_et_al_2022_data/fig_2e_cyto.npz -model_info_file ../models/MM_single.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nwarmup 3000 -nsamples 3000 -nchains 4 -sampler NUTS -lower_mult 1e-2 -upper_mult 1e4 -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 --sample_posterior

# ADVI with mean field approximation
XLA_FLAGS="--xla_force_host_platform_device_count=1" python inference_pymc.py -model MM_single -compartment cyto -free_params kOffAMP,kOffADP,kOffATP,kLKB1,kCaMKK -data_file ../../../Schmitt_et_al_2022_data/fig_2e_cyto.npz -model_info_file ../models/MM_single.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nsamples 4000 -nchains 4 -sampler ADVI -lower_mult 1e-2 -upper_mult 1e4 -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 -n_advi_iter 60000 --sample_posterior

# MM single, lysosome
# NUTS with default init
XLA_FLAGS="--xla_force_host_platform_device_count=1" python inference_pymc.py -model MM_single -compartment lyso -free_params kOffAMP,kOffADP,kOffATP,kLKB1,kCaMKK -data_file ../../../Schmitt_et_al_2022_data/fig_2b_lyso.npz -model_info_file ../models/MM_single.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nwarmup 3000 -nsamples 3000 -nchains 4 -sampler NUTS -lower_mult 1e-2 -upper_mult 1e4 -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 --sample_posterior

# ADVI with mean field approximation
XLA_FLAGS="--xla_force_host_platform_device_count=1" python inference_pymc.py -model MM_single -compartment lyso -free_params kOffAMP,kOffADP,kOffATP,kLKB1,kCaMKK -data_file ../../../Schmitt_et_al_2022_data/fig_2b_lyso.npz -model_info_file ../models/MM_single.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nsamples 4000 -nchains 4 -sampler ADVI -lower_mult 1e-2 -upper_mult 1e4 -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 -n_advi_iter 60000 --sample_posterior

# MM single, mitochondria
# NUTS with default init
XLA_FLAGS="--xla_force_host_platform_device_count=1" python inference_pymc.py -model MM_single -compartment mito -free_params kOffAMP,kOffADP,kOffATP,kLKB1,kCaMKK -data_file ../../../Schmitt_et_al_2022_data/fig_2c_mito.npz -model_info_file ../models/MM_single.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nwarmup 3000 -nsamples 3000 -nchains 4 -sampler NUTS -lower_mult 1e-2 -upper_mult 1e4 -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 --sample_posterior

# ADVI with mean field approximation
XLA_FLAGS="--xla_force_host_platform_device_count=1" python inference_pymc.py -model MM_single -compartment mito -free_params kOffAMP,kOffADP,kOffATP,kLKB1,kCaMKK -data_file ../../../Schmitt_et_al_2022_data/fig_2c_mito.npz -model_info_file ../models/MM_single.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nsamples 4000 -nchains 4 -sampler ADVI -lower_mult 1e-2 -upper_mult 1e4 -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 -n_advi_iter 60000 --sample_posterior