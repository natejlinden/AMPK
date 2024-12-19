#! /bin/zsh
# script to run inference for all models

# AMPK Coccimiglio, cytosol
# NUTS with default init
XLA_FLAGS="--xla_force_host_platform_device_count=1" python inference_pymc.py -model ampk_Coccimiglio -compartment cyto -free_params Km_ADP_pAMPK,k_ADP_pAMPK,Km16,Km_AMP_pAMPK,VmaxkinaseADP,k10r,Km_pAMPK,Km13,Km_ATP_pAMPK,k_AMP_pAMPK,Km17,Vmaxppase,k_ATP_pAMPK,Km15,k11r,k9r,Km14,VmaxppaseATP,Km18,k7r,Km12,k_pAMPK -data_file ../../../Schmitt_et_al_2022_data/fig_2e_cyto.npz -model_info_file ../models/ampk_Coccimiglio.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nwarmup 1000 -nsamples 2000 -nchains 4 -sampler NUTS -lower_mult 1e-2 -upper_mult 1e2 -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 --sample_posterior

# ADVI with mean field approximation
XLA_FLAGS="--xla_force_host_platform_device_count=1" python inference_pymc.py -model ampk_Coccimiglio -compartment cyto -free_params Km_ADP_pAMPK,k_ADP_pAMPK,Km16,Km_AMP_pAMPK,VmaxkinaseADP,k10r,Km_pAMPK,Km13,Km_ATP_pAMPK,k_AMP_pAMPK,Km17,Vmaxppase,k_ATP_pAMPK,Km15,k11r,k9r,Km14,VmaxppaseATP,Km18,k7r,Km12,k_pAMPK -data_file ../../../Schmitt_et_al_2022_data/fig_2e_cyto.npz -model_info_file ../models/ampk_Coccimiglio.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nsamples 4000 -nchains 4 -sampler ADVI -lower_mult 1e-2 -upper_mult 1e2 -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 -n_advi_iter 5000 --sample_posterior

# AMPK Coccimiglio, lysosome
# NUTS with default init
XLA_FLAGS="--xla_force_host_platform_device_count=1" python inference_pymc.py -model ampk_Coccimiglio -compartment lyso -free_params Km_ADP_pAMPK,k_ADP_pAMPK,Km16,Km_AMP_pAMPK,VmaxkinaseADP,k10r,Km_pAMPK,Km13,Km_ATP_pAMPK,k_AMP_pAMPK,Km17,Vmaxppase,k_ATP_pAMPK,Km15,k11r,k9r,Km14,VmaxppaseATP,Km18,k7r,Km12,k_pAMPK -data_file ../../../Schmitt_et_al_2022_data/fig_2b_lyso.npz -model_info_file ../models/ampk_Coccimiglio.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nwarmup 1000 -nsamples 2000 -nchains 4 -sampler NUTS -lower_mult 1e-2 -upper_mult 1e2 -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 --sample_posterior

# ADVI with mean field approximation
XLA_FLAGS="--xla_force_host_platform_device_count=1" python inference_pymc.py -model ampk_Coccimiglio -compartment lyso -free_params Km_ADP_pAMPK,k_ADP_pAMPK,Km16,Km_AMP_pAMPK,VmaxkinaseADP,k10r,Km_pAMPK,Km13,Km_ATP_pAMPK,k_AMP_pAMPK,Km17,Vmaxppase,k_ATP_pAMPK,Km15,k11r,k9r,Km14,VmaxppaseATP,Km18,k7r,Km12,k_pAMPK -data_file ../../../Schmitt_et_al_2022_data/fig_2b_lyso.npz -model_info_file ../models/ampk_Coccimiglio.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nsamples 4000 -nchains 4 -sampler ADVI -lower_mult 1e-2 -upper_mult 1e2 -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 -n_advi_iter 5000 --sample_posterior

# AMPK Coccimiglio, mitochondria
# NUTS with default init
XLA_FLAGS="--xla_force_host_platform_device_count=1" python inference_pymc.py -model ampk_Coccimiglio -compartment mito -free_params Km_ADP_pAMPK,k_ADP_pAMPK,Km16,Km_AMP_pAMPK,VmaxkinaseADP,k10r,Km_pAMPK,Km13,Km_ATP_pAMPK,k_AMP_pAMPK,Km17,Vmaxppase,k_ATP_pAMPK,Km15,k11r,k9r,Km14,VmaxppaseATP,Km18,k7r,Km12,k_pAMPK -data_file ../../../Schmitt_et_al_2022_data/fig_2c_mito.npz -model_info_file ../models/ampk_Coccimiglio.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nwarmup 1000 -nsamples 2000 -nchains 4 -sampler NUTS -lower_mult 1e-2 -upper_mult 1e2 -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 --sample_posterior

# ADVI with mean field approximation
XLA_FLAGS="--xla_force_host_platform_device_count=1" python inference_pymc.py -model ampk_Coccimiglio -compartment mito -free_params Km_ADP_pAMPK,k_ADP_pAMPK,Km16,Km_AMP_pAMPK,VmaxkinaseADP,k10r,Km_pAMPK,Km13,Km_ATP_pAMPK,k_AMP_pAMPK,Km17,Vmaxppase,k_ATP_pAMPK,Km15,k11r,k9r,Km14,VmaxppaseATP,Km18,k7r,Km12,k_pAMPK -data_file ../../../Schmitt_et_al_2022_data/fig_2c_mito.npz -model_info_file ../models/ampk_Coccimiglio.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nsamples 4000 -nchains 4 -sampler ADVI -lower_mult 1e-2 -upper_mult 1e2 -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 -n_advi_iter 5000 --sample_posterior


########################################################################################

# MA single, cytosol
# NUTS with default init
XLA_FLAGS="--xla_force_host_platform_device_count=1" python inference_pymc.py -model MA_single -compartment cyto -free_params kOffAMP,kOffADP,kOffATP,kOffAMPK,kPhosAMPK,kOffPP1,kDephosPP1 -data_file ../../../Schmitt_et_al_2022_data/fig_2e_cyto.npz -model_info_file ../models/MA_single.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nwarmup 1000 -nsamples 2000 -nchains 4 -sampler NUTS -lower_mult 1e-2 -upper_mult 1e2 -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 --sample_posterior

# ADVI with mean field approximation
XLA_FLAGS="--xla_force_host_platform_device_count=1" python inference_pymc.py -model MA_single -compartment cyto -free_params kOffAMP,kOffADP,kOffATP,kOffAMPK,kPhosAMPK,kOffPP1,kDephosPP1 -data_file ../../../Schmitt_et_al_2022_data/fig_2e_cyto.npz -model_info_file ../models/MA_single.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nsamples 4000 -nchains 4 -sampler ADVI -lower_mult 1e-2 -upper_mult 1e2 -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 -n_advi_iter 5000 --sample_posterior

# MA single, lysosome
# NUTS with default init
XLA_FLAGS="--xla_force_host_platform_device_count=1" python inference_pymc.py -model MA_single -compartment lyso -free_params kOffAMP,kOffADP,kOffATP,kOffAMPK,kPhosAMPK,kOffPP1,kDephosPP1 -data_file ../../../Schmitt_et_al_2022_data/fig_2b_lyso.npz -model_info_file ../models/MA_single.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nwarmup 1000 -nsamples 2000 -nchains 4 -sampler NUTS -lower_mult 1e-2 -upper_mult 1e2 -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 --sample_posterior

# ADVI with mean field approximation
XLA_FLAGS="--xla_force_host_platform_device_count=1" python inference_pymc.py -model MA_single -compartment lyso -free_params kOffAMP,kOffADP,kOffATP,kOffAMPK,kPhosAMPK,kOffPP1,kDephosPP1 -data_file ../../../Schmitt_et_al_2022_data/fig_2b_lyso.npz -model_info_file ../models/MA_single.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nsamples 4000 -nchains 4 -sampler ADVI -lower_mult 1e-2 -upper_mult 1e2 -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 -n_advi_iter 5000 --sample_posterior

# MA single, mitochondria
# NUTS with default init
XLA_FLAGS="--xla_force_host_platform_device_count=1" python inference_pymc.py -model MA_single -compartment mito -free_params kOffAMP,kOffADP,kOffATP,kOffAMPK,kPhosAMPK,kOffPP1,kDephosPP1 -data_file ../../../Schmitt_et_al_2022_data/fig_2c_mito.npz -model_info_file ../models/MA_single.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nwarmup 1000 -nsamples 2000 -nchains 4 -sampler NUTS -lower_mult 1e-2 -upper_mult 1e2 -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 --sample_posterior

# ADVI with mean field approximation
XLA_FLAGS="--xla_force_host_platform_device_count=1" python inference_pymc.py -model MA_single -compartment mito -free_params kOffAMP,kOffADP,kOffATP,kOffAMPK,kPhosAMPK,kOffPP1,kDephosPP1 -data_file ../../../Schmitt_et_al_2022_data/fig_2c_mito.npz -model_info_file ../models/MA_single.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nsamples 4000 -nchains 4 -sampler ADVI -lower_mult 1e-2 -upper_mult 1e2 -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 -n_advi_iter 5000 --sample_posterior

########################################################################################

# MA nonessential, cytosol
# NUTS with default init
XLA_FLAGS="--xla_force_host_platform_device_count=1" python inference_pymc.py -model MA_nonessential -compartment cyto -free_params kOffAMP,kOffADP,kOffATP,kOffAMPK,kPhosAMPK,kOffPP1,kDephosPP1,betaAMP -data_file ../../../Schmitt_et_al_2022_data/fig_2e_cyto.npz -model_info_file ../models/MA_nonessential.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nwarmup 1000 -nsamples 2000 -nchains 4 -sampler NUTS -lower_mult 1e-2 -upper_mult 1e2 -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 --sample_posterior

# ADVI with mean field approximation
XLA_FLAGS="--xla_force_host_platform_device_count=1" python inference_pymc.py -model MA_nonessential -compartment cyto -free_params kOffAMP,kOffADP,kOffATP,kOffAMPK,kPhosAMPK,kOffPP1,kDephosPP1,betaAMP -data_file ../../../Schmitt_et_al_2022_data/fig_2e_cyto.npz -model_info_file ../models/MA_nonessential.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nsamples 4000 -nchains 4 -sampler ADVI -lower_mult 1e-2 -upper_mult 1e2 -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 -n_advi_iter 5000 --sample_posterior

# MA nonessential, lysosome
# NUTS with default init
XLA_FLAGS="--xla_force_host_platform_device_count=1" python inference_pymc.py -model MA_nonessential -compartment lyso -free_params kOffAMP,kOffADP,kOffATP,kOffAMPK,kPhosAMPK,kOffPP1,kDephosPP1,betaAMP -data_file ../../../Schmitt_et_al_2022_data/fig_2b_lyso.npz -model_info_file ../models/MA_nonessential.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nwarmup 1000 -nsamples 2000 -nchains 4 -sampler NUTS -lower_mult 1e-2 -upper_mult 1e2 -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 --sample_posterior

# ADVI with mean field approximation
XLA_FLAGS="--xla_force_host_platform_device_count=1" python inference_pymc.py -model MA_nonessential -compartment lyso -free_params kOffAMP,kOffADP,kOffATP,kOffAMPK,kPhosAMPK,kOffPP1,kDephosPP1,betaAMP -data_file ../../../Schmitt_et_al_2022_data/fig_2b_lyso.npz -model_info_file ../models/MA_nonessential.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nsamples 4000 -nchains 4 -sampler ADVI -lower_mult 1e-2 -upper_mult 1e2 -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 -n_advi_iter 5000 --sample_posterior

# MA nonessential, mitochondria
# NUTS with default init
XLA_FLAGS="--xla_force_host_platform_device_count=1" python inference_pymc.py -model MA_nonessential -compartment mito -free_params kOffAMP,kOffADP,kOffATP,kOffAMPK,kPhosAMPK,kOffPP1,kDephosPP1,betaAMP -data_file ../../../Schmitt_et_al_2022_data/fig_2c_mito.npz -model_info_file ../models/MA_nonessential.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nwarmup 1000 -nsamples 2000 -nchains 4 -sampler NUTS -lower_mult 1e-2 -upper_mult 1e2 -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 --sample_posterior

# ADVI with mean field approximation
XLA_FLAGS="--xla_force_host_platform_device_count=1" python inference_pymc.py -model MA_nonessential -compartment mito -free_params kOffAMP,kOffADP,kOffATP,kOffAMPK,kPhosAMPK,kOffPP1,kDephosPP1,betaAMP -data_file ../../../Schmitt_et_al_2022_data/fig_2c_mito.npz -model_info_file ../models/MA_nonessential.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nsamples 4000 -nchains 4 -sampler ADVI -lower_mult 1e-2 -upper_mult 1e2 -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 -n_advi_iter 5000 --sample_posterior

########################################################################################

# MM single, cytosol
# NUTS with default init
XLA_FLAGS="--xla_force_host_platform_device_count=1" python inference_pymc.py -model MM_single -compartment cyto -free_params kOffAMP,kOffADP,kOffATP,kLKB1,kCaMKK -data_file ../../../Schmitt_et_al_2022_data/fig_2e_cyto.npz -model_info_file ../models/MM_single.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nwarmup 1000 -nsamples 2000 -nchains 4 -sampler NUTS -lower_mult 1e-2 -upper_mult 1e2 -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 --sample_posterior

# ADVI with mean field approximation
XLA_FLAGS="--xla_force_host_platform_device_count=1" python inference_pymc.py -model MM_single -compartment cyto -free_params kOffAMP,kOffADP,kOffATP,kLKB1,kCaMKK -data_file ../../../Schmitt_et_al_2022_data/fig_2e_cyto.npz -model_info_file ../models/MM_single.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nsamples 4000 -nchains 4 -sampler ADVI -lower_mult 1e-2 -upper_mult 1e2 -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 -n_advi_iter 5000 --sample_posterior

# MM single, lysosome
# NUTS with default init
XLA_FLAGS="--xla_force_host_platform_device_count=1" python inference_pymc.py -model MM_single -compartment lyso -free_params kOffAMP,kOffADP,kOffATP,kLKB1,kCaMKK -data_file ../../../Schmitt_et_al_2022_data/fig_2b_lyso.npz -model_info_file ../models/MM_single.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nwarmup 1000 -nsamples 2000 -nchains 4 -sampler NUTS -lower_mult 1e-2 -upper_mult 1e2 -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 --sample_posterior

# ADVI with mean field approximation
XLA_FLAGS="--xla_force_host_platform_device_count=1" python inference_pymc.py -model MM_single -compartment lyso -free_params kOffAMP,kOffADP,kOffATP,kLKB1,kCaMKK -data_file ../../../Schmitt_et_al_2022_data/fig_2b_lyso.npz -model_info_file ../models/MM_single.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nsamples 4000 -nchains 4 -sampler ADVI -lower_mult 1e-2 -upper_mult 1e2 -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 -n_advi_iter 5000 --sample_posterior

# MM single, mitochondria
# NUTS with default init
XLA_FLAGS="--xla_force_host_platform_device_count=1" python inference_pymc.py -model MM_single -compartment mito -free_params kOffAMP,kOffADP,kOffATP,kLKB1,kCaMKK -data_file ../../../Schmitt_et_al_2022_data/fig_2c_mito.npz -model_info_file ../models/MM_single.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nwarmup 1000 -nsamples 2000 -nchains 4 -sampler NUTS -lower_mult 1e-2 -upper_mult 1e2 -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 --sample_posterior

# ADVI with mean field approximation
XLA_FLAGS="--xla_force_host_platform_device_count=1" python inference_pymc.py -model MM_single -compartment mito -free_params kOffAMP,kOffADP,kOffATP,kLKB1,kCaMKK -data_file ../../../Schmitt_et_al_2022_data/fig_2c_mito.npz -model_info_file ../models/MM_single.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nsamples 4000 -nchains 4 -sampler ADVI -lower_mult 1e-2 -upper_mult 1e2 -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 -n_advi_iter 5000 --sample_posterior

########################################################################################

# MM nonessential, cytosol
# NUTS with default init
XLA_FLAGS="--xla_force_host_platform_device_count=1" python inference_pymc.py -model MM_nonessential -compartment cyto -free_params kOffAMP,kOffADP,kOffATP,betaAMP -data_file ../../../Schmitt_et_al_2022_data/fig_2e_cyto.npz -model_info_file ../models/MM_nonessential.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nwarmup 1000 -nsamples 2000 -nchains 4 -sampler NUTS -lower_mult 1e-2 -upper_mult 1e2 -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 --sample_posterior

# ADVI with mean field approximation
XLA_FLAGS="--xla_force_host_platform_device_count=1" python inference_pymc.py -model MM_nonessential -compartment cyto -free_params kOffAMP,kOffADP,kOffATP,betaAMP -data_file ../../../Schmitt_et_al_2022_data/fig_2e_cyto.npz -model_info_file ../models/MM_nonessential.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nsamples 4000 -nchains 4 -sampler ADVI -lower_mult 1e-2 -upper_mult 1e2 -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 -n_advi_iter 5000 --sample_posterior

# MM nonessential, lysosome
# NUTS with default init
XLA_FLAGS="--xla_force_host_platform_device_count=1" python inference_pymc.py -model MM_nonessential -compartment lyso -free_params kOffAMP,kOffADP,kOffATP,betaAMP -data_file ../../../Schmitt_et_al_2022_data/fig_2b_lyso.npz -model_info_file ../models/MM_nonessential.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nwarmup 1000 -nsamples 2000 -nchains 4 -sampler NUTS -lower_mult 1e-2 -upper_mult 1e2 -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 --sample_posterior

# ADVI with mean field approximation
XLA_FLAGS="--xla_force_host_platform_device_count=1" python inference_pymc.py -model MM_nonessential -compartment lyso -free_params kOffAMP,kOffADP,kOffATP,betaAMP -data_file ../../../Schmitt_et_al_2022_data/fig_2b_lyso.npz -model_info_file ../models/MM_nonessential.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nsamples 4000 -nchains 4 -sampler ADVI -lower_mult 1e-2 -upper_mult 1e2 -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 -n_advi_iter 5000 --sample_posterior

# MM nonessential, mitochondria
# NUTS with default init
XLA_FLAGS="--xla_force_host_platform_device_count=1" python inference_pymc.py -model MM_nonessential -compartment mito -free_params kOffAMP,kOffADP,kOffATP,betaAMP -data_file ../../../Schmitt_et_al_2022_data/fig_2c_mito.npz -model_info_file ../models/MM_nonessential.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nwarmup 1000 -nsamples 2000 -nchains 4 -sampler NUTS -lower_mult 1e-2 -upper_mult 1e2 -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 --sample_posterior

# ADVI with mean field approximation
XLA_FLAGS="--xla_force_host_platform_device_count=1" python inference_pymc.py -model MM_nonessential -compartment mito -free_params kOffAMP,kOffADP,kOffATP,betaAMP -data_file ../../../Schmitt_et_al_2022_data/fig_2c_mito.npz -model_info_file ../models/MM_nonessential.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nsamples 4000 -nchains 4 -sampler ADVI -lower_mult 1e-2 -upper_mult 1e2 -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 -n_advi_iter 5000 --sample_posterior