#! /bin/zsh
# script to run inference for all models

# MM nonessential, cytosol
# NUTS with default init
python inference_pymc.py -model MM_nonessential -compartment cyto -free_params kOffAMP,kOffADP,kOffATP,kCaMKK,KmCaMKK,kPP,KmPP,alphaPP,betaAMP -data_file ../../../Schmitt_et_al_2022_data/fig_2e_cyto.npz -model_info_file ../models/MM_nonessential.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nwarmup 300 -nsamples 1000 -nchains 4 -sampler Nutpie -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 --sample_posterior

# ADVI with mean field approximation
python inference_pymc.py -model MM_nonessential -compartment cyto -free_params kOffAMP,kOffADP,kOffATP,kCaMKK,KmCaMKK,kPP,KmPP,alphaPP,betaAMP -data_file ../../../Schmitt_et_al_2022_data/fig_2e_cyto.npz -model_info_file ../models/MM_nonessential.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nsamples 4000 -nchains 4 -sampler ADVI -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 -n_advi_iter 150000 --sample_posterior

# MM nonessential, lysosome
# NUTS with default init
python inference_pymc.py -model MM_nonessential -compartment lyso -free_params kOffAMP,kOffADP,kOffATP,kCaMKK,KmCaMKK,kPP,KmPP,alphaPP,betaAMP -data_file ../../../Schmitt_et_al_2022_data/fig_2b_lyso.npz -model_info_file ../models/MM_nonessential.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nwarmup 300 -nsamples 1000 -nchains 4 -sampler Nutpie -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 --sample_posterior

# ADVI with mean field approximation
python inference_pymc.py -model MM_nonessential -compartment lyso -free_params kOffAMP,kOffADP,kOffATP,kCaMKK,KmCaMKK,kPP,KmPP,alphaPP,betaAMP -data_file ../../../Schmitt_et_al_2022_data/fig_2b_lyso.npz -model_info_file ../models/MM_nonessential.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nsamples 4000 -nchains 4 -sampler ADVI -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 -n_advi_iter 150000 --sample_posterior

# MM nonessential, mitochondria
# NUTS with default init
python inference_pymc.py -model MM_nonessential -compartment mito -free_params kOffAMP,kOffADP,kOffATP,kCaMKK,KmCaMKK,kPP,KmPP,alphaPP,betaAMP -data_file ../../../Schmitt_et_al_2022_data/fig_2c_mito.npz -model_info_file ../models/MM_nonessential.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nwarmup 300 -nsamples 1000 -nchains 4 -sampler Nutpie -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 --sample_posterior

# ADVI with mean field approximation
python inference_pymc.py -model MM_nonessential -compartment mito -free_params kOffAMP,kOffADP,kOffATP,kCaMKK,KmCaMKK,kPP,KmPP,alphaPP,betaAMP -data_file ../../../Schmitt_et_al_2022_data/fig_2c_mito.npz -model_info_file ../models/MM_nonessential.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nsamples 4000 -nchains 4 -sampler ADVI -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 -n_advi_iter 150000 --sample_posterior