#! /bin/zsh
# script to run inference for all models

# MM single, cytosol
# NUTS with default init
# python inference_pymc.py -model MM_simple1 -compartment cyto -free_params kOffAMP,kOffADP,kOffATP,kCaMKK,KmCaMKK,kLKB1,KmLKB1,kPP,KmPP,betaAMPK,betaLKB1,betaCaMKK,alphaPP -data_file ../../../Schmitt_et_al_2022_data/fig_2e_cyto.npz -model_info_file ../models/MM_simple1.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nwarmup 300 -nsamples 1000 -nchains 4 -sampler Nutpie -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 --sample_posterior

python inference_pymc.py -model MM_simple1 -compartment cyto -free_params kOffAMP,kOffADP,kOffATP,kCaMKK,KmCaMKK,kLKB1,KmLKB1,kPP,KmPP,betaAMPK,betaLKB1,betaCaMKK,alphaPP -data_file ../../../Schmitt_et_al_2022_data/fig_2e_cyto.npz -model_info_file ../models/MM_simple1.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nsamples 1000 -nchains 1 -sampler Pathfinder -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 --sample_posterior --compute_llike -n_checkpoints 5000

# MM single, lysosome
# NUTS with default init
# python inference_pymc.py -model MM_simple1 -compartment lyso -free_params kOffAMP,kOffADP,kOffATP,kCaMKK,KmCaMKK,kLKB1,KmLKB1,kPP,KmPP,betaAMPK,betaLKB1,betaCaMKK,alphaPP -data_file ../../../Schmitt_et_al_2022_data/fig_2b_lyso.npz -model_info_file ../models/MM_simple1.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nwarmup 300 -nsamples 1000 -nchains 4 -sampler Nutpie -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 --sample_posterior

python inference_pymc.py -model MM_simple1 -compartment lyso -free_params kOffAMP,kOffADP,kOffATP,kCaMKK,KmCaMKK,kLKB1,KmLKB1,kPP,KmPP,betaAMPK,betaLKB1,betaCaMKK,alphaPP -data_file ../../../Schmitt_et_al_2022_data/fig_2b_lyso.npz -model_info_file ../models/MM_simple1.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nsamples 1000 -nchains 1 -sampler Pathfinder -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 --sample_posterior --compute_llike -n_checkpoints 5000

# MM single, mitochondria
# NUTS with default init
# python inference_pymc.py -model MM_simple1 -compartment mito -free_params kOffAMP,kOffADP,kOffATP,kCaMKK,KmCaMKK,kLKB1,KmLKB1,kPP,KmPP,betaAMPK,betaLKB1,betaCaMKK,alphaPP -data_file ../../../Schmitt_et_al_2022_data/fig_2c_mito.npz -model_info_file ../models/MM_simple1.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nwarmup 300 -nsamples 1000 -nchains 4 -sampler Nutpie -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 --sample_posterior

python inference_pymc.py -model MM_simple1 -compartment mito -free_params kOffAMP,kOffADP,kOffATP,kCaMKK,KmCaMKK,kLKB1,KmLKB1,kPP,KmPP,betaAMPK,betaLKB1,betaCaMKK,alphaPP -data_file ../../../Schmitt_et_al_2022_data/fig_2c_mito.npz -model_info_file ../models/MM_simple1.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nsamples 1000 -nchains 1 -sampler Pathfinder -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 --sample_posterior --compute_llike -n_checkpoints 5000