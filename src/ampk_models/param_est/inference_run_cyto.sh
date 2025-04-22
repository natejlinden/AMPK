# Inference to CYTO WT data only

# MA single
python inference_pymc.py -model MA_single -compartment cyto -free_params kOffAMP,kOffADP,kOffATP,kOffCaMKK,kPhosCaMKK,kOffLKB1,kPhosLKB1,kOffAMPK,kPhosAMPK -data_file ../../../Schmitt_et_al_2022_data/fig_2e_cyto.npz -model_info_file ../models/MA_single.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nsamples 1000 -nchains 1 -sampler Pathfinder -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 --sample_posterior --compute_llike -ca_stress 0.25

# # MM single
python inference_pymc.py -model MM_single -compartment cyto -free_params kOffAMP,kOffADP,kOffATP,KmCaMKK,kLKB1,KmLKB1,kPP -data_file ../../../Schmitt_et_al_2022_data/fig_2e_cyto.npz -model_info_file ../models/MM_single.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nsamples 1000 -nchains 1 -sampler Pathfinder -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 --sample_posterior --compute_llike -ca_stress 0.25

# MA nonessential
python inference_pymc.py -model MA_nonessential -compartment cyto -free_params kOffAMP,kOffADP,kOffATP,kOffCaMKK,kPhosCaMKK,kOffLKB1,kPhosLKB1,kOffPP,kDephosPP,kOffAMPK,kPhosAMPK,kOffPP1,kDephosPP1,alphaLKB1,betaAMP -data_file ../../../Schmitt_et_al_2022_data/fig_2e_cyto.npz -model_info_file ../models/MA_nonessential.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nsamples 1000 -nchains 1 -sampler Pathfinder -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 --sample_posterior

# MM nonessential
python inference_pymc.py -model MM_nonessential -compartment cyto -free_params kOffAMP,kOffADP,kOffATP,KmCaMKK,kLKB1,KmLKB1,alphaLKB1,kPP,KmPP,alphaPP,betaAMP -data_file ../../../Schmitt_et_al_2022_data/fig_2e_cyto.npz -model_info_file ../models/MM_nonessential.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nsamples 1000 -nchains 1 -sampler Pathfinder -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 --sample_posterior --compute_llike -ca_stress 0.25

# # MA nonessential all
python inference_pymc.py -model MA_nonessential_all -compartment cyto -free_params kOffAMP,kOffADP,kOffATP,kPhosCaMKK,kOffLKB1,kPhosLKB1,kOffPP,kDephosPP,kOffAMPK,kPhosAMPK,kOffPP1,kDephosPP1,alphaPP,betaAMP,betaLKB1 -data_file ../../../Schmitt_et_al_2022_data/fig_2e_cyto.npz -model_info_file ../models/MA_nonessential_all.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nsamples 1000 -nchains 1 -sampler Pathfinder -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 --sample_posterior --compute_llike -ca_stress 0.25

# MM nonessential all
python inference_pymc.py -model MM_nonessential_all -compartment cyto -free_params kOffAMP,kOffADP,kOffATP,KmCaMKK,kLKB1,KmLKB1,kPP,KmPP,betaAMPK,betaLKB1,betaCaMKK,alphaPP -data_file ../../../Schmitt_et_al_2022_data/fig_2e_cyto.npz -model_info_file ../models/MM_nonessential_all.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ../../../results/param_est/ -nsamples 1000 -nchains 1 -sampler Pathfinder -pcoeff 0.3 -icoeff 0.4 -rtol 1e-6 -atol 1e-6 --sample_posterior --compute_llike -ca_stress 0.25