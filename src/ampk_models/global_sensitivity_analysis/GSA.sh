# Coccimiglio
python GSA_sampling.py -model ampk_Coccimiglio -free_params k6r,k7r,k8r,k9r,k10r,k11r,Km12,Km13,Km14,Km15,Km16,Km17,Km18,Km19,Vmaxkinase,VmaxkinaseATP,VmaxkinaseADP,VmaxkinaseAMP,Vmaxppase,VmaxppaseATP,VmaxppaseADP,VmaxppaseAMP,Km_pAMPK,k_pAMPK,Km_AMP_pAMPK,k_AMP_pAMPK,Km_ADP_pAMPK,k_ADP_pAMPK,Km_ATP_pAMPK,k_ATP_pAMPK -model_info_file ../models/ampk_Coccimiglio.json -upper_mult 1e2 -lower_mult 1e-2 -metab_params_file ../models/metabolism_params_Coccimiglio.json -nsamples 2048 -gsa_method sobol -savedir ../../../results/GSA/ -tmax 5e5

# MA Single
python GSA_sampling.py -model MA_single -free_params kOffAMP,kOffADP,kOffATP,kOffCaMKK,kPhosCaMKK,kOffLKB1,kPhosLKB1,kOffPP,kDephosPP,kOffAMPK,kPhosAMPK,kOffPP1,kDephosPP1 -model_info_file ../models/MA_single.json -upper_mult 1e2 -lower_mult 1e-2 -metab_params_file ../models/metabolism_params_Coccimiglio.json -nsamples 2048 -gsa_method sobol -savedir ../../../results/GSA/ -tmax 5e5

# MA Single - no sensor
python GSA_sampling.py -model MA_single_noSensor -free_params kOffAMP,kOffADP,kOffATP,kOffCaMKK,kPhosCaMKK,kOffLKB1,kPhosLKB1,kOffPP,kDephosPP -model_info_file ../models/MA_single_noSensor.json -upper_mult 1e2 -lower_mult 1e-2 -metab_params_file ../models/metabolism_params_Coccimiglio.json -nsamples 2048 -gsa_method sobol -savedir ../../../results/GSA/ -tmax 5e5

# MM Single
python GSA_sampling.py -model MM_single -free_params kOffAMP,kOffADP,kOffATP,kCaMKK,KmCaMKK,kLKB1,KmLKB1,kPP,KmPP -model_info_file ../models/MM_single.json -upper_mult 1e2 -lower_mult 1e-2 -metab_params_file ../models/metabolism_params_Coccimiglio.json -nsamples 2048 -gsa_method sobol -savedir ../../../results/GSA/ -tmax 5e5

# MA Nonessential
python GSA_sampling.py -model MA_nonessential -free_params kOffAMP,kOffADP,kOffATP,kOffCaMKK,kPhosCaMKK,kOffLKB1,kPhosLKB1,kOffPP,kDephosPP,kOffAMPK,kPhosAMPK,kOffPP1,kDephosPP1,alphaLKB1,alphaPP,betaAMP -model_info_file ../models/MA_nonessential.json -upper_mult 1e2 -lower_mult 1e-2 -special_bounds ./special_bounds.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -nsamples 2048 -gsa_method sobol -savedir ../../../results/GSA/ -tmax 5e5

# MM Nonessential
python GSA_sampling.py -model MM_nonessential -free_params kOffAMP,kOffADP,kOffATP,kPhosCaMKK,KmCaMKK,kPhosLKB1,KmLKB1,kDephosPP,KmPP,alphaLKB1,alphaPP,betaAMP -model_info_file ../models/MM_nonessential.json -upper_mult 1e2 -lower_mult 1e-2 -special_bounds ./special_bounds.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -nsamples 2048 -gsa_method sobol -savedir ../../../results/GSA/ -tmax 5e5