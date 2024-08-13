# MA Double
python GSA_sampling.py -model MA_double_mech -free_params kOffAMP,kOffADP,kOffATP,kOffCaMKK,kPhosCaMKK,kOffLKB1,kPhosLKB1,kOffPP,kDephosPP,kOffAMPK,kPhosAMPK,kOffPP1,kDephosPP1 -model_info_file ../odes/MA_double.json -upper_mult 1e2 -lower_mult 1e-2 -metab_params_file ../odes/metabolism_params.json -nsamples 2048 -gsa_method sobol -savedir /tscc/lustre/ddn/scratch/nlinden/ampk -tmax 5e5

# MA Single
python GSA_sampling.py -model MA_single_mech -free_params kOffAMP,kOffADP,kOffATP,kOffCaMKK,kPhosCaMKK,kOffLKB1,kPhosLKB1,kOffPP,kDephosPP,kOffAMPK,kPhosAMPK,kOffPP1,kDephosPP1 -model_info_file ../odes/MA_single.json -upper_mult 1e2 -lower_mult 1e-2 -metab_params_file ../odes/metabolism_params.json -nsamples 2048 -gsa_method sobol -savedir /tscc/lustre/ddn/scratch/nlinden/ampk -tmax 5e5

# MM Double
python GSA_sampling.py -model MM_double_mech -free_params kOffAMP,kOffADP,kOffATP,kPhosCaMKK,KmCaMKK,kPhosLKB1,KmLKB1,kDephosPP,KmPP -model_info_file ../odes/MA_double.json -upper_mult 1e2 -lower_mult 1e-2 -metab_params_file ../odes/metabolism_params.json -nsamples 2048 -gsa_method sobol -savedir /tscc/lustre/ddn/scratch/nlinden/ampk -tmax 5e5

# MM Single
python GSA_sampling.py -model MM_single_mech -free_params kOffAMP,kOffADP,kOffATP,kPhosCaMKK,KmCaMKK,kPhosLKB1,KmLKB1,kDephosPP,KmPP -model_info_file ../odes/MA_single.json -upper_mult 1e2 -lower_mult 1e-2 -metab_params_file ../odes/metabolism_params.json -nsamples 2048 -gsa_method sobol -savedir /tscc/lustre/ddn/scratch/nlinden/ampk -tmax 5e5

# newmech MA single
python GSA_sampling.py -model newmech_MA_single -free_params kOffAMP,kOffADP,kOffATP,kOffCaMKK,kPhosCaMKK,kOffLKB1,kPhosLKB1,kOffPP,kDephosPP,kOffAMPK,kPhosAMPK,kOffPP1,kDephosPP1 -model_info_file ../odes/newmech_MA_single.json -upper_mult 1e2 -lower_mult 1e-2 -metab_params_file ../odes/metabolism_params.json -nsamples 2048 -gsa_method sobol -savedir /tscc/lustre/ddn/scratch/nlinden/ampk -tmax 5e5

