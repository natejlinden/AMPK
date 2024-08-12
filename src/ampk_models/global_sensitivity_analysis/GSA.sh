# MA Double
python GSA_sampling.py -model MA_double_mech -free_params kOffAMP,kOffADP,kOffATP,kOffCaMKK,kPhosCaMKK,kOffLKB1,kPhosLKB1,kOffPP,kDephosPP,kOffAMPK,kPhosAMPK,kOffPP1,kDephosPP1 -model_info_file ../odes/MA_double.json -upper_mult 1e2 -lower_mult 1e-2 -metab_params_file ../odes/metabolism_params.json -nsamples 1024 -gsa_method sobol -savedir /tscc/lustre/ddn/scratch/nlinden/ampk

# MA Single
python GSA_sampling.py -model MA_single_mech -free_params kOffAMP,kOffADP,kOffATP,kOffCaMKK,kPhosCaMKK,kOffLKB1,kPhosLKB1,kOffPP,kDephosPP,kOffAMPK,kPhosAMPK,kOffPP1,kDephosPP1 -model_info_file ../odes/MA_single.json -upper_mult 1e2 -lower_mult 1e-2 -metab_params_file ../odes/metabolism_params.json -nsamples 1024 -gsa_method sobol -savedir /tscc/lustre/ddn/scratch/nlinden/ampk

# MM Double
python GSA_sampling.py -model MM_double_mech -free_params kOffAMP,kOffADP,kOffATP,VmaxCaMKK,KmCaMKK,VmaxLKB1,KmLKB1,VmaxPP,KmPP -model_info_file ../odes/MA_double.json -upper_mult 1e2 -lower_mult 1e-2 -metab_params_file ../odes/metabolism_params.json -nsamples 1024 -gsa_method sobol -savedir /tscc/lustre/ddn/scratch/nlinden/ampk

# MM Single
python GSA_sampling.py -model MM_single_mech -free_params kOffAMP,kOffADP,kOffATP,VmaxCaMKK,KmCaMKK,VmaxLKB1,KmLKB1,VmaxPP,KmPP -model_info_file ../odes/MA_single.json -upper_mult 1e2 -lower_mult 1e-2 -metab_params_file ../odes/metabolism_params.json -nsamples 1024 -gsa_method sobol -savedir /tscc/lustre/ddn/scratch/nlinden/ampk

# newmech MA single
python GSA_sampling.py -model newmech_MA_single -free_params kOffAMP,kOffADP,kOffATP,kOffCaMKK,kPhosCaMKK,kOffLKB1,kPhosLKB1,kOffPP,kDephosPP,kOffAMPK,kPhosAMPK,kOffPP1,kDephosPP1 -model_info_file ../odes/newmech_MA_single.json -upper_mult 1e2 -lower_mult 1e-2 -metab_params_file ../odes/metabolism_params.json -nsamples 1024 -gsa_method sobol -savedir /tscc/lustre/ddn/scratch/nlinden/ampk

