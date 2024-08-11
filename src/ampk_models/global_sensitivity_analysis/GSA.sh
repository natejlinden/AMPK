# MA Double
python GSA_sampling.py -model MA_double_mech -free_params kOffAMP,kOffADP,kOffATP,kOffCaMKK,kPhosCaMKK,kOffLKB1,kPhosLKB1,kOffPP,kDephosPP,kOffAMPK,kPhosAMPK,kOffPP1,kDephosPP1 -model_info_file ../odes/MA_double.json -upper_mult 1e2 -lower_mult 1e-2 -metab_params_file ../odes/metabolism_params.json -nsamples 512 -gsa_method sobol -savedir /tscc/lustre/ddn/scratch/nlinden/ampk 
