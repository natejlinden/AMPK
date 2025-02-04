# Coccimiglio
/tscc/nfs/home/nlinden/miniforge3/envs/pymc/bin/python GSA_sampling.py -model ampk_Coccimiglio -free_params k6r,k7r,k8r,k9r,k10r,k11r,Km12,Km13,Km14,Km15,Km16,Km17,Km18,Km19,Vmaxkinase,VmaxkinaseATP,VmaxkinaseADP,VmaxkinaseAMP,Vmaxppase,VmaxppaseATP,VmaxppaseADP,VmaxppaseAMP,Km_pAMPK,k_pAMPK,Km_AMP_pAMPK,k_AMP_pAMPK,Km_ADP_pAMPK,k_ADP_pAMPK,Km_ATP_pAMPK,k_ATP_pAMPK -model_info_file ../models/ampk_Coccimiglio.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -nsamples 4096 -gsa_method sobol -savedir /tscc/lustre/ddn/scratch/nlinden/ampk/MA_single/ -tmax 1800

# MA Single
/tscc/nfs/home/nlinden/miniforge3/envs/pymc/bin/python GSA_sampling.py -model MA_single -free_params kOffAMP,kOffADP,kOffATP,kOffCaMKK,kPhosCaMKK,kOffLKB1,kPhosLKB1,kOffPP,kDephosPP,kOffAMPK,kPhosAMPK,kOffPP1,kDephosPP1 -model_info_file ../models/MA_single.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -nsamples 4096 -gsa_method sobol -savedir /tscc/lustre/ddn/scratch/nlinden/ampk/MA_single/ -tmax 1800

# MA Single -- No Sensor
/tscc/nfs/home/nlinden/miniforge3/envs/pymc/bin/python GSA_sampling.py -model MA_single -free_params kOffAMP,kOffADP,kOffATP,kOffCaMKK,kPhosCaMKK,kOffLKB1,kPhosLKB1,kOffPP,kDephosPP,kOffAMPK,kPhosAMPK,kOffPP1,kDephosPP1 -model_info_file ../models/MA_single_noSensor.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -nsamples 4096 -gsa_method sobol -savedir /tscc/lustre/ddn/scratch/nlinden/ampk/MA_single/no_Sensor_ -tmax 1800

# MM Single
/tscc/nfs/home/nlinden/miniforge3/envs/pymc/bin/python GSA_sampling.py -model MM_single -free_params kOffAMP,kOffADP,kOffATP,kCaMKK,KmCaMKK,kLKB1,KmLKB1,kPP,KmPP -model_info_file ../models/MM_single.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -nsamples 4096 -gsa_method sobol -savedir /tscc/lustre/ddn/scratch/nlinden/ampk/MA_single/ -tmax 1800

# MM Single -- No Sensor
/tscc/nfs/home/nlinden/miniforge3/envs/pymc/bin/python GSA_sampling.py -model MM_single -free_params kOffAMP,kOffADP,kOffATP,kCaMKK,KmCaMKK,kLKB1,KmLKB1,kPP,KmPP -model_info_file ../models/MM_single_noSensor.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -nsamples 4096 -gsa_method sobol -savedir /tscc/lustre/ddn/scratch/nlinden/ampk/MA_single/no_Sensor_ -tmax 1800

# MA Nonessential
/tscc/nfs/home/nlinden/miniforge3/envs/pymc/bin/python GSA_sampling.py -model MA_nonessential -free_params kOffAMP,kOffADP,kOffATP,kOffCaMKK,kPhosCaMKK,kOffLKB1,kPhosLKB1,kOffPP,kDephosPP,kOffAMPK,kPhosAMPK,kOffPP1,kDephosPP1,alphaLKB1,alphaPP,betaAMP -model_info_file ../models/MA_nonessential.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -nsamples 4096 -gsa_method sobol -savedir /tscc/lustre/ddn/scratch/nlinden/ampk/MA_single/ -tmax 1800

# MM Nonessential
/tscc/nfs/home/nlinden/miniforge3/envs/pymc/bin/python GSA_sampling.py -model MM_nonessential -free_params kOffAMP,kOffADP,kOffATP,kPhosCaMKK,KmCaMKK,kPhosLKB1,KmLKB1,kDephosPP,KmPP,alphaLKB1,alphaPP,betaAMP -model_info_file ../models/MM_nonessential.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -nsamples 4096 -gsa_method sobol -savedir /tscc/lustre/ddn/scratch/nlinden/ampk/MA_single/ -tmax 1800