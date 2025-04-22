#! /bin/zsh

# script to run global sensitivity analysis sampling for all models
# only sample locally identifiable parameters

# MA Single
python GSA_sampling.py -model MA_single -free_params kOffAMP,kOffADP,kOffATP,kOffCaMKK,kPhosCaMKK,kOffLKB1,kPhosLKB1,kOffPP,kDephosPP,kOffAMPK,kPhosAMPK,kOffPP1,kDephosPP1 -model_info_file ../models/MA_single.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -nsamples 256 -savedir ../../../results/GSA/MA_single/ -tmax 1800 -ca_stress 0.25

# MM Single
# include nonindentifiable parameters
python GSA_sampling.py -model MM_single -free_params kOffAMP,kOffADP,kOffATP,KmCaMKK,kLKB1,KmLKB1,kPP,KmPP -model_info_file ../models/MM_single.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -nsamples 256 -savedir ../../../results/GSA/MM_single/ -tmax 1800 -ca_stress 0.25

# MA Nonessential
python GSA_sampling.py -model MA_nonessential -free_params kOffAMP,kOffADP,kOffATP,kOffCaMKK,kPhosCaMKK,kOffLKB1,kPhosLKB1,kOffPP,kDephosPP,kOffAMPK,kPhosAMPK,kOffPP1,kDephosPP1,alphaLKB1,alphaPP,betaAMP -model_info_file ../models/MA_nonessential.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -nsamples 256 -savedir ../../../results/GSA/MA_nonessential/ -tmax 1800 -ca_stress 0.25

# MM Nonessential
python GSA_sampling.py -model MM_nonessential -free_params kOffAMP,kOffADP,kOffATP,KmCaMKK,kLKB1,KmLKB1,alphaLKB1,kPP,KmPP,alphaPP,betaAMP -model_info_file ../models/MM_nonessential.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -nsamples 256 -savedir ../../../results/GSA/MM_nonessential/ -tmax 1800 -ca_stress 0.25

# MA nonessential all
python GSA_sampling.py -model MA_nonessential_all -free_params kOffAMP,kOffADP,kOffATP,kOffCaMKK,kPhosCaMKK,kOffLKB1,kPhosLKB1,kOffPP,kDephosPP,kOffAMPK,kPhosAMPK,kOffPP1,kDephosPP1,alphaPP,betaAMP,betaLKB1,betaCaMKK -model_info_file ../models/MA_nonessential_all.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -nsamples 256 -savedir ../../../results/GSA/MA_nonessential_all/ -tmax 1800 -ca_stress 0.25

# MM nonessential all
python GSA_sampling.py -model MM_nonessential_all -free_params kOffAMP,kOffADP,kOffATP,KmCaMKK,kLKB1,KmLKB1,kPP,KmPP,betaAMPK,betaLKB1,betaCaMKK,alphaPP -model_info_file ../models/MM_nonessential_all.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -nsamples 256 -savedir ../../../results/GSA/MM_nonessential_all/ -tmax 1800 -ca_stress 0.25