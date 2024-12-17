#! /bin/zsh
# script to call inference for the model calibration

# MA single, cytosol
python inference_run.py -model MA_single -compartment cyto -free_params kOffAMP -data_file ../../../Schmitt_et_al_2022_data/fig_2e_cyto.npz -model_info_file ../models/MA_single.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ./test/ -nwarmup 3000 -nsamples 10000 -nchains 14 -sampler AIES

# MA single, lysosome
python inference_run.py -model MA_single -compartment lyso -free_params kOffAMP -data_file ../../../Schmitt_et_al_2022_data/fig_2b_lyso.npz -model_info_file ../models/MA_single.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ./test/ -nwarmup 3000 -nsamples 1000 -nchains 14 -sampler AIES

# MA single, mitochondria
python inference_run.py -model MA_single -compartment mito -free_params kOffAMP -data_file ../../../Schmitt_et_al_2022_data/fig_2c_mito.npz -model_info_file ../models/MA_single.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ./test/ -nwarmup 3000 -nsamples 1000 -nchains 14 -sampler AIES

# MM single, cytosol
python inference_run.py -model MM_single -compartment cyto -free_params kOffAMP -data_file ../../../Schmitt_et_al_2022_data/fig_2e_cyto.npz -model_info_file ../models/MM_single.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ./test/ -nwarmup 3000 -nsamples 1000 -nchains 14 -sampler AIES

# MM single, lysosome
python inference_run.py -model MM_single -compartment lyso -free_params kOffAMP -data_file ../../../Schmitt_et_al_2022_data/fig_2b_lyso.npz -model_info_file ../models/MM_single.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ./test/ -nwarmup 3000 -nsamples 1000 -nchains 14 -sampler AIES

# MM single, mitochondria
python inference_run.py -model MA_single -compartment mito -free_params kOffAMP -data_file ../../../Schmitt_et_al_2022_data/fig_2c_mito.npz -model_info_file ../models/MA_single.json -metab_params_file ../models/metabolism_params_Coccimiglio.json -savedir ./test/ -nwarmup 3000 -nsamples 1000 -nchains 14 -sampler AIES