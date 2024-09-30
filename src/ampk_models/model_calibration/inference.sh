#! /bin/zsh
# script to call inference for the model calibration

# single binding, mass action, cytosol
python inference_run.py -model MA_single_mech -free_params kOffAMP -data_file ../../../Schmitt_et_al_2022_data/fig_2e_cyto.npz -model_info_file ../odes/MA_single.json -metab_params_file ../odes/metabolism_params.json -savedir ./test/ -ntune 10 -nsamples 10 -nchains 1 -nuts_sampler pymc -ncores 1