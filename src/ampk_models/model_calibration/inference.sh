#! /bin/zsh
# script to call inference for the model calibration

# single binding, mass action, cytosol
XLA_FLAGS="--xla_force_host_platform_device_count=11" python inference_run.py -model MA_single_mech -free_params kOffAMP -data_file ../../../Schmitt_et_al_2022_data/fig_2e_cyto.npz -model_info_file ../odes/MA_single.json -metab_params_file ../odes/metabolism_params.json -savedir ./test/ -nwarmup 500 -nsamples 1000 -nchains 2