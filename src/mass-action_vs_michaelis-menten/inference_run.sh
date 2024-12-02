# MA - Duggleby and Clarke - Pyruvate kinase data
XLA_FLAGS="--xla_force_host_platform_device_count=6" python inference.py -model MA -free_params k_f,k_r,k_cat -data_file ./data/Duggleby_Clarke_1991_Fig2.csv -model_info_file ./MA_pyruvate_kinase.json -savedir ../../results/MA_v_MM/pyru_kin_ -nwarmup 10000 -nsamples 6000 -nchains 12 -rtol 1e-7 -atol 1e-7 -pcoeff 0.3 -icoeff 0.4 -sampler AIES

# MM - Duggleby and Clarke - Pyruvate kinase data
XLA_FLAGS="--xla_force_host_platform_device_count=6" python inference.py -model MM -free_params V_max,K_m -data_file ./data/Duggleby_Clarke_1991_Fig2.csv -model_info_file ./MM_pyruvate_kinase.json -savedir ../../results/MA_v_MM/pyru_kin_ -nwarmup 10000 -nsamples 6000 -nchains 12 -rtol 1e-7 -atol 1e-7 -pcoeff 0.3 -icoeff 0.4 -sampler AIES

# Hill - Duggleby and Clarke - Pyruvate kinase data
XLA_FLAGS="--xla_force_host_platform_device_count=6" python inference.py -model Hill -free_params V_max,K_m,n -data_file ./data/Duggleby_Clarke_1991_Fig2.csv -model_info_file ./Hill_pyruvate_kinase.json -savedir ../../results/MA_v_MM/pyru_kin_ -nwarmup 10000 -nsamples 6000 -nchains 12 -rtol 1e-7 -atol 1e-7 -pcoeff 0.3 -icoeff 0.4 -sampler AIES