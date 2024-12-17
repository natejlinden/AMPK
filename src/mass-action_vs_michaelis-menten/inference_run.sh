# MA - Duggleby and Clarke - Pyruvate kinase data
XLA_FLAGS="--xla_force_host_platform_device_count=4" python inference_pymc.py -model MA -free_params k_f,k_r,k_cat -data_file ./data/Duggleby_Clarke_1991_Fig2.csv -model_info_file ./MA_pyruvate_kinase.json -savedir ../../results/MA_v_MM/pyru_kin_ -nwarmup 3000 -nsamples 2000 -nchains 4 -rtol 1e-9 -atol 1e-9 -pcoeff 0.3 -icoeff 0.4 -sampler NUTS -seed 0

# MM - Duggleby and Clarke - Pyruvate kinase data
XLA_FLAGS="--xla_force_host_platform_device_count=4" python inference_pymc.py -model MM -free_params V_max,K_m -data_file ./data/Duggleby_Clarke_1991_Fig2.csv -model_info_file ./MM_pyruvate_kinase.json -savedir ../../results/MA_v_MM/pyru_kin_ -nwarmup 3000 -nsamples 2000 -nchains 4 -rtol 1e-9 -atol 1e-9 -pcoeff 0.3 -icoeff 0.4 -sampler NUTS -seed 0

# Hill - Duggleby and Clarke - Pyruvate kinase data
XLA_FLAGS="--xla_force_host_platform_device_count=4" python inference_pymc.py -model Hill -free_params V_max,K_m,n -data_file ./data/Duggleby_Clarke_1991_Fig2.csv -model_info_file ./Hill_pyruvate_kinase.json -savedir ../../results/MA_v_MM/pyru_kin_ -nwarmup 3000 -nsamples 2000 -nchains 4 -prior_family "[['Gamma()',['alpha', 'beta']],['Gamma()',['alpha', 'beta']],['TruncatedNormal(lower=1)',['mu', 'sigma']]]" -pcoeff 0.3 -icoeff 0.4 -sampler NUTS -rtol 1e-10 -atol 1e-10 -seed 0
