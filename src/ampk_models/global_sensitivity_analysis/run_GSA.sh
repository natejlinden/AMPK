echo "Sampling MA_double"
python GSA_sampling.py 4 ./ MA_double_mech ampk_MA_double_mech_diffrax \
    MA_double.json nominal_params_MA.csv param_bounds_MA.csv \
    compute_MA_params

echo "Sampling MA_single"
python GSA_sampling.py 4 ./ MA_single_mech ampk_MA_single_mech_diffrax \
    MA_single.json nominal_params_MA.csv param_bounds_MA.csv \
    compute_MA_params

echo "Sampling MM_double"
python GSA_sampling.py 4 ./ MM_double_mech ampk_MM_double_mech_diffrax \
    MM_double.json nominal_params_MM.csv param_bounds_MM.csv \
    compute_MM_params

echo "Sampling MM_single"
python GSA_sampling.py 4 ./ MM_single_mech ampk_MM_single_mech_diffrax \
    MM_single.json nominal_params_MM.csv param_bounds_MM.csv \
    compute_MM_params

echo "Sampling newmech_MA_single"
python GSA_sampling.py 4 ./ newmech_MA_double ampk_newmech_MA_single_diffrax \
    newmech_MA_single.json nominal_params_newmech_MA.csv param_bounds_newmech_MA.csv \
    compute_newmech_MA_params