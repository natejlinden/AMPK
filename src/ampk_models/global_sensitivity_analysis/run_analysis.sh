#echo "Running MA_double"
#python analyze_GSA.py MA_double_mech nominal_params_MA.csv param_bounds_MA.csv ../../../figures/
#python plot_trajectories.py ./ MA_double_mech ampk_MA_double_mech_diffrax \
#    MA_double.json compute_MA_params 100 ../../../figures/
#echo "done"

#echo "Running MA_single"
#python analyze_GSA.py MA_single_mech nominal_params_MA.csv param_bounds_MA.csv ../../../figures/
#python plot_trajectories.py ./ MA_single_mech ampk_MA_single_mech_diffrax \
#    MA_single.json compute_MA_params 100 ../../../figures/
#echo "done"

echo "Running MM_double"
python analyze_GSA.py MM_double_mech nominal_params_MM.csv param_bounds_MM.csv ../../../figures/
python plot_trajectories.py ./ MM_double_mech ampk_MM_double_mech_diffrax \
    MM_double.json compute_MM_params 100 ../../../figures/
echo "done"

echo "Running MM_single"
python analyze_GSA.py MM_single_mech nominal_params_MM.csv param_bounds_MM.csv ../../../figures/
python plot_trajectories.py ./ MM_single_mech ampk_MM_single_mech_diffrax \
    MM_single.json compute_MM_params 100 ../../../figures/
echo "done"

echo "Running newmech_MA_single"
python analyze_GSA.py newmech_MA_single nominal_params_newmech_MA.csv param_bounds_newmech_MA.csv ../../../figures/
python plot_trajectories.py ./ newmech_MA_single ampk_newmech_MA_single_diffrax \
    newmech_MA_single.json compute_newmech_MA_params 100 ../../../figures/
echo "done"

echo "Analysis complete"
