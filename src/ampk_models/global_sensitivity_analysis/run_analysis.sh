echo "Running GSA analysis"
python analyze_GSA.py

echo "plotting trajectories"
python MA_double_mech_trajectories.py
#python newmech_MA_single_trajectories.py

echo "Done"
