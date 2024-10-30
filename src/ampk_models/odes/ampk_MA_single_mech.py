"""
    Nathaniel Linden (UCSD MAE)
    Created: February 24, 2023

Contains a function to return the sympy representation of the right hand side of the
AMPK_ma_single_mech model. This function is used to analyze the right null space of
the Jacobian matrix of the model to determine is mass is conserved in the model.

"""
import sympy as sym
import numpy as np

# Function to create dictionary for the parameters
#   all parameters are scalars, so use key/data pairs of 'state_name': ()
def ampk_MA_single_mech_get_params():
     return {
        'kOnAMP': (), # AMP binding
        'kOffAMP': (),
        'kOnADP': (), # ADP binding
        'kOffADP': (),
        'kOnATP': (), # ATP binding
        'kOffATP': (),
        'kOnCaMKK': (), # CaMKK binding
        'kOffCaMKK': (),
        'kPhosCaMKK': (), # phosphorylation
        'kOnLKB1': (), # LKB1 binding
        'kOffLKB1': (),
        'kPhosLKB1': (), # phosphorylation
        'kOnPP': (), # Phosphatase AMPK binding
        'kOffPP': (),
        'kDephosPP': (), # dephosphorylation
        'kOnAMPK': (), # pAMPK binds AMPKAR
        'kOffAMPK': (),
        'kPhosAMPK': (), # pAMPK phosphorylates AMPKAR
        'kOnPP1': (), # Phosphatase AMPKAR binding
        'kOffPP1': (), 
        'kDephosPP1': (),
        'kOffCaMKK': (),
        'kPhosCaMKK': (), # dephosphorylation
        # glycolysis flux
        'kGly':(),
        # ATP hydrolysis
        'kHydro':(),
        # Adenylate kinase
        'VforAK':(),
        'kmm':(),
        'kmd':(),
        'kmt':(),
        'KeqAK':(),
        # Oxidative phos
        'VmaxOxPhos':(),
        'Kadp':(),
        'n':(),
        
    }

# Function to create dictionary for the states
#   all states, so use key/data pairs of 'state_name': ()
def ampk_MA_single_mech_get_states():
    return {
        # free adenine nucleotides
        'AMP':(), 
        'ADP':(),
        'ATP':(),
        # free AMPK
        'AMPK':(),
        'pAMPK':(),
        # single AXP-AMPK complexes
        'AMP_AMPK':(),
        'ADP_AMPK':(),
        'ATP_AMPK':(),
        # single AXP-pAMPK complexes
        'AMP_pAMPK':(),
        'ADP_pAMPK':(),
        'ATP_pAMPK':(),
        # CaMKK complexes
        'CaMKK':(),
        'CaMKK_AMPK':(),
        'CaMKK_AMP_AMPK':(),
        'CaMKK_ADP_AMPK':(),
        'CaMKK_ATP_AMPK':(),
        # LKB1 complexes
        'LKB1':(),
        'LKB1_AMP_AMPK':(),
        'LKB1_ADP_AMPK':(),
        # AMPK phosphatase complexes
        'PP':(),
        'PP_pAMPK':(),
        'PP_ATP_pAMPK':(),
        # free AMPKAR
        'AMPKAR':(),
        'pAMPKAR':(),
        # AMPKAR-pAMPK complexes
        'AMPKAR_AMP_pAMPK':(),
        # AMPKAR phosphatase complexes
        'PP1':(),
        'PP1_pAMPKAR':(),
    }

def ampk_MA_single_mech_RHS_sympyFluxVars():
    """Right hand side of the AMPK_ma_double_mech regulation model.

    WARNING! This is different than the syntax for scipy.integrate!!
    Note from sunode syntax: "All inputs are dataclasses of sympy vars, or numpy
        arrays of sympy vars"
    """

    # create sympy vars for FLUXES
    num_fluxes = 26
    fluxes = sym.symbols(['J'+str(i) for i in range(1,num_fluxes+1)])
    Jgly = sym.symbols("Jgly")
    Jhydro= sym.symbols("Jhydro")
    JAK = sym.symbols("JAK")
    Joxphos = sym.symbols("Joxphos")

    for item in [Jgly, Jhydro, JAK, Joxphos]: # add metab fluxes to flux list
        fluxes.append(item)

    
    return {
        'AMP': -fluxes[0]-fluxes[3]-JAK,
        'ADP': -fluxes[1]-fluxes[4]-Jgly+2*JAK+Jhydro-Joxphos,
        'ATP': -fluxes[2]-fluxes[5]+Jgly-JAK-Jhydro+Joxphos,
        # free AMPK
        'AMPK': -fluxes[0]-fluxes[1]-fluxes[2]-fluxes[6]+fluxes[19],
        'pAMPK': -fluxes[3]-fluxes[4]-fluxes[5]+fluxes[7]-fluxes[18],
        # single AXP-AMPK complexes
        'AMP_AMPK': fluxes[0]-fluxes[8]-fluxes[14],
        'ADP_AMPK': fluxes[1]-fluxes[10]-fluxes[16],
        'ATP_AMPK': fluxes[2]-fluxes[12]+fluxes[21],
        # single AXP-pAMPK complexes
        'AMP_pAMPK': fluxes[3]+fluxes[9]+fluxes[15]-fluxes[22]+fluxes[23],
        'ADP_pAMPK': fluxes[4]+fluxes[11]+fluxes[17],
        'ATP_pAMPK': fluxes[5]+fluxes[13]-fluxes[20],
        # CaMKK complexes
        'CaMKK': -fluxes[6]+fluxes[7]-fluxes[8]+fluxes[9]-fluxes[10]+fluxes[11]-fluxes[12]+fluxes[13],
        'CaMKK_AMPK': fluxes[6]-fluxes[7],
        'CaMKK_AMP_AMPK': fluxes[8]-fluxes[9],
        'CaMKK_ADP_AMPK': fluxes[10]-fluxes[11],
        'CaMKK_ATP_AMPK': fluxes[12]-fluxes[13],
        # LKB1 complexes
        'LKB1': -fluxes[14]+fluxes[15]-fluxes[16]+fluxes[17],
        'LKB1_AMP_AMPK': fluxes[14]-fluxes[15],
        'LKB1_ADP_AMPK': fluxes[16]-fluxes[17],
        # AMPK phosphatase complexes
        'PP': -fluxes[18]+fluxes[19]-fluxes[20]+fluxes[21],
        'PP_pAMPK': fluxes[18]-fluxes[19],
        'PP_ATP_pAMPK': fluxes[20]-fluxes[21],
        # free AMPKAR
        'AMPKAR': -fluxes[22]+fluxes[25],
        'pAMPKAR': fluxes[23]-fluxes[24],
        # AMPKAR-pAMPK complexes
        'AMPKAR_AMP_pAMPK': fluxes[22]-fluxes[23],
        # AMPKAR phosphatase complexes
        'PP1': -fluxes[24]+fluxes[25],
        'PP1_pAMPKAR': fluxes[24]-fluxes[25]
    }, fluxes




# # Code for replacing flux terms with iterable indexes
# for i in reversed(range(27)):
#     #read input file
#     fin = open("../odes/tmp.txt", "rt")
#     #read file contents to string
#     data = fin.read()
#     #replace all occurrences of the required string
#     data = data.replace('J'+str(i), 'fluxes[{i}]'.format(i=i-1))
#     #close the input file
#     fin.close()
#     #open the input file in write mode
#     fin = open("../odes/tmp.txt", "wt")
#     #overrite the input file with the resulting data
#     fin.write(data)
#     #close the file
#     fin.close()