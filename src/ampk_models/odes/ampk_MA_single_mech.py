"""
    Nathaniel Linden (UCSD MAE)
    Created: February 24, 2023

    This file contains the functions for a model of MAPK activation. That model makes the 
    following high-level assumptions:
        - allow single adenine nucleotide AMPK binding
        - use only mass action kinetics
        - the reaction mechanism reflects specific activation and inhibition of
            AMPK and phos/dephos by AXPs

    The get_params functions returns a dictionary of the parameters in the correct
    format.

    The get_states function returns a dictionary od the parameters in the correct
    format.

    The RHS function is written following the syntax specified for the pymc/sunode
    package. See docs here https://sunode.readthedocs.io/en/latest/without_pymc.html
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
        'kForAK':(),
        'kRevAK':(),
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

def ampk_MA_single_mech_RHS(t, y, p):
    """Right hand side of the AMPK_ma_single_mech regulation model.

    WARNING! This is different than the syntax for scipy.integrate!!
    Note from sunode syntax: "All inputs are dataclasses of sympy vars, or numpy
        arrays of sympy vars"
    """
    
    # FLUXES
    # single AXP complexing
    J1 = p.kOnAMP*y.AMP*y.AMPK - p.kOffAMP*y.AMP_AMPK # AMPK
    J2 = p.kOnADP*y.ADP*y.AMPK - p.kOffADP*y.ADP_AMPK
    J3 = p.kOnATP*y.ATP*y.AMPK - p.kOffATP*y.ATP_AMPK
    J4 = p.kOnAMP*y.AMP*y.pAMPK - p.kOffAMP*y.AMP_pAMPK # pAMPK
    J5 = p.kOnADP*y.ADP*y.pAMPK - p.kOffADP*y.ADP_pAMPK
    J6 = p.kOnATP*y.ATP*y.pAMPK - p.kOffATP*y.ATP_pAMPK
    # CaMKK complexing and phosphorylation
    J7 = p.kOnCaMKK*y.CaMKK*y.AMPK - p.kOffCaMKK*y.CaMKK_AMPK
    J8 = p.kPhosCaMKK*y.CaMKK_AMPK
    J9 = p.kOnCaMKK*y.CaMKK*y.AMP_AMPK - p.kOffCaMKK*y.CaMKK_AMP_AMPK  
    J10 = p.kPhosCaMKK*y.CaMKK_AMP_AMPK 
    J11 = p.kOnCaMKK*y.CaMKK*y.ADP_AMPK - p.kOffCaMKK*y.CaMKK_ADP_AMPK
    J12 = p.kPhosCaMKK*y.CaMKK_ADP_AMPK
    J13 = p.kOnCaMKK*y.CaMKK*y.ATP_AMPK - p.kOffCaMKK*y.CaMKK_ATP_AMPK
    J14 = p.kPhosCaMKK*y.CaMKK_ATP_AMPK
    # LKB1 complexing and phosphorylation
    J15 = p.kOnLKB1*y.LKB1*y.AMP_AMPK - p.kOffLKB1*y.LKB1_AMP_AMPK
    J16 = p.kPhosLKB1*y.LKB1_AMP_AMPK
    J17 = p.kOnLKB1*y.LKB1*y.ADP_AMPK -  p.kOffLKB1*y.LKB1_ADP_AMPK
    J18 = p.kPhosLKB1*y.LKB1_ADP_AMPK
    # phosphatase binding and dephosphorylation
    J19 = p.kOnPP*y.PP*y.pAMPK - p.kOffPP*y.PP_pAMPK
    J20 = p. kDephosPP*y.PP_pAMPK
    J21 = p.kOnPP*y.PP*y.ATP_pAMPK - p.kOffPP*y.PP_ATP_pAMPK
    J22 = p. kDephosPP*y.PP_ATP_pAMPK
    # AMPK binding to AMPAKAR and phosphorylation
    J23 = p.kOnAMPK*y.AMPKAR*y.AMP_pAMPK - p.kOffAMPK*y.AMPKAR_AMP_pAMPK
    J24 = p.kPhosAMPK*y.AMPKAR_AMP_pAMPK
    # PP1 binding to AMPKAR and dephosphorylation  
    J25 = p.kOnPP1*y.PP1*y.pAMPKAR - p.kOffPP1*y.PP1_pAMPKAR
    J26 = p.kDephosPP1*y.PP1_pAMPKAR
    # Metabolic fluxes
    # glycolysis
    Jgly = 2*p.kGly*y.ADP*y.ADP
    # ATP hydrolysis
    Jhydro = p.kHydro*y.ATP
    # Adenylate Kinase
    Jak = (p.kForAK*y.ATP*y.AMP) - (p.kRevAK*y.ADP*y.ADP) # MASS ACTION KINETICS!
    # Oxidative Phos
    Joxphos = (p.VmaxOxPhos * ((y.ADP/p.Kadp)**p.n))/(1 + ((y.ADP/p.Kadp)**p.n))

    # now return the odes for each state variable
    return {
        'AMP': -J1-J4-Jak,
        'ADP': -J2-J5-Jgly+2*Jak+Jhydro-Joxphos,
        'ATP': -J3-J6+Jgly-Jak-Jhydro+Joxphos,
        # free AMPK
        'AMPK': -J1-J2-J3-J7+J20,
        'pAMPK': -J4-J5-J6+J8-J19,
        # single AXP-AMPK complexes
        'AMP_AMPK': J1-J9-J15,
        'ADP_AMPK': J2-J11-J17,
        'ATP_AMPK': J3-J13+J22,
        # single AXP-pAMPK complexes
        'AMP_pAMPK': J4+J10+J16-J23+J24,
        'ADP_pAMPK': J5+J12+J18,
        'ATP_pAMPK': J6+J14-J21,
        # CaMKK complexes
        'CaMKK': -J7+J8-J9+J10-J11+J12-J13+J14,
        'CaMKK_AMPK': J7-J8,
        'CaMKK_AMP_AMPK': J9-J10,
        'CaMKK_ADP_AMPK': J11-J12,
        'CaMKK_ATP_AMPK': J13-J14,
        # LKB1 complexes
        'LKB1': -J15+J16-J17+J18,
        'LKB1_AMP_AMPK': J15-J16,
        'LKB1_ADP_AMPK': J17-J18,
        # AMPK phosphatase complexes
        'PP': -J19+J20-J21+J22,
        'PP_pAMPK': J19-J20,
        'PP_ATP_pAMPK': J21-J22,
        # free AMPKAR
        'AMPKAR': -J23+J26,
        'pAMPKAR': J24-J25,
        # AMPKAR-pAMPK complexes
        'AMPKAR_AMP_pAMPK': J23-J24,
        # AMPKAR phosphatase complexes
        'PP1': -J25+J26,
        'PP1_pAMPKAR': J25-J26
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
    Jak = sym.symbols("Jak")
    Joxphos = sym.symbols("Joxphos")

    for item in [Jgly, Jhydro, Jak, Joxphos]: # add metab fluxes to flux list
        fluxes.append(item)

    
    return {
        'AMP': -fluxes[0]-fluxes[3]-Jak,
        'ADP': -fluxes[1]-fluxes[4]-Jgly+2*Jak+Jhydro-Joxphos,
        'ATP': -fluxes[2]-fluxes[5]+Jgly-Jak-Jhydro+Joxphos,
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