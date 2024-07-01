"""
    Nathaniel Linden (UCSD MAE)
    Created: February 17, 2023

    This file contains the functions for a model of MAPK activation. That model makes the 
    following high-level assumptions:
        - allow double adenine nucleotide AMPK binding
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
def ampk_MA_double_mech_get_params():
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
def ampk_MA_double_mech_get_states():
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
        # double AXP-AMPK complexes
        'AMP_AMP_AMPK':(),
        'AMP_ADP_AMPK':(),
        'AMP_ATP_AMPK':(),
        'ADP_ADP_AMPK':(),
        'ADP_ATP_AMPK':(),
        'ATP_ATP_AMPK':(),
        # double AXP-pAMPK complexes
        'AMP_AMP_pAMPK':(),
        'AMP_ADP_pAMPK':(),
        'AMP_ATP_pAMPK':(),
        'ADP_ADP_pAMPK':(),
        'ADP_ATP_pAMPK':(),
        'ATP_ATP_pAMPK':(),
        # CaMKK complexes
        'CaMKK_AMPK':(),
        'CaMKK_AMP_AMPK':(),
        'CaMKK_ADP_AMPK':(),
        'CaMKK_ATP_AMPK':(),
        'CaMKK_AMP_AMP_AMPK':(),
        'CaMKK_AMP_ADP_AMPK':(),
        'CaMKK_AMP_ATP_AMPK':(),
        'CaMKK_ADP_ADP_AMPK':(),
        'CaMKK_ADP_ATP_AMPK':(),
        'CaMKK_ATP_ATP_AMPK':(),
        # LKB1 complexes
        'LKB1_AMP_AMPK':(),
        'LKB1_ADP_AMPK':(),
        'LKB1_AMP_AMP_AMPK':(),
        'LKB1_AMP_ADP_AMPK':(),
        'LKB1_ADP_ADP_AMPK':(),
        # AMPK phosphatase complexes
        'PP_pAMPK':(),
        'PP_ATP_pAMPK':(),
        'PP_AMP_ATP_pAMPK':(),
        'PP_ADP_ATP_pAMPK':(),
        'PP_ATP_ATP_pAMPK':(),
        # free AMPKAR
        'AMPKAR':(),
        'pAMPKAR':(),
        # AMPKAR-pAMPK complexes
        'AMPKAR_AMP_pAMPK':(),
        'AMPKAR_AMP_AMP_pAMPK':(),
        'AMPKAR_AMP_ADP_pAMPK':(),
        # AMPKAR phosphatase complexes
        'PP1_pAMPKAR':(),
        'CaMKK':(),
        'LKB1':(),
        'PP':(),
        'PP1':(),
    }
def ampk_MA_double_mech_RHS(t, y, p):
    """Right hand side of the AMPK_ma_double_mech regulation model.

    WARNING! This is different than the syntax for scipy.integrate!!
    Note from sunode syntax: "All inputs are dataclasses of sympy vars, or numpy
        arrays of sympy vars"
    """
    # vectorize sympy operations
    mul = np.vectorize(sym.Mul)
    
    # FLUXES
    # TODO: check if it is okay to write fluxes first like this!
    # single AXP complexing
    J1 = mul(p.kOnAMP, y.AMP, y.AMPK) - mul(p.kOffAMP, y.AMP_AMPK) # AMPK
    J2 = mul(p.kOnADP, y.ADP, y.AMPK) - mul(p.kOffADP, y.ADP_AMPK)
    J3 = mul(p.kOnATP, y.ATP, y.AMPK) - mul(p.kOffATP, y.ATP_AMPK)
    J4 = mul(p.kOnAMP, y.AMP, y.pAMPK) - mul(p.kOffAMP, y.AMP_pAMPK) # pAMPK
    J5 = mul(p.kOnADP, y.ADP, y.pAMPK) - mul(p.kOffADP, y.ADP_pAMPK)
    J6 = mul(p.kOnATP, y.ATP, y.pAMPK) - mul(p.kOffATP, y.ATP_pAMPK)
    # double AXP complexing
    J7 = mul(p.kOnAMP, y.AMP, y.AMP_AMPK) - mul(p.kOffAMP, y.AMP_AMP_AMPK) # AMPK
    J8 = mul(p.kOnAMP, y.AMP, y.ADP_AMPK) - mul(p.kOffAMP, y.AMP_ADP_AMPK)
    J9 = mul(p.kOnAMP, y.AMP, y.ATP_AMPK) - mul(p.kOffAMP, y.AMP_ATP_AMPK)
    J10 = mul(p.kOnADP, y.ADP, y.AMP_AMPK) - mul(p.kOffADP, y.AMP_ADP_AMPK)
    J11 = mul(p.kOnADP, y.ADP, y.ADP_AMPK) - mul(p.kOffADP, y.ADP_ADP_AMPK)
    J12 = mul(p.kOnADP, y.ADP, y.ATP_AMPK) - mul(p.kOffADP, y.ADP_ATP_AMPK)
    J13 = mul(p.kOnATP, y.ATP, y.AMP_AMPK) -  mul(p.kOffATP, y.AMP_ATP_AMPK)
    J14 = mul(p.kOnATP, y.ATP, y.ADP_AMPK) - mul(p.kOffATP, y.ADP_ATP_AMPK)
    J15 = mul(p.kOnATP, y.ATP, y.ATP_AMPK) - mul(p.kOffATP, y.ATP_ATP_AMPK)
    J16 = mul(p.kOnAMP, y.AMP, y.AMP_pAMPK) - mul(p.kOffAMP, y.AMP_AMP_pAMPK) # pAMPK
    J17 = mul(p.kOnAMP, y.AMP, y.ADP_pAMPK) - mul(p.kOffAMP, y.AMP_ADP_pAMPK)
    J18 = mul(p.kOnAMP, y.AMP, y.ATP_pAMPK) - mul(p.kOffAMP, y.AMP_ATP_pAMPK)
    J19 = mul(p.kOnADP, y.ADP, y.AMP_pAMPK) - mul(p.kOffADP, y.AMP_ADP_pAMPK)
    J20 = mul(p.kOnADP, y.ADP, y.ADP_pAMPK) - mul(p.kOffADP, y.ADP_ADP_pAMPK)
    J21 = mul(p.kOnADP, y.ADP, y.ATP_pAMPK) - mul(p.kOffADP, y.ADP_ATP_pAMPK)
    J22 = mul(p.kOnATP, y.ATP, y.AMP_pAMPK) - mul(p.kOffATP, y.AMP_ATP_pAMPK)
    J23 = mul(p.kOnATP, y.ATP, y.ADP_pAMPK) - mul(p.kOffATP, y.ADP_ATP_pAMPK)
    J24 = mul(p.kOnATP, y.ATP, y.ATP_pAMPK) - mul(p.kOffATP, y.ATP_ATP_pAMPK)
    # CaMKK complexing and phosphorylation    
    J25 = mul(p.kOnCaMKK, y.CaMKK, y.AMPK) - mul(p.kOffCaMKK, y.CaMKK_AMPK)
    J26 = mul(p.kPhosCaMKK, y.CaMKK_AMPK)   
    J27 = mul(p.kOnCaMKK, y.CaMKK, y.AMP_AMPK) - mul(p.kOffCaMKK, y.CaMKK_AMP_AMPK)   
    J28 = mul(p.kPhosCaMKK, y.CaMKK_AMP_AMPK)  
    J29 = mul(p.kOnCaMKK, y.CaMKK, y.ADP_AMPK) - mul(p.kOffCaMKK, y.CaMKK_ADP_AMPK)   
    J30 = mul(p.kPhosCaMKK, y.CaMKK_ADP_AMPK)  
    J31 = mul(p.kOnCaMKK, y.CaMKK, y.ATP_AMPK) - mul(p.kOffCaMKK, y.CaMKK_ATP_AMPK)   
    J32 = mul(p.kPhosCaMKK, y.CaMKK_ATP_AMPK) 
    J33 = mul(p.kOnCaMKK, y.CaMKK, y.AMP_AMP_AMPK) - mul(p.kOffCaMKK, y.CaMKK_AMP_AMP_AMPK)   
    J34 = mul(p.kPhosCaMKK, y.CaMKK_AMP_AMP_AMPK)  
    J35 = mul(p.kOnCaMKK, y.CaMKK, y.AMP_ADP_AMPK) - mul(p.kOffCaMKK, y.CaMKK_AMP_ADP_AMPK)   
    J36 = mul(p.kPhosCaMKK, y.CaMKK_AMP_ADP_AMPK)  
    J37 = mul(p.kOnCaMKK, y.CaMKK, y.AMP_ATP_AMPK) - mul(p.kOffCaMKK, y.CaMKK_AMP_ATP_AMPK)   
    J38 = mul(p.kPhosCaMKK, y.CaMKK_AMP_ATP_AMPK)  
    J39 = mul(p.kOnCaMKK, y.CaMKK, y.ADP_ADP_AMPK) - mul(p.kOffCaMKK, y.CaMKK_ADP_ADP_AMPK)   
    J40 = mul(p.kPhosCaMKK, y.CaMKK_ADP_ADP_AMPK)  
    J41 = mul(p.kOnCaMKK, y.CaMKK, y.ADP_ATP_AMPK) - mul(p.kOffCaMKK, y.CaMKK_ADP_ATP_AMPK)   
    J42 = mul(p.kPhosCaMKK, y.CaMKK_ADP_ATP_AMPK)  
    J43 = mul(p.kOnCaMKK, y.CaMKK, y.ATP_ATP_AMPK) - mul(p.kOffCaMKK, y.CaMKK_ATP_ATP_AMPK)   
    J44 = mul(p.kPhosCaMKK, y.CaMKK_ATP_ATP_AMPK)
    # LKB1 complexing and phosphorylation
    J45 = mul(p.kOnLKB1, y.LKB1, y.AMP_AMPK) - mul(p.kOffLKB1, y.LKB1_AMP_AMPK)   
    J46 = mul(p.kPhosLKB1, y.LKB1_AMP_AMPK)  
    J47 = mul(p.kOnLKB1, y.LKB1, y.ADP_AMPK) - mul(p.kOffLKB1, y.LKB1_ADP_AMPK)   
    J48 = mul(p.kPhosLKB1, y.LKB1_ADP_AMPK)  
    J49 = mul(p.kOnLKB1, y.LKB1, y.AMP_AMP_AMPK) - mul(p.kOffLKB1, y.LKB1_AMP_AMP_AMPK)   
    J50 = mul(p.kPhosLKB1, y.LKB1_AMP_AMP_AMPK)  
    J51 = mul(p.kOnLKB1, y.LKB1, y.AMP_ADP_AMPK) - mul(p.kOffLKB1, y.LKB1_AMP_ADP_AMPK)   
    J52 = mul(p.kPhosLKB1, y.LKB1_AMP_ADP_AMPK)  
    J53 = mul(p.kOnLKB1, y.LKB1, y.ADP_ADP_AMPK) - mul(p.kOffLKB1, y.LKB1_ADP_ADP_AMPK)   
    J54 = mul(p.kPhosLKB1, y.LKB1_ADP_ADP_AMPK)  
    # phosphatase binding and dephosphorylation
    J55 = mul(p.kOnPP, y.PP, y.pAMPK) - mul(p.kOffPP, y.PP_pAMPK)    
    J56 = mul(p. kDephosPP, y.PP_pAMPK)
    J57 = mul(p.kOnPP, y.PP, y.ATP_pAMPK) - mul(p.kOffPP, y.PP_ATP_pAMPK)    
    J58 = mul(p. kDephosPP, y.PP_ATP_pAMPK)
    J59 = mul(p.kOnPP, y.PP, y.AMP_ATP_pAMPK) - mul(p.kOffPP, y.PP_AMP_ATP_pAMPK)    
    J60 = mul(p. kDephosPP, y.PP_AMP_ATP_pAMPK)
    J61 = mul(p.kOnPP, y.PP, y.ADP_ATP_pAMPK) - mul(p.kOffPP, y.PP_ADP_ATP_pAMPK)    
    J62 = mul(p. kDephosPP, y.PP_ADP_ATP_pAMPK)
    J63 = mul(p.kOnPP, y.PP, y.ATP_ATP_pAMPK)  - mul(p.kOffPP, y.PP_ATP_ATP_pAMPK)
    J64 = mul(p. kDephosPP, y.PP_ATP_ATP_pAMPK)
    # AMPK binding to AMPAKAR and phosphorylation
    J65 = mul(p.kOnAMPK, y.AMPKAR, y.AMP_pAMPK) - mul(p.kOffAMPK, y.AMPKAR_AMP_pAMPK)   
    J66 = mul(p.kPhosAMPK, y.AMPKAR_AMP_pAMPK)  
    J67 = mul(p.kOnAMPK, y.AMPKAR, y.AMP_AMP_pAMPK) - mul(p.kOffAMPK, y.AMPKAR_AMP_AMP_pAMPK)   
    J68 = mul(p.kPhosAMPK, y.AMPKAR_AMP_AMP_pAMPK)  
    J69 = mul(p.kOnAMPK, y.AMPKAR, y.AMP_ADP_pAMPK) - mul(p.kOffAMPK, y.AMPKAR_AMP_ADP_pAMPK)   
    J70 = mul(p.kPhosAMPK, y.AMPKAR_AMP_ADP_pAMPK)
    # PP1 binding to AMPKAR and dephosphorylation  
    J71 = mul(p.kOnPP1, y.PP1, y.pAMPKAR) - mul(p.kOffPP1, y.PP1_pAMPKAR)
    J72 = mul(p.kDephosPP1, y.PP1_pAMPKAR)
    # Metabolic fluxes
    # glycolysis
    Jgly = mul(2, p.kGly, y.ADP, y.ADP)
    # ATP hydrolysis
    Jhydro = mul(p.kHydro, y.ATP)
    # Adenylate Kinase
    num_for = (p.VforAK*y.ATP*y.AMP)/(p.kmt*p.kmm)
    den = (1 + (y.ATP/p.kmt) + (y.AMP/p.kmm) + ((y.ATP*y.AMP)/(p.kmt*p.kmm)) + 
                ((2*y.ADP)/p.kmd) + ((y.ADP**2)/(p.kmd**2)))
    VrevAK = (p.VforAK*(p.kmd**2))/(p.KeqAK*p.kmt*p.kmm)
    num_rev = (VrevAK*(y.ADP**2))/(p.kmd**2)
    JAK = (num_for - num_rev)/den
    # Oxidative Phos
    Joxphos = (p.VmaxOxPhos * ((y.ADP/p.Kadp)**p.n))/(1 + ((y.ADP/p.Kadp)**p.n))

    # now return the odes for each state variable
    return {
        'AMP': -J1-J4-J7-J8-J9-J16-J17-J18-JAK,
        'ADP': -J2-J5-J10-J11-J12-J19-J20-J21-Jgly+2*JAK+Jhydro-Joxphos,
        'ATP': -J3-J6-J13-J14-J15-J22-J23-J24+Jgly-JAK-Jhydro+Joxphos,
        # free AMPK
        'AMPK': -J1-J2-J3-J25+J56,
        'pAMPK': -J4-J5-J6+J26-J55,
        # single AXP-AMPK complexes
        'AMP_AMPK': J1-J7-J10-J13-J27-J45,
        'ADP_AMPK': J2-J8-J11-J14-J29-J47,
        'ATP_AMPK': J3-J9-J12-J15-J31+J58,
        # single AXP-pAMPK complexes
        'AMP_pAMPK': J4-J16-J19-J22+J28+J46-J65+J66,
        'ADP_pAMPK': J5-J17-J20-J23+J30+J48,
        'ATP_pAMPK': J6-J18-J21-J24+J32-J57,
        # double AXP-AMPK complexes
        'AMP_AMP_AMPK': J7-J33-J49,
        'AMP_ADP_AMPK': J8+J10-J35-J51,
        'AMP_ATP_AMPK': J9+J13-J37+J60,
        'ADP_ADP_AMPK': J11-J39-J53,
        'ADP_ATP_AMPK': J12+J14-J41+J62,
        'ATP_ATP_AMPK': J15-J43+J64,
        # double AXP-pAMPK complexes
        'AMP_AMP_pAMPK': J16+J34+J50-J67+J68,
        'AMP_ADP_pAMPK': J17+J19+J36+J52-J69+J70,
        'AMP_ATP_pAMPK': J18+J22+J38-J59,
        'ADP_ADP_pAMPK': J20+J40+J54,
        'ADP_ATP_pAMPK': J21+J23+J42-J61,
        'ATP_ATP_pAMPK': J24+J44-J63,
        # CaMKK complexes
        'CaMKK': -J25+J26-J27+J28-J29+J30-J31+J32-J33+J34-J35+J36-J37+J38-J39+J40-J41+J42-J43+J44,
        'CaMKK_AMPK': J25-J26,
        'CaMKK_AMP_AMPK': J27-J28,
        'CaMKK_ADP_AMPK': J29-J30,
        'CaMKK_ATP_AMPK': J31-J32,
        'CaMKK_AMP_AMP_AMPK': J33-J34,
        'CaMKK_AMP_ADP_AMPK': J35-J36,
        'CaMKK_AMP_ATP_AMPK': J37-J38,
        'CaMKK_ADP_ADP_AMPK': J39-J40,
        'CaMKK_ADP_ATP_AMPK': J41-J42,
        'CaMKK_ATP_ATP_AMPK': J43-J44,
        # LKB1 complexes
        'LKB1': -J45+J46-J47+J48-J49+J50-J51+J52-J53+J54,
        'LKB1_AMP_AMPK': J45-J46,
        'LKB1_ADP_AMPK': J47-J48,
        'LKB1_AMP_AMP_AMPK': J49-J50,
        'LKB1_AMP_ADP_AMPK': J51-J52,
        'LKB1_ADP_ADP_AMPK': J53-J54,
        # AMPK phosphatase complexes
        'PP': -J55+J56-J57+J58-J59+J60-J61+J62-J63+J64,
        'PP_pAMPK': J55-J56,
        'PP_ATP_pAMPK': J57-J58,
        'PP_AMP_ATP_pAMPK': J59-J60,
        'PP_ADP_ATP_pAMPK': J61-J62,
        'PP_ATP_ATP_pAMPK': J63-J64,
        # free AMPKAR
        'AMPKAR': -J65-J67-J69+J72,
        'pAMPKAR': J66+J68+J70-J71,
        # AMPKAR-pAMPK complexes
        'AMPKAR_AMP_pAMPK': J65-J66,
        'AMPKAR_AMP_AMP_pAMPK': J67-J68,
        'AMPKAR_AMP_ADP_pAMPK': J69-J70,
        # AMPKAR phosphatase complexes
        'PP1': -J71+J72,
        'PP1_pAMPKAR': J71-J72,
    }

def ampk_MA_double_mech_RHS_sympyFluxVars():
    """Right hand side of the AMPK_ma_double_mech regulation model.

    WARNING! This is different than the syntax for scipy.integrate!!
    Note from sunode syntax: "All inputs are dataclasses of sympy vars, or numpy
        arrays of sympy vars"
    """

    # create sympy vars for FLUXES
    num_fluxes = 72
    fluxes = sym.symbols(['J'+str(i) for i in range(1,num_fluxes+1)])
    Jgly = sym.symbols("Jgly")
    Jhydro= sym.symbols("Jhydro")
    JAK = sym.symbols("JAK")
    Joxphos = sym.symbols("Joxphos")

    for item in [Jgly, Jhydro, JAK, Joxphos]: # add metab fluxes to flux list
        fluxes.append(item)

    
    return {
        'AMP': -fluxes[0]-fluxes[3]-fluxes[6]-fluxes[7]-fluxes[8]-fluxes[15]-fluxes[16]-fluxes[17]-JAK,
        'ADP': -fluxes[1]-fluxes[4]-fluxes[9]-fluxes[10]-fluxes[11]-fluxes[18]-fluxes[19]-fluxes[20]-Jgly+2*JAK+Jhydro-Joxphos,
        'ATP': -fluxes[2]-fluxes[5]-fluxes[12]-fluxes[13]-fluxes[14]-fluxes[21]-fluxes[22]-fluxes[23]+Jgly-JAK-Jhydro+Joxphos,
        # free AMPK
        'AMPK': -fluxes[0]-fluxes[1]-fluxes[2]-fluxes[24]+fluxes[55],
        'pAMPK': -fluxes[3]-fluxes[4]-fluxes[5]+fluxes[25]-fluxes[54],
        # single AXP-AMPK complexes
        'AMP_AMPK': fluxes[0]-fluxes[6]-fluxes[9]-fluxes[12]-fluxes[26]-fluxes[44],
        'ADP_AMPK': fluxes[1]-fluxes[7]-fluxes[10]-fluxes[13]-fluxes[28]-fluxes[46],
        'ATP_AMPK': fluxes[2]-fluxes[8]-fluxes[11]-fluxes[14]-fluxes[30]+fluxes[57],
        # single AXP-pAMPK complexes
        'AMP_pAMPK': fluxes[3]-fluxes[15]-fluxes[18]-fluxes[21]+fluxes[27]+fluxes[45]-fluxes[64]+fluxes[65],
        'ADP_pAMPK': fluxes[4]-fluxes[16]-fluxes[19]-fluxes[22]+fluxes[29]+fluxes[47],
        'ATP_pAMPK': fluxes[5]-fluxes[17]-fluxes[20]-fluxes[23]+fluxes[31]-fluxes[56],
        # double AXP-AMPK complexes
        'AMP_AMP_AMPK': fluxes[6]-fluxes[32]-fluxes[48],
        'AMP_ADP_AMPK': fluxes[7]+fluxes[9]-fluxes[34]-fluxes[50],
        'AMP_ATP_AMPK': fluxes[8]+fluxes[12]-fluxes[36]+fluxes[59],
        'ADP_ADP_AMPK': fluxes[10]-fluxes[38]-fluxes[52],
        'ADP_ATP_AMPK': fluxes[11]+fluxes[13]-fluxes[40]+fluxes[61],
        'ATP_ATP_AMPK': fluxes[14]-fluxes[42]+fluxes[63],
        # double AXP-pAMPK complexes
        'AMP_AMP_pAMPK': fluxes[15]+fluxes[33]+fluxes[49]-fluxes[66]+fluxes[67],
        'AMP_ADP_pAMPK': fluxes[16]+fluxes[18]+fluxes[35]+fluxes[51]-fluxes[68]+fluxes[69],
        'AMP_ATP_pAMPK': fluxes[17]+fluxes[21]+fluxes[37]-fluxes[58],
        'ADP_ADP_pAMPK': fluxes[19]+fluxes[39]+fluxes[53],
        'ADP_ATP_pAMPK': fluxes[20]+fluxes[22]+fluxes[41]-fluxes[60],
        'ATP_ATP_pAMPK': fluxes[23]+fluxes[43]-fluxes[62],
        # CaMKK complexes
        'CaMKK': -fluxes[24]+fluxes[25]-fluxes[26]+fluxes[27]-fluxes[28]+fluxes[29]-fluxes[30]+fluxes[31]-fluxes[32]+fluxes[33]-fluxes[34]+fluxes[35]-fluxes[36]+fluxes[37]-fluxes[38]+fluxes[39]-fluxes[40]+fluxes[41]-fluxes[42]+fluxes[43],
        'CaMKK_AMPK': fluxes[24]-fluxes[25],
        'CaMKK_AMP_AMPK': fluxes[26]-fluxes[27],
        'CaMKK_ADP_AMPK': fluxes[28]-fluxes[29],
        'CaMKK_ATP_AMPK': fluxes[30]-fluxes[31],
        'CaMKK_AMP_AMP_AMPK': fluxes[32]-fluxes[33],
        'CaMKK_AMP_ADP_AMPK': fluxes[34]-fluxes[35],
        'CaMKK_AMP_ATP_AMPK': fluxes[36]-fluxes[37],
        'CaMKK_ADP_ADP_AMPK': fluxes[38]-fluxes[39],
        'CaMKK_ADP_ATP_AMPK': fluxes[40]-fluxes[41],
        'CaMKK_ATP_ATP_AMPK': fluxes[42]-fluxes[43],
        # LKB1 complexes
        'LKB1': -fluxes[44]+fluxes[45]-fluxes[46]+fluxes[47]-fluxes[48]+fluxes[49]-fluxes[50]+fluxes[51]-fluxes[52]+fluxes[53],
        'LKB1_AMP_AMPK': fluxes[44]-fluxes[45],
        'LKB1_ADP_AMPK': fluxes[46]-fluxes[47],
        'LKB1_AMP_AMP_AMPK': fluxes[48]-fluxes[49],
        'LKB1_AMP_ADP_AMPK': fluxes[50]-fluxes[51],
        'LKB1_ADP_ADP_AMPK': fluxes[52]-fluxes[53],
        # AMPK phosphatase complexes
        'PP': -fluxes[54]+fluxes[55]-fluxes[56]+fluxes[57]-fluxes[58]+fluxes[59]-fluxes[60]+fluxes[61]-fluxes[62]+fluxes[63],
        'PP_pAMPK': fluxes[54]-fluxes[55],
        'PP_ATP_pAMPK': fluxes[56]-fluxes[57],
        'PP_AMP_ATP_pAMPK': fluxes[58]-fluxes[59],
        'PP_ADP_ATP_pAMPK': fluxes[60]-fluxes[61],
        'PP_ATP_ATP_pAMPK': fluxes[62]-fluxes[63],
        # free AMPKAR
        'AMPKAR': -fluxes[64]-fluxes[66]-fluxes[68]+fluxes[71],
        'pAMPKAR': fluxes[65]+fluxes[67]+fluxes[69]-fluxes[70],
        # AMPKAR-pAMPK complexes
        'AMPKAR_AMP_pAMPK': fluxes[64]-fluxes[65],
        'AMPKAR_AMP_AMP_pAMPK': fluxes[66]-fluxes[67],
        'AMPKAR_AMP_ADP_pAMPK': fluxes[68]-fluxes[69],
        # AMPKAR phosphatase complexes
        'PP1':-fluxes[70]+fluxes[71],
        'PP1_pAMPKAR': fluxes[70]-fluxes[71],
    }, fluxes

# # Code for replacing flux terms with iterable indexes
# for i in range(10, 73):
#     #read input file
#     fin = open("temp.txt", "rt")
#     #read file contents to string
#     data = fin.read()
#     #replace all occurrences of the required string
#     data = data.replace('J'+str(i), 'fluxes[{i}]'.format(i=i-1))
#     #close the input file
#     fin.close()
#     #open the input file in write mode
#     fin = open("temp.txt", "wt")
#     #overrite the input file with the resulting data
#     fin.write(data)
#     #close the file
#     fin.close()

def ampk_MA_double_mech_RHS_scipy(t, y, p):
    """Right hand side of the AMPK_ma_double_mech regulation model.
    """
    # FLUXES
    # single AXP complexing
    J1 = p['kOnAMP']*y[0]*y[3] - p['kOffAMP']*y[5] # AMPK
    J2 = p['kOnADP']*y[1]*y[3] - p['kOffADP']*y[6]
    J3 = p['kOnATP']*y[2]*y[3] - p['kOffATP']*y[7]
    J4 = p['kOnAMP']*y[0]*y[4] - p['kOffAMP']*y[8] # pAMPK
    J5 = p['kOnADP']*y[1]*y[4] - p['kOffADP']*y[9]
    J6 = p['kOnATP']*y[2]*y[4] - p['kOffATP']*y[10]
    # double AXP complexing
    J7 = p['kOnAMP']*y[0]*y[5] - p['kOffAMP']*y[11] # AMPK
    J8 = p['kOnAMP']*y[0]*y[6] - p['kOffAMP']*y[12]
    J9 = p['kOnAMP']*y[0]*y[7] - p['kOffAMP']*y[13]
    J10 = p['kOnADP']*y[1]*y[5] - p['kOffADP']*y[12]
    J11 = p['kOnADP']*y[1]*y[6] - p['kOffADP']*y[14]
    J12 = p['kOnADP']*y[1]*y[7] - p['kOffADP']*y[15]
    J13 = p['kOnATP']*y[2]*y[5] -  p['kOffATP']*y[13]
    J14 = p['kOnATP']*y[2]*y[6] - p['kOffATP']*y[15]
    J15 = p['kOnATP']*y[2]*y[7] - p['kOffATP']*y[16]
    J16 = p['kOnAMP']*y[0]*y[8] - p['kOffAMP']*y[17] # pAMPK
    J17 = p['kOnAMP']*y[0]*y[9] - p['kOffAMP']*y[18]
    J18 = p['kOnAMP']*y[0]*y[10] - p['kOffAMP']*y[19]
    J19 = p['kOnADP']*y[1]*y[8] - p['kOffADP']*y[18]
    J20 = p['kOnADP']*y[1]*y[9] - p['kOffADP']*y[20]
    J21 = p['kOnADP']*y[1]*y[10] - p['kOffADP']*y[21]
    J22 = p['kOnATP']*y[2]*y[8] - p['kOffATP']*y[19]
    J23 = p['kOnATP']*y[2]*y[9] - p['kOffATP']*y[21]
    J24 = p['kOnATP']*y[2]*y[10] - p['kOffATP']*y[22]
    # CaMKK'] complexing and phosphorylation    
    J25 = p['kOnCaMKK']*y[23]*y[3] - p['kOffCaMKK']*y[24]
    J26 = p['kPhosCaMKK']*y[24]
    J27 = p['kOnCaMKK']*y[23]*y[5] - p['kOffCaMKK']*y[25]   
    J28 = p['kPhosCaMKK']*y[25] 
    J29 = p['kOnCaMKK']*y[23]*y[6] - p['kOffCaMKK']*y[26]   
    J30 = p['kPhosCaMKK']*y[26]  
    J31 = p['kOnCaMKK']*y[23]*y[7] - p['kOffCaMKK']*y[27]   
    J32 = p['kPhosCaMKK']*y[27] 
    J33 = p['kOnCaMKK']*y[23]*y[11] - p['kOffCaMKK']*y[28]   
    J34 = p['kPhosCaMKK']*y[28]  
    J35 = p['kOnCaMKK']*y[23]*y[12] - p['kOffCaMKK']*y[29]   
    J36 = p['kPhosCaMKK']*y[29]  
    J37 = p['kOnCaMKK']*y[23]*y[13] - p['kOffCaMKK']*y[30]   
    J38 = p['kPhosCaMKK']*y[30]  
    J39 = p['kOnCaMKK']*y[23]*y[14] - p['kOffCaMKK']*y[31]   
    J40 = p['kPhosCaMKK']*y[31]  
    J41 = p['kOnCaMKK']*y[23]*y[15] - p['kOffCaMKK']*y[32]   
    J42 = p['kPhosCaMKK']*y[32]  
    J43 = p['kOnCaMKK']*y[23]*y[16] - p['kOffCaMKK']*y[33]   
    J44 = p['kPhosCaMKK']*y[33]
    # LKB1 complexing and phosphorylation
    J45 = p['kOnLKB1']*y[34]*y[5] - p['kOffLKB1']*y[35]   
    J46 = p['kPhosLKB1']*y[35]  
    J47 = p['kOnLKB1']*y[34]*y[6] - p['kOffLKB1']*y[36]   
    J48 = p['kPhosLKB1']*y[36]  
    J49 = p['kOnLKB1']*y[34]*y[11] - p['kOffLKB1']*y[37]   
    J50 = p['kPhosLKB1']*y[37]  
    J51 = p['kOnLKB1']*y[34]*y[12] - p['kOffLKB1']*y[38]   
    J52 = p['kPhosLKB1']*y[38]  
    J53 = p['kOnLKB1']*y[34]*y[14] - p['kOffLKB1']*y[39]   
    J54 = p['kPhosLKB1']*y[39]  
    # phosphatase binding and dephosphorylation
    J55 = p['kOnPP']*y[40]*y[4] - p['kOffPP']*y[41]    
    J56 = p['kDephosPP']*y[41]
    J57 = p['kOnPP']*y[40]*y[10] - p['kOffPP']*y[42]    
    J58 = p['kDephosPP']*y[42]
    J59 = p['kOnPP']*y[40]*y[19] - p['kOffPP']*y[43]    
    J60 = p['kDephosPP']*y[43]
    J61 = p['kOnPP']*y[40]*y[21] - p['kOffPP']*y[44]    
    J62 = p['kDephosPP']*y[44]
    J63 = p['kOnPP']*y[40]*y[22]  - p['kOffPP']*y[45]
    J64 = p['kDephosPP']*y[45]
    # AMPK'] binding to AMPAKAR and phosphorylation
    J65 = p['kOnAMPK']*y[46]*y[8] - p['kOffAMPK']*y[48]   
    J66 = p['kPhosAMPK']*y[48]  
    J67 = p['kOnAMPK']*y[46]*y[17] - p['kOffAMPK']*y[49]   
    J68 = p['kPhosAMPK']*y[49]  
    J69 = p['kOnAMPK']*y[46]*y[18] - p['kOffAMPK']*y[50]   
    J70 = p['kPhosAMPK']*y[50]
    # PP1 binding to AMPKAR and dephosphorylation  
    J71 = p['kOnPP1']*y[51]*y[47] - p['kOffPP1']*y[52]
    J72 = p['kDephosPP1']*y[52]
    # Metabolic fluxes
    # glycolysis
    Jgly = 2*p['kGly']*y[1]*y[1]
    # ATP hydrolysis
    Jhydro = p['kHydro']*y[2]
    # Adenylate Kinase
    Jak = p['kForAK']*y[2]*y[0] - p['kRevAK']*y[1]*y[1] # MASS ACTION KINETICS!
    # Oxidative Phos
    Joxphos = (p['VmaxOxPhos'] *((y[1]/p['Kadp'])**p['n']))/(1 + ((y[1]/p['Kadp'])**p['n']))

    # now return the odes for each state variable
    dydt = np.zeros((53,))
    dydt[0] = -J1-J4-J7-J8-J9-J16-J17-J18-Jak
    dydt[1] = -J2-J5-J10-J11-J12-J19-J20-J21-Jgly+2*Jak+Jhydro-Joxphos
    dydt[2] = -J3-J6-J13-J14-J15-J22-J23-J24+Jgly-Jak-Jhydro+Joxphos
    dydt[3] = -J1-J2-J3-J25+J56
    dydt[4] = -J4-J5-J6+J26-J55
    dydt[5] = J1-J7-J10-J13-J27-J45
    dydt[6] = J2-J8-J11-J14-J29-J47
    dydt[7] = J3-J9-J12-J15-J31+J58
    dydt[8] = J4-J16-J19-J22+J28+J46-J65+J66
    dydt[9] = J5-J17-J20-J23+J30+J48
    dydt[10] = J6-J18-J21-J24+J32-J57
    dydt[11] = J7-J33-J49
    dydt[12] = J8+J10-J35-J51
    dydt[13] = J9+J13-J37+J60
    dydt[14] = J11-J39-J53
    dydt[15] = J12+J14-J41+J62
    dydt[16] = J15-J43+J64
    dydt[17] = J16+J34+J50-J67+J68
    dydt[18] = J17+J19+J36+J52-J69+J70
    dydt[19] = J18+J22+J38-J59
    dydt[20] = J20+J40+J54
    dydt[21] = J21+J23+J42-J61
    dydt[22] = J24+J44-J63
    dydt[23] = -J25+J26-J27+J28-J29+J30-J31+J32-J33+J34-J35+J36-J37+J38-J39+J40-J41+J42-J43+J44
    dydt[24] = J25-J26
    dydt[25] = J27-J28
    dydt[26] = J29-J30
    dydt[27] = J31-J32
    dydt[28] = J33-J34
    dydt[29] = J35-J36
    dydt[30] = J37-J38
    dydt[31] = J39-J40
    dydt[32] = J41-J42
    dydt[33] = J43-J44
    dydt[34] = -J45+J46-J47+J48-J49+J50-J51+J52-J53+J54
    dydt[35] = J45-J46
    dydt[36] = J47-J48
    dydt[37] = J49-J50
    dydt[38] = J51-J52
    dydt[39] = J53-J54
    dydt[40] = -J55+J56-J57+J58-J59+J60-J61+J62-J63+J64
    dydt[41] = J55-J56
    dydt[42] = J57-J58
    dydt[43] = J59-J60
    dydt[44] = J61-J62
    dydt[45] = J63-J64
    dydt[46] = -J65-J67-J69+J72
    dydt[47] = J66+J68+J70-J71
    dydt[48] = J65-J66
    dydt[49] = J67-J68
    dydt[50] = J69-J70
    dydt[51] =-J71+J72
    dydt[52] = J71-J72

    return dydt