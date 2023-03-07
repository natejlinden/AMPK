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
        # # total enzyme concentrations
        # 'CaMKKtot': (),
        # 'LKB1tot':(),
        # 'PPtot':(),
        # 'PP1tot':(),
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
    add = np.vectorize(sym.Add)
    mul = np.vectorize(sym.Mul)

    # compute enzyme concentrations
    # CaMKK2 = add(p.CaMKKtot, -y.CaMKK_AMPK, -y.CaMKK_AMP_AMPK, -y.CaMKK_ADP_AMPK,
    #     -y.CaMKK_ATP_AMPK, -y.CaMKK_AMP_AMP_AMPK, -y.CaMKK_AMP_ADP_AMPK,
    #     -y.CaMKK_AMP_ATP_AMPK, -y.CaMKK_ADP_ADP_AMPK, -y.CaMKK_ADP_ATP_AMPK,
    #     -y.CaMKK_ATP_ATP_AMPK)
    # LKB1 = add(p.LKB1tot, -y.LKB1_AMP_AMPK, -y.LKB1_ADP_AMPK, -y.LKB1_AMP_AMP_AMPK,
    #     -y.LKB1_AMP_ADP_AMPK, -y.LKB1_ADP_ADP_AMPK)
    # PP = add(p.PPtot, -y.PP_pAMPK, -y.PP_ATP_pAMPK, -y.PP_AMP_ATP_pAMPK,
    #     -y.PP_ADP_ATP_pAMPK, -y.PP_ATP_ATP_pAMPK)
    # PP1 = add(p.PP1tot, -y.PP1_pAMPKAR)
    
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
    Jak = mul(p.kForAK, y.ATP, y.AMP) - mul(p.kRevAK, y.ADP, y.ADP) # MASS ACTION KINETICS!
    # Oxidative Phos
    Joxphos = (p.VmaxOxPhos * ((y.ADP/p.Kadp)**p.n))/(1 + ((y.ADP/p.Kadp)**p.n))

    # now return the odes for each state variable
    return {
        'AMP': -J1-J4-J7-J8-J9-J16-J17-J18-Jak,
        'ADP': -J2-J5-J10-J11-J12-J19-J20-J21-Jgly+2*Jak+Jhydro-Joxphos,
        'ATP': -J3-J6-J13-J14-J15-J22-J23-J24+Jgly-Jak-Jhydro+Joxphos,
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
        'PP1':-J71+J72,
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
    Jak = sym.symbols("Jak")
    Joxphos = sym.symbols("Joxphos")

    for item in [Jgly, Jhydro, Jak, Joxphos]: # add metab fluxes to flux list
        fluxes.append(item)

    
    return {
        'AMP': -fluxes[0]-fluxes[3]-fluxes[6]-fluxes[7]-fluxes[8]-fluxes[15]-fluxes[16]-fluxes[17]-Jak,
        'ADP': -fluxes[1]-fluxes[4]-fluxes[9]-fluxes[10]-fluxes[11]-fluxes[18]-fluxes[19]-fluxes[20]-Jgly+2*Jak+Jhydro-Joxphos,
        'ATP': -fluxes[2]-fluxes[5]-fluxes[12]-fluxes[13]-fluxes[14]-fluxes[21]-fluxes[22]-fluxes[23]+Jgly-Jak-Jhydro+Joxphos,
        # free AMPK
        'AMPK': -fluxes[0]-fluxes[1]-fluxes[2]-fluxes[24]+fluxes[55],
        'pAMPK': -fluxes[3]-fluxes[4]-fluxes[5]-fluxes[25]-fluxes[54],
        # single AXP-AMPK complexes
        'AMP_AMPK': fluxes[0]-fluxes[6]-fluxes[9]-fluxes[12]-fluxes[26]-fluxes[44],
        'ADP_AMPK': fluxes[1]-fluxes[7]-fluxes[10]-fluxes[13]-fluxes[28]-fluxes[46],
        'ATP_AMPK': fluxes[2]-fluxes[8]-fluxes[11]-fluxes[14]-fluxes[30]+fluxes[57],
        # single AXP-pAMPK complexes
        'AMP_pAMPK': fluxes[3]-fluxes[15]-fluxes[18]-fluxes[21]+fluxes[27]+fluxes[45]-fluxes[64]+fluxes[65],
        'ADP_pAMPK': fluxes[4]-fluxes[16]-fluxes[19]-fluxes[22]+fluxes[29]+fluxes[47],
        'ATP_pAMPK': fluxes[5]-fluxes[17]-fluxes[20]-fluxes[23]-fluxes[56]+fluxes[31],
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