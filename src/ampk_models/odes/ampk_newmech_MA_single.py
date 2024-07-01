"""
    Nathaniel Linden (UCSD MAE)
    Created: February 17, 2023

    This file contains the functions for a model of MAPK activation. That model makes the 
    following high-level assumptions:
        - allow double adenine nucleotide AMPK binding
        - assume Michaelis-Menten kinetics for enzyme catalyzed reactions
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


# Function to create dictionary for the parameters
#   all parameters are scalars, so use key/data pairs of 'state_name': ()
def ampk_newmech_MA_single_get_params():
     return {
        'KdAMP':(), # AMP binding
        'kCaMKK':(), # CaMKK
        'konCaMKK': (), # CaMKK binding
        'koffCaMKK': (),
        'kCaMKK': (), # phosphorylation
        'konLKB1': (), # LKB1 binding
        'koffLKB1': (),
        'kLKB1': (), # phosphorylation
        'konPP': (), # Phosphatase AMPK binding
        'koffPP': (),
        'kPP': (), # dephosphorylation
        'konAMPK': (), # pAMPK binds AMPKAR
        'koffAMPK': (),
        'kAMPK': (), # pAMPK phosphorylates AMPKAR
        'konPP1': (), # Phosphatase AMPKAR binding
        'koffPP1': (), 
        'kPP1': (),
        'alpha':(),
        'beta':(),
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
def ampk_newmech_MA_single_get_states():
    return {
        'AMP':(),
        'ADP':(),
        'ATP':(),
        'AMPK':(),
        'pAMPK':(),
        'AMP_AMPK':(),
        'AMP_pAMPK':(),
        'CaMKK':(),
        'CaMKK_AMPK':(),
        'CaMKK_AMP_AMPK':(),
        'LKB1':(),
        'LKB1_AMPK':(),
        'LKB1_AMP_AMPK':(),
        'PP':(),
        'PP_pAMPK':(),
        'AMPKAR':(),
        'pAMPKAR':(),
        'AMPKAR_pAMPK':(),
        'AMPKAR_AMP_pAMPK':(),
        'PP1':(),
        'PP1_pAMPKAR':(),
    }

        

def ampk_newmech_MA_single_RHS(t, y, p):
    """Right hand side of the AMPK_ma_double_mech regulation model.

    WARNING! This is different than the syntax for scipy.integrate!!
    Note from sunode syntax: "All inputs are dataclasses of sympy vars, or numpy
        arrays of sympy vars"
    """
    konAMP = 1.0
    
    # FLUXES
    J1  = konAMP*y.AMP*y.AMPK  - p.KdAMP*y.AMP_AMPK 
    J2  = konAMP*y.AMP*y.pAMPK  - p.KdAMP*y.AMP_pAMPK 
    J3  = konAMP*y.AMP*y.AMPKAR_pAMPK  - p.KdAMP*y.AMPKAR_AMP_pAMPK
    J4  = p.konCaMKK*y.AMPK*y.CaMKK - p.koffCaMKK*y.CaMKK_AMPK 
    J5  = p.kCaMKK*y.CaMKK_AMPK 
    J6  = p.konCaMKK*y.AMP_AMPK*y.CaMKK - p.koffCaMKK*y.CaMKK_AMP_AMPK 
    J7  = p.kCaMKK*y.CaMKK_AMP_AMPK 
    J8  = p.konLKB1*y.AMPK*y.LKB1 - p.koffLKB1*y.LKB1_AMPK 
    J9  = p.kLKB1*y.LKB1_AMPK 
    J10 = p.konLKB1*y.AMP_AMPK*y.LKB1 - p.koffLKB1*y.LKB1_AMP_AMPK 
    J11 = p.kLKB1*y.LKB1_AMP_AMPK 
    J12 = p.konPP*y.PP*y.pAMPK - p.koffPP*y.PP_pAMPK 
    J13 = p.kLKB1*y.PP_pAMPK 
    J14 = p.konAMPK*y.pAMPK*y.AMPKAR - p.koffAMPK*y.AMPKAR_pAMPK 
    J15 = p.kAMPK*y.AMPKAR_pAMPK 
    J16 = p.konAMPK*y.AMP_pAMPK*y.AMPKAR - p.alpha*p.koffAMPK*y.AMPKAR_AMP_pAMPK 
    J17 = p.beta*p.kAMPK*y.AMPKAR_AMP_pAMPK 
    J18 = p.konPP1*y.PP1*y.pAMPKAR - p.koffPP1*y.PP1_pAMPKAR 
    J19 = p.kPP1*y.PP1_pAMPKAR
    
    # Metabolic fluxes
    # glycolysis
    Jgly = 2*p.kGly*y.ADP*y.ADP
    # ATP hydrolysis
    Jhydro = p.kHydro*y.ATP
    # Adenylate Kinase
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
        'AMP':-J1 - J2 - J3 -JAK,
        'ADP':-Jgly+2*JAK+Jhydro-Joxphos,
        'ATP':Jgly-JAK-Jhydro+Joxphos,
        'AMPK':-J1 - J4 - J8 + J13,
        'pAMPK':-J2 + J5 + J9 - J12 - J14 + J15,
        'AMP_AMPK':J1 - J6 - J10,
        'AMP_pAMPK':J2 + J7 + J11 - J16 + J17,
        'CaMKK':-J4 - J6 + J5 + J7,
        'CaMKK_AMPK':J4 - J5,
        'CaMKK_AMP_AMPK':J6 - J7,
        'LKB1':-J8 -J10 + J9 + J11,
        'LKB1_AMPK':J8 - J9,
        'LKB1_AMP_AMPK':J10 - J11,
        'PP':-J12 + J13,
        'PP_pAMPK':J12 - J13,
        'AMPKAR':-J14 - J16 + J19,
        'pAMPKAR':J15 + J17 - J18,
        'AMPKAR_pAMPK':-J3 + J14 - J15,
        'AMPKAR_AMP_pAMPK':J3 + J16 - J17,
        'PP1':-J18 + J19,
        'PP1_pAMPKAR':J18 - J19,
    }

def ampk_newmech_MA_single_RHS_sympyFluxVars():
    """Right hand side of the AMPK_MM_double_mech regulation model.

    WARNING! This is different than the syntax for scipy.integrate!!
    Note from sunode syntax: "All inputs are dataclasses of sympy vars, or numpy
        arrays of sympy vars"
    """

    # create sympy vars for FLUXES
    num_fluxes = 19
    fluxes = sym.symbols(['J'+str(i) for i in range(1,num_fluxes+1)])
    Jgly = sym.symbols("Jgly")
    Jhydro= sym.symbols("Jhydro")
    JAK = sym.symbols("JAK")
    Joxphos = sym.symbols("Joxphos")

    for item in [Jgly, Jhydro, JAK, Joxphos]: # add metab fluxes to flux list
        fluxes.append(item)

    return {
        'AMP':-fluxes[0] - fluxes[1] - fluxes[2] -JAK,
        'ADP':-2*Jgly+2*JAK+Jhydro-Joxphos,
        'ATP':2*Jgly-JAK-Jhydro+Joxphos,
        'AMPK':-fluxes[0] - fluxes[3] - fluxes[7] + fluxes[12],
        'pAMPK':-fluxes[1] + fluxes[4] + fluxes[8] - fluxes[11] - fluxes[13] + fluxes[14],
        'AMP_AMPK':fluxes[0] - fluxes[5] - fluxes[9],
        'AMP_pAMPK':fluxes[1] + fluxes[6] + fluxes[10] - fluxes[15] + fluxes[16],
        'CaMKK':-fluxes[3] - fluxes[5] + fluxes[4] + fluxes[6],
        'CaMKK_AMPK':fluxes[3] - fluxes[4],
        'CaMKK_AMP_AMPK':fluxes[5] - fluxes[6],
        'LKB1':-fluxes[7] -fluxes[9] + fluxes[8] + fluxes[10],
        'LKB1_AMPK':fluxes[7] - fluxes[8],
        'LKB1_AMP_AMPK':fluxes[9] - fluxes[10],
        'PP':-fluxes[11] + fluxes[12],
        'PP_pAMPK':fluxes[11] - fluxes[12],
        'AMPKAR':-fluxes[13] - fluxes[15] + fluxes[18],
        'pAMPKAR':fluxes[14] + fluxes[16] - fluxes[17],
        'AMPKAR_pAMPK':-fluxes[2] + fluxes[13] - fluxes[14],
        'AMPKAR_AMP_pAMPK':fluxes[2] + fluxes[15] - fluxes[16],
        'PP1':-fluxes[17] + fluxes[18],
        'PP1_pAMPKAR':fluxes[17] - fluxes[18],
    }, fluxes