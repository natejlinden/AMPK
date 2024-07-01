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
def ampk_MM_single_mech_get_params():
     return {
        'kOnAMP': (), # AMP binding
        'kOffAMP': (),
        'kOnADP': (), # ADP binding
        'kOffADP': (),
        'kOnATP': (), # ATP binding
        'kOffATP': (),
        'kCaMKK': (), # CaMKK
        'KmCaMKK': (),
        'kLKB1': (), # LKB1 binding
        'KmLKB1': (),
        'kPP': (), # AMPK Phosphatase
        'KmPP': (),
        'kAMPK': (), # AMPK kinase
        'KmAMPK': (),
        'kPP1': (), # pAMPKAR Phosphatase
        'KmPP1': (), 
        # external enzyme concentrations
        'CaMKKtot': (),
        'LKB1tot':(),
        'PPtot':(),
        'PP1tot':(),
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
def ampk_MM_single_mech_get_states():
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
        # free AMPKAR
        'AMPKAR':(),
        'pAMPKAR':(),
    }

def ampk_MM_single_mech_RHS(t, y, p):
    """Right hand side of the AMPK_ma_double_mech regulation model.

    WARNING! This is different than the syntax for scipy.integrate!!
    Note from sunode syntax: "All inputs are dataclasses of sympy vars, or numpy
        arrays of sympy vars"
    """
    
    # FLUXES
    # single AXP complexing
    J1 = p.kOnAMP*y.AMP*y.AMPK - p.kOffAMP*y.AMP_AMPK# AMPK
    J2 = p.kOnADP*y.ADP*y.AMPK - p.kOffADP*y.ADP_AMPK
    J3 = p.kOnATP*y.ATP*y.AMPK - p.kOffATP*y.ATP_AMPK
    J4 = p.kOnAMP*y.AMP*y.pAMPK - p.kOffAMP*y.AMP_pAMPK# pAMPK
    J5 = p.kOnADP*y.ADP*y.pAMPK - p.kOffADP*y.ADP_pAMPK
    J6 = p.kOnATP*y.ATP*y.pAMPK - p.kOffATP*y.ATP_pAMPK
    J7 = (p.kCaMKK*p.CaMKKtot*y.AMPK)/(p.KmCaMKK + y.AMPK) # CaMKK phosphorylation
    J8 = (p.kCaMKK*p.CaMKKtot*y.AMP_AMPK)/(p.KmCaMKK + y.AMP_AMPK)
    J9 = (p.kCaMKK*p.CaMKKtot*y.ADP_AMPK)/(p.KmCaMKK + y.ADP_AMPK)
    J10 = (p.kCaMKK*p.CaMKKtot*y.ATP_AMPK)/(p.KmCaMKK + y.ATP_AMPK)
    J11 = (p.kLKB1*p.LKB1tot*y.AMP_AMPK)/(p.KmLKB1 + y.AMP_AMPK)
    J12 = (p.kLKB1*p.LKB1tot*y.ADP_AMPK)/(p.KmLKB1 + y.ADP_AMPK)
    J13 = (p.kPP*p.PPtot*y.pAMPK)/(p.KmPP + y.pAMPK)
    J14 = (p.kPP*p.PPtot*y.ATP_pAMPK)/(p.KmPP + y.ATP_pAMPK)
    J15 = (p.kAMPK*y.AMP_pAMPK*y.AMPKAR)/(p.KmAMPK + y.AMPKAR)
    J16 = (p.kPP1*p.PP1tot*y.pAMPKAR)/(p.KmPP1 + y.pAMPKAR)
    # Metabolic fluxes
    # glycolysis
    Jgly = 2*p.kGly*y.ADP*y.ADP
    # ATP hydrolysis
    Jhydro = p.kHydro*y.ATP
    # Adenylate Kinase from Lambeth and Kushmerick 2002
    num_for = (p.VforAK*y.ATP*y.AMP)/(p.kmt*p.kmm)
    den = (1 + (y.ATP/p.kmt) + (y.AMP/p.kmm) + ((y.ATP*y.AMP)/(p.kmt*p.kmm)) + 
                ((2*y.ADxP)/p.kmd) + ((y.ADP**2)/(p.kmd**2)))
    VrevAK = (p.VforAK*(p.kmd**2))/(p.KeqAK*p.kmt*p.kmm)
    num_rev = (VrevAK*(y.ADP**2))/(p.kmd**2)
    JAK = (num_for - num_rev)/den
    # Oxidative Phos
    Joxphos = (p.VmaxOxPhos * ((y.ADP/p.Kadp)**p.n))/(1 + ((y.ADP/p.Kadp)**p.n))

    # now return the odes for each state variable
    return {
        'AMP': -J1-J4-JAK,
        'ADP': -J2-J5-Jgly+2*JAK+Jhydro-Joxphos,
        'ATP': -J3-J6+Jgly-JAK-Jhydro+Joxphos,
        # free AMPK
        'AMPK': -J1-J2-J3-J7+J13,
        'pAMPK': -J4-J5-J6+J7-J13,
        # single AXP-AMPK complexes
        'AMP_AMPK': J1-J8-J11,
        'ADP_AMPK': J2-J9-J12,
        'ATP_AMPK': J3-J10+J14,
        # single AXP-pAMPK complexes
        'AMP_pAMPK': J4+J8+J11,
        'ADP_pAMPK': J5+J9+J12,
        'ATP_pAMPK': J6+J10-J14,
        # AMPKAR
        'AMPKAR': -J15+J16,
        'pAMPKAR': J15-J16,
    }

def ampk_MM_single_mech_RHS_sympyFluxVars():
    """Right hand side of the AMPK_MM_double_mech regulation model.

    WARNING! This is different than the syntax for scipy.integrate!!
    Note from sunode syntax: "All inputs are dataclasses of sympy vars, or numpy
        arrays of sympy vars"
    """

    # create sympy vars for FLUXES
    num_fluxes = 16
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
        'AMPK': -fluxes[0]-fluxes[1]-fluxes[2]-fluxes[6]+fluxes[12],
        'pAMPK': -fluxes[3]-fluxes[4]-fluxes[5]+fluxes[6]-fluxes[12],
        # single AXP-AMPK complexes
        'AMP_AMPK': fluxes[0]-fluxes[7]-fluxes[10],
        'ADP_AMPK': fluxes[1]-fluxes[8]-fluxes[11],
        'ATP_AMPK': fluxes[2]-fluxes[9]+fluxes[13],
        # single AXP-pAMPK complexes
        'AMP_pAMPK': fluxes[3]+fluxes[7]+fluxes[10],
        'ADP_pAMPK': fluxes[4]+fluxes[8]+fluxes[11],
        'ATP_pAMPK': fluxes[5]+fluxes[9]-fluxes[13],
        # AMPKAR
        'AMPKAR': -fluxes[14]+fluxes[15],
        'pAMPKAR': fluxes[14]-fluxes[15],
    }, fluxes
    


# # Code for replacing flux terms with iterable indexes
# for i in reversed(range(17)):
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