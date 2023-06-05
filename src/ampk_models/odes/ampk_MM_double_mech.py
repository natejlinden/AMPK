"""
    Nathaniel Linden (UCSD MAE)
    Created: February 17, 2023

    This file contains the functions for a model of MAPK activation. That model mAKes the 
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
def ampk_MM_double_mech_get_params():
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
def ampk_MM_double_mech_get_states():
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
        # free AMPKAR
        'AMPKAR':(),
        'pAMPKAR':(),
    }

def ampk_MM_double_mech_RHS(t, y, p):
    """Right hand side of the AMPK_ma_double_mech regulation model.

    WARNING! This is different than the syntax for scipy.integrate!!
    Note from sunode syntax: "All inputs are dataclasses of sympy vars, or numpy
        arrays of sympy vars"
    """
    
    # FLUXES
    # single AXP complexing
    J1 = p.kOnAMP*y.AMP*y.AMPK - p.kOffAMP*y.AMP_AMPK # AMPK
    J2 = p.kOnADP*y.ADP*y.AMPK - p.kOffADP*y.ADP_AMPK
    J3 = p.kOnATP*y.ATP*y.AMPK - p.kOffATP*y.ATP_AMPK
    J4 = p.kOnAMP*y.AMP*y.pAMPK  - p.kOffAMP*y.AMP_pAMPK# pAMPK
    J5 = p.kOnADP*y.ADP*y.pAMPK - p.kOffADP*y.ADP_pAMPK
    J6 = p.kOnATP*y.ATP*y.pAMPK - p.kOffATP*y.ATP_pAMPK
    # double AXP complexing
    J7 = p.kOnAMP*y.AMP*y.AMP_AMPK - p.kOffAMP*y.AMP_AMP_AMPK# AMPK
    J8 = p.kOnAMP*y.AMP*y.ADP_AMPK - p.kOffAMP*y.AMP_ADP_AMPK
    J9 = p.kOnAMP*y.AMP*y.ATP_AMPK - p.kOffAMP*y.AMP_ATP_AMPK
    J10 = p.kOnADP*y.ADP*y.AMP_AMPK - p.kOffADP*y.AMP_ADP_AMPK
    J11 = p.kOnADP*y.ADP*y.ADP_AMPK - p.kOffADP*y.ADP_ADP_AMPK
    J12 = p.kOnADP*y.ADP*y.ATP_AMPK - p.kOffADP*y.ADP_ATP_AMPK
    J13 = p.kOnATP*y.ATP*y.AMP_AMPK - p.kOffATP*y.AMP_ATP_AMPK
    J14 = p.kOnATP*y.ATP*y.ADP_AMPK -  p.kOffATP*y.ADP_ATP_AMPK
    J15 = p.kOnATP*y.ATP*y.ATP_AMPK - p.kOffATP*y.ATP_ATP_AMPK
    J16 = p.kOnAMP*y.AMP*y.AMP_pAMPK -  p.kOffAMP*y.AMP_AMP_pAMPK # pAMPK
    J17 = p.kOnAMP*y.AMP*y.ADP_pAMPK -  p.kOffAMP*y.AMP_ADP_pAMPK
    J18 = p.kOnAMP*y.AMP*y.ATP_pAMPK - p.kOffAMP*y.AMP_ATP_pAMPK
    J19 = p.kOnADP*y.ADP*y.AMP_pAMPK - p.kOffADP*y.AMP_ADP_pAMPK
    J20 = p.kOnADP*y.ADP*y.ADP_pAMPK -  p.kOffADP*y.ADP_ADP_pAMPK
    J21 = p.kOnADP*y.ADP*y.ATP_pAMPK -  p.kOffADP*y.ADP_ATP_pAMPK
    J22 = p.kOnATP*y.ATP*y.AMP_pAMPK - p.kOffATP*y.AMP_ATP_pAMPK
    J23 = p.kOnATP*y.ATP*y.ADP_pAMPK - p.kOffATP*y.ADP_ATP_pAMPK
    J24 = p.kOnATP*y.ATP*y.ATP_pAMPK - p.kOffATP*y.ATP_ATP_pAMPK
    J25 = (p.kCaMKK*p.CaMKKtot*y.AMPK)/(p.KmCaMKK + y.AMPK) # CaMKK phosphorylation
    J26 = (p.kCaMKK*p.CaMKKtot*y.AMP_AMPK)/(p.KmCaMKK + y.AMP_AMPK)
    J27 = (p.kCaMKK*p.CaMKKtot*y.ADP_AMPK)/(p.KmCaMKK + y.ADP_AMPK)
    J28 = (p.kCaMKK*p.CaMKKtot*y.ATP_AMPK)/(p.KmCaMKK + y.ATP_AMPK)
    J29 = (p.kCaMKK*p.CaMKKtot*y.AMP_AMP_AMPK)/(p.KmCaMKK + y.AMP_AMP_AMPK)
    J30 = (p.kCaMKK*p.CaMKKtot*y.AMP_ADP_AMPK)/(p.KmCaMKK + y.AMP_ADP_AMPK)
    J31 = (p.kCaMKK*p.CaMKKtot*y.AMP_ATP_AMPK)/(p.KmCaMKK + y.AMP_ATP_AMPK)
    J32 = (p.kCaMKK*p.CaMKKtot*y.ADP_ADP_AMPK)/(p.KmCaMKK + y.ADP_ADP_AMPK)
    J33 = (p.kCaMKK*p.CaMKKtot*y.ADP_ATP_AMPK)/(p.KmCaMKK + y.ADP_ATP_AMPK)
    J34 = (p.kCaMKK*p.CaMKKtot*y.ATP_ATP_AMPK)/(p.KmCaMKK + y.ATP_ATP_AMPK)
    J35 = (p.kLKB1*p.LKB1tot*y.AMP_AMPK)/(p.KmLKB1 + y.AMP_AMPK)
    J36 = (p.kLKB1*p.LKB1tot*y.ADP_AMPK)/(p.KmLKB1 + y.ADP_AMPK)
    J37 = (p.kLKB1*p.LKB1tot*y.AMP_AMP_AMPK)/(p.KmLKB1 + y.AMP_AMP_AMPK)
    J38 = (p.kLKB1*p.LKB1tot*y.AMP_ADP_AMPK)/(p.KmLKB1 + y.AMP_ADP_AMPK)
    J39 = (p.kLKB1*p.LKB1tot*y.ADP_ADP_AMPK)/(p.KmLKB1 + y.ADP_ADP_AMPK)
    J40 = (p.kPP*p.PPtot*y.pAMPK)/(p.KmPP + y.pAMPK)
    J41 = (p.kPP*p.PPtot*y.ATP_pAMPK)/(p.KmPP + y.ATP_pAMPK)
    J42 = (p.kPP*p.PPtot*y.ATP_ATP_pAMPK)/(p.KmPP + y.ATP_ATP_pAMPK)
    J43 = (p.kPP*p.PPtot*y.AMP_ATP_pAMPK)/(p.KmPP + y.AMP_ATP_pAMPK)
    J44 = (p.kPP*p.PPtot*y.ADP_ATP_pAMPK)/(p.KmPP + y.ADP_ATP_pAMPK)
    J45 = (p.kAMPK*y.AMP_pAMPK*y.AMPKAR)/(p.KmAMPK + y.AMPKAR)
    J46 = (p.kAMPK*y.AMP_AMP_pAMPK*y.AMPKAR)/(p.KmAMPK + y.AMPKAR)
    J47 = (p.kAMPK*y.AMP_ADP_pAMPK*y.AMPKAR)/(p.KmAMPK + y.AMPKAR)
    J48 = (p.kPP1*p.PP1tot*y.pAMPKAR)/(p.KmPP1 + y.pAMPKAR)
    # Metabolic fluxes
    # glycolysis
    Jgly = 2*p.kGly*y.ADP*y.ADP
    # ATP hydrolysis
    Jhydro = p.kHydro*y.ATP
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
        'AMPK': -J1-J2-J3-J25+J40,
        'pAMPK': -J4-J5-J6+J25-J40,
        # single AXP-AMPK complexes
        'AMP_AMPK': J1-J7-J10-J13-J26-J35,
        'ADP_AMPK': J2-J8-J11-J14-J27-J36,
        'ATP_AMPK': J3-J9-J12-J15-J28+J41,
        # single AXP-pAMPK complexes
        'AMP_pAMPK': J4-J16-J19-J22+J26+J35,
        'ADP_pAMPK': J5-J17-J20-J23+J27+J36,
        'ATP_pAMPK': J6-J18-J21-J24+J28-J41,
        # double AXP-AMPK complexes
        'AMP_AMP_AMPK': J7-J29-J37,
        'AMP_ADP_AMPK': J8+J10-J30-J38,
        'AMP_ATP_AMPK': J9+J13-J31+J43,
        'ADP_ADP_AMPK': J11-J32-J39,
        'ADP_ATP_AMPK': J12+J14-J33+J44,
        'ATP_ATP_AMPK': J15-J34+J42,
        # double AXP-pAMPK complexes
        'AMP_AMP_pAMPK': J16+J29+J37, 
        'AMP_ADP_pAMPK': J17+J19+J30+J38, 
        'AMP_ATP_pAMPK': J18+J22+J31-J43,
        'ADP_ADP_pAMPK': J20+J32+J39,
        'ADP_ATP_pAMPK': J21+J23+J33-J44,
        'ATP_ATP_pAMPK': J24+J34-J42,
        # AMPKAR
        'AMPKAR': -J45-J46-J47+J48,
        'pAMPKAR': J45+J46+J47-J48,
    }

def ampk_MM_double_mech_RHS_sympyFluxVars():
    """Right hand side of the AMPK_MM_double_mech regulation model.

    WARNING! This is different than the syntax for scipy.integrate!!
    Note from sunode syntax: "All inputs are dataclasses of sympy vars, or numpy
        arrays of sympy vars"
    """

    # create sympy vars for FLUXES
    num_fluxes = 48
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
        'AMPK': -fluxes[0]-fluxes[1]-fluxes[2]-fluxes[24]+fluxes[39],
        'pAMPK': -fluxes[3]-fluxes[4]-fluxes[5]+fluxes[24]-fluxes[39],
        'AMP_AMPK': fluxes[0]-fluxes[6]-fluxes[9]-fluxes[12]-fluxes[25]-fluxes[34],
        'ADP_AMPK': fluxes[1]-fluxes[7]-fluxes[10]-fluxes[13]-fluxes[26]-fluxes[35],
        'ATP_AMPK': fluxes[2]-fluxes[8]-fluxes[11]-fluxes[14]-fluxes[27]+fluxes[40],
        'AMP_pAMPK': fluxes[3]-fluxes[15]-fluxes[18]-fluxes[21]+fluxes[25]+fluxes[34],
        'ADP_pAMPK': fluxes[4]-fluxes[16]-fluxes[19]-fluxes[22]+fluxes[26]+fluxes[35],
        'ATP_pAMPK': fluxes[5]-fluxes[17]-fluxes[20]-fluxes[23]+fluxes[27]-fluxes[40],
        'AMP_AMP_AMPK': fluxes[6]-fluxes[28]-fluxes[36],
        'AMP_ADP_AMPK': fluxes[7]+fluxes[9]-fluxes[29]-fluxes[37],
        'AMP_ATP_AMPK': fluxes[8]+fluxes[12]-fluxes[30]+fluxes[42],
        'ADP_ADP_AMPK': fluxes[10]-fluxes[31]-fluxes[38],
        'ADP_ATP_AMPK': fluxes[11]+fluxes[13]-fluxes[32]+fluxes[43],
        'ATP_ATP_AMPK': fluxes[14]-fluxes[33]+fluxes[41],
        'AMP_AMP_pAMPK': fluxes[15]+fluxes[28]+fluxes[36], 
        'AMP_ADP_pAMPK': fluxes[16]+fluxes[18]+fluxes[29]+fluxes[37], 
        'AMP_ATP_pAMPK': fluxes[17]+fluxes[21]+fluxes[30]-fluxes[42],
        'ADP_ADP_pAMPK': fluxes[19]+fluxes[31]+fluxes[38],
        'ADP_ATP_pAMPK': fluxes[20]+fluxes[22]+fluxes[32]-fluxes[43],
        'ATP_ATP_pAMPK': fluxes[23]+fluxes[33]-fluxes[41],
        'AMPKAR': -fluxes[44]-fluxes[45]-fluxes[46]+fluxes[47],
        'pAMPKAR': fluxes[44]+fluxes[45]+fluxes[46]-fluxes[47],
    }, fluxes




# # Code for replacing flux terms with iterable indexes
# for i in reversed(range(49)):
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