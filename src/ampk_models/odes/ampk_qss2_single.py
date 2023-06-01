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
def ampk_qss2_single_get_params():
     return {
        'KdAMP':(), # AMP binding
        'kCaMKK':(), # CaMKK
        'KmCaMKK':(),
        'kLKB1':(), # LKB1 binding
        'KmLKB1':(),
        'kPP':(), # AMPK Phosphatase
        'KmPP':(),
        'kAMPK':(), # AMPK kinase
        'KmAMPK':(),
        'kPP1':(), # pAMPKAR Phosphatase
        'KmPP1':(),
        'alpha':(),
        'beta':(),
        'alpha2':(),
        'beta2':(),
        # external enzyme concentrations
        'CaMKKtot':(),
        'LKB1tot':(),
        'PPtot':(),
        'PP1tot':(),
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
def ampk_qss2_single_get_states():
    return {
        # free adenine nucleotides
        'AMP':(), 
        'ADP':(),
        'ATP':(),
        # free AMPK
        'AMPK':(),
        'pAMPK':(),
        # free AMPKAR
        'AMPKAR':(),
        'pAMPKAR':(),
    }

def ampk_qss2_single_RHS(t, y, p):
    """Right hand side of the AMPK_ma_double_mech regulation model.

    WARNING! This is different than the syntax for scipy.integrate!!
    Note from sunode syntax: "All inputs are dataclasses of sympy vars, or numpy
        arrays of sympy vars"
    """
    
    # FLUXES
    # single AXP complexing
    # substrate conc for pAMPK dephos
    S = 0.5*(((((y.AMP - y.pAMPK + p.KdAMP)**2) + 4*p.KdAMP*y.pAMPK)**0.5) - (y.AMP - y.pAMPK + p.KdAMP))

    # FLUXES
    # single AXP complexing
    J1 = (p.kCaMKK*p.CaMKKtot*y.AMPK)/(p.KmCaMKK + y.AMPK)
    J2 = (((p.kLKB1*p.LKB1tot*y.AMPK)/p.KmLKB1) + ((p.beta2*p.kLKB1*p.LKB1tot*y.AMPK*y.AMP)/(p.alpha2*p.KmLKB1*p.KdAMP)))/(1 + (y.AMPK/p.KmLKB1) + (y.AMP/p.KdAMP) + ((y.AMPK*y.AMP)/(p.alpha2*p.KmLKB1*p.KdAMP)))
    J3 = (p.kPP*p.PPtot*S)/(p.KmPP + S) 
    J4 = (((p.kAMPK*y.pAMPK*y.AMPKAR)/p.KmAMPK) + ((p.beta*p.kAMPK*y.pAMPK*y.AMPKAR*y.AMP)/(p.alpha*p.KmAMPK*p.KdAMP)))/(1 + (y.AMPKAR/p.KmAMPK) + (y.AMP/p.KdAMP) + ((y.AMPKAR*y.AMP)/(p.alpha*p.KmAMPK*p.KdAMP)))
    J5 = (p.kPP1*p.PP1tot*y.pAMPKAR)/(p.KmPP1 + y.pAMPKAR) 
    # Metabolic fluxes
    # glycolysis
    Jgly = 2*p.kGly*y.ADP*y.ADP
    # ATP hydrolysis
    Jhydro = p.kHydro*y.ATP
    # Adenylate Kinase
    JakFor = (p.kForAK*y.ATP*y.AMP) 
    JakRev = (p.kRevAK*y.ADP*y.ADP)
    # Oxidative Phos
    Joxphos = (p.VmaxOxPhos * ((y.ADP/p.Kadp)**p.n))/(1 + ((y.ADP/p.Kadp)**p.n))

    # now return the odes for each state variable
    return {
        'AMP': -JakFor + JakRev,
        'ADP': -2*Jgly + 2*JakFor - 2*JakRev + Jhydro - Joxphos,
        'ATP': 2*Jgly - JakFor + JakRev - Jhydro + Joxphos,
        'AMPK': -J1 - J2 + J3,
        'pAMPK': J1 + J2 - J3,
        'AMPKAR': -J4 + J5,
        'pAMPKAR': J4 - J5
    }

def ampk_qss2_single_RHS_sympyFluxVars():
    """Right hand side of the AMPK_MM_double_mech regulation model.

    WARNING! This is different than the syntax for scipy.integrate!!
    Note from sunode syntax: "All inputs are dataclasses of sympy vars, or numpy
        arrays of sympy vars"
    """

    # create sympy vars for FLUXES
    num_fluxes = 5
    fluxes = sym.symbols(['J'+str(i) for i in range(1,num_fluxes+1)])
    Jgly = sym.symbols("Jgly")
    Jhydro= sym.symbols("Jhydro")
    JakFor = sym.symbols("JakFor")
    JakRev = sym.symbols("JakRev")
    Joxphos = sym.symbols("Joxphos")

    for item in [Jgly, Jhydro, JakFor, JakRev, Joxphos]: # add metab fluxes to flux list
        fluxes.append(item)

    return {
        'AMP': -JakFor + JakRev,
        'ADP': -2*Jgly + 2*JakFor - 2*JakRev + Jhydro - Joxphos,
        'ATP': 2*Jgly - JakFor + JakRev - Jhydro + Joxphos,
        'AMPK': -fluxes[0] - fluxes[1] + fluxes[2],
        'pAMPK': fluxes[0] + fluxes[1] - fluxes[2],
        'AMPKAR': -fluxes[3] + fluxes[4],
        'pAMPKAR': fluxes[3] - fluxes[4]
    }, fluxes