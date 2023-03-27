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
def metab_get_params():
     return {
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

def metab_get_params_cocciAK():
     return {
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
def metab_get_states():
    return {
        # free adenine nucleotides
        'AMP':(), 
        'ADP':(),
        'ATP':(),
    }

def metab_RHS(t, y, p):
    """Right hand side of the AMPK_ma_double_mech regulation model.

    WARNING! This is different than the syntax for scipy.integrate!!
    Note from sunode syntax: "All inputs are dataclasses of sympy vars, or numpy
        arrays of sympy vars"
    """
    
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
        'AMP': -Jak,
        'ADP': -Jgly+2*Jak+Jhydro-Joxphos,
        'ATP': Jgly-Jak-Jhydro+Joxphos,
    }

def metab_RHS_cocciAK(t, y, p):
    """Right hand side of the AMPK_ma_double_mech regulation model.

    WARNING! This is different than the syntax for scipy.integrate!!
    Note from sunode syntax: "All inputs are dataclasses of sympy vars, or numpy
        arrays of sympy vars"
    """
    
    # Metabolic fluxes
    # glycolysis
    print(p.kGly)
    Jgly = 2*p.kGly*y.ADP*y.ADP
    # ATP hydrolysis
    Jhydro = p.kHydro*y.ATP
    # Adenylate Kinase
    JAKfor = (((p.VforAK*y.ATP*y.AMP)/(p.kmt*p.kmm))/
             (1 + (y.ATP/p.kmt) + (y.AMP/p.kmm) + ((y.ATP*y.AMP)/(p.kmt*p.kmm)) + (2*y.ADP/p.kmd) + ((y.ADP/p.kmd)**2)))
    VrevAK = (p.VforAK*p.kmd*p.kmd)/(p.KeqAK*p.kmt*p.kmm)
    JAKrev = (((VrevAK*y.ADP*y.ADP)/(p.kmd*p.kmd))/
             (1 + (y.ATP/p.kmt) + (y.AMP/p.kmm) + ((y.ATP*y.AMP)/(p.kmt*p.kmm)) + (2*y.ADP/p.kmd) + ((y.ADP/p.kmd)**2)))
    # Oxidative Phos
    Joxphos = (p.VmaxOxPhos * ((y.ADP/p.Kadp)**p.n))/(1 + ((y.ADP/p.Kadp)**p.n))

    # now return the odes for each state variable
    return {
        'AMP': -JAKfor+JAKrev,
        'ADP': -Jgly+Jhydro-Joxphos+2*JAKfor-2*JAKrev,
        'ATP': Jgly-Jhydro+Joxphos-JAKfor+JAKrev,
    }

def metab_RHS_sympyFluxVars():
    """Right hand side of the AMPK_MM_double_mech regulation model.

    WARNING! This is different than the syntax for scipy.integrate!!
    Note from sunode syntax: "All inputs are dataclasses of sympy vars, or numpy
        arrays of sympy vars"
    """

    # create sympy vars for FLUXES
    Jgly = sym.symbols("Jgly")
    Jhydro= sym.symbols("Jhydro")
    Jak = sym.symbols("Jak")
    Joxphos = sym.symbols("Joxphos")

    fluxes = [Jgly, Jhydro, Jak, Joxphos]

    return {
        'AMP': -Jak,
        'ADP': -Jgly+2*Jak+Jhydro-Joxphos,
        'ATP': Jgly-Jak-Jhydro+Joxphos,
    }, fluxes