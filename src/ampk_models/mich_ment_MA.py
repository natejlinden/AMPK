import sympy as sym
import numpy as np

# Function to create dictionary for the parameters
#   all parameters are scalars, so use key/data pairs of 'state_name': ()
def mich_ment_get_params():
     return {
        'kOn':(),
        'kOff':(),
        'kCat':(), 
        # 'Et':(),
        # 'Vmax': (), # AMP binding
        # 'Km': (),
    }

# Function to create dictionary for the states
#   all states, so use key/data pairs of 'state_name': ()
def mich_ment_get_states():
    return {
        # free adenine nucleotides
       'S':(),
       'E':(),
       'ES':(),
       'P':(),
    }
def mich_ment_RHS_odeEnzyme(t, y, p):
    """Right hand side of the AMPK_ma_double_mech regulation model.

    WARNING! This is different than the syntax for scipy.integrate!!
    Note from sunode syntax: "All inputs are dataclasses of sympy vars, or numpy
        arrays of sympy vars"
    """
    
    # FLUXES
    J1 = p.kOn*y.S*y.E - p.kOff*y.ES
    J2 = p.kCat*y.ES

    # now return the odes for each state variable
    return {
        'S': -J1,
        'E': -J1+J2,
        'ES': J1-J2,
        'P': J2
    }

def mich_ment_RHS_algEnzyme(t, y, p):
    """Right hand side of the AMPK_ma_double_mech regulation model.

    WARNING! This is different than the syntax for scipy.integrate!!
    Note from sunode syntax: "All inputs are dataclasses of sympy vars, or numpy
        arrays of sympy vars"
    """
    
    # FLUXES
    E = p.Et - y.ES
    J1 = p.kOn*y.S*E - p.kOff*y.ES
    J2 = p.kCat*y.ES

    # now return the odes for each state variable
    return {
        'S': -J1,
        'ES': J1-J2,
        'P': J2
    }

def mich_ment_RHS_odeEnzyme_sympyFluxVars():
    """Right hand side of the AMPK_ma_double_mech regulation model.

    WARNING! This is different than the syntax for scipy.integrate!!
    Note from sunode syntax: "All inputs are dataclasses of sympy vars, or numpy
        arrays of sympy vars"
    """

    # create sympy vars for FLUXES
    J1 = sym.symbols("J1")
    J2 = sym.symbols("J2")

    return {
        'S': -J1,
        'E': -J1+J2,
        'ES': J1-J2,
        'P': J2
    }, [J1, J2]

def mich_ment_RHS_algEnzyme_sympyFluxVars():
    """Right hand side of the AMPK_ma_double_mech regulation model.

    WARNING! This is different than the syntax for scipy.integrate!!
    Note from sunode syntax: "All inputs are dataclasses of sympy vars, or numpy
        arrays of sympy vars"
    """

    # create sympy vars for FLUXES
    J1 = sym.symbols("J1")
    J2 = sym.symbols("J2")

    return {
        'S': -J1,
        'ES': J1-J2,
        'P': J2
    }, [J1, J2]
   