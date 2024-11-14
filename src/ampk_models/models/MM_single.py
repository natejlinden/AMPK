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

def MM_single_sympyFluxVars():
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