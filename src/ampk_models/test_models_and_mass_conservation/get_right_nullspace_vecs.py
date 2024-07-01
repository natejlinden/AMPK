# Function that takes the right hand side of the model and returns the 
# stoichiometry matrix, a list of vectors in the left nullspace, and the 
# conservation laws in the model

import sympy as sym

def get_right_nullspace_vecs(rhs, sym_fluxes):
    """Function to find vectors in the left nullspace of a model stoichiometry matrix.

    Inputs: 
    - rhs: Dict[name, expr] of sympy expressions of the flux variables. E.g.,
        {'S1':J1-J2}
    - sym_fluxes: List of sympy expressions of the flux variables.

    Returns:
    - stoichiometry: sympy matrix with shape (num_states, num_fluxes) such that
        dXdt = stoichiometry * fluxes (X is vector of states and fluxes is vec of fluxes)
    - left_nulls: List of vectors in the left nullspace (nullspace of A^T) of
        the stoichiometry matrix
    - invariants: sympy matrices of inner product between left nullspace vectors
        and the state vector

    Nathaniel Linden (UCSD MAE) - March  2023
    Requires: sympy
    """
    # initialize states as sympy vars and store in a list
    state_names = rhs.keys()
    sym_states = []
    exprs = []
    for state in state_names:
        tmp = state.replace('_', '')
        exprs.append(rhs[state])
        exec('{i} = sym.symbols("{i}")'.format(i=tmp))
        exec('sym_states.append({i})'.format(i=tmp))

    # compute stoichiometry matrix by treating the RHS as a system of linear eqns    
    stoichiometry, _ = sym.linear_eq_to_matrix(exprs, sym_fluxes)

    # find left nullspace of stoichiometry matrix
    right_nulls = stoichiometry.nullspace()


    return stoichiometry, right_nulls