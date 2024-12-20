"""
    Nathaniel Linden (UCSD MAE)
    Created: November 26th, 2024
"""
import jax.numpy as jnp
import equinox as eqx

class MA(eqx.Module):
    def __call__(self, t, y, args):
        
        # unpack parameters
        k_f_1     = args[0]
        k_r_1     = args[1]
        k_cat_1   = args[2]
        Vmax_2    = args[3]
        Km_2      = args[4]


        # unpack states
        S = y[0]
        E_1 = y[1]
        E_1S = y[2]
        P_1 = y[3]
    
    
        # FLUXES
        # single AXP complexing
        J1 = k_f_1*S*E_1 - k_r_1*E_1S
        J2 = k_cat_1*E_1S
        J3 = Vmax_2*P_1/(Km_2 + P_1) # forms S

        # now return the odes for each state variable
        d_S = -J1 + J3
        d_E_1 = -J1 + J2
        d_E_1S = J1 - J2
        d_P_1 = J2 - J3

        return jnp.array([d_S, d_E_1, d_E_1S, d_P_1])