"""
    Nathaniel Linden (UCSD MAE)
    Created: November 26th, 2024
"""
import jax.numpy as jnp
import equinox as eqx

class MA(eqx.Module):
    def __call__(self, t, y, args):
        
        # unpack parameters
        k_f_2     = args[0]
        k_r_2     = args[1]
        k_cat_2   = args[2]
        Vmax_1    = args[3]
        Km_1      = args[4]


        # unpack states
        S = y[0]
        P_1 = y[1]
        E_2 = y[2]
        E_2P_1 = y[3]
    
        # FLUXES
        # single AXP complexing
        J1 = Vmax_1*S/(Km_1 + S) # forms P_1x
        J2 = k_f_2*P_1*E_2 - k_r_2*E_2P_1
        J3 = k_cat_2*E_2P_1 # forms S


        # now return the odes for each state variable
        d_S = -J1 + J3
        d_P_1 = J1 - J2
        d_E_2 = -J2 + J3
        d_E_2P_1 = J2 - J3

        return jnp.array([d_S, d_P_1, d_E_2, d_E_2P_1])