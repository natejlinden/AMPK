"""
    Nathaniel Linden (UCSD MAE)
    Created: November 26th, 2024
"""
import jax.numpy as jnp
import equinox as eqx

class MA(eqx.Module):
    def __call__(self, t, y, args):
        
        # unpack parameters
        Vmax_1    = args[0]
        Km_1      = args[1]
        Vmax_2    = args[2]
        Km_2      = args[3]


        # unpack states
        S = y[0]
        P_1 = y[1]
        
    
        # FLUXES
        # single AXP complexing
        J1 = Vmax_1*S/(Km_1 + S) # forms P_1x
        J2 = Vmax_2*P_1/(Km_2 + P_1) # forms S

        # now return the odes for each state variable
        d_S = -J1 + J2
        d_P_1 = J1 - J2

        return jnp.array([d_S, d_P_1])