import jax.numpy as jnp
import equinox as eqx

class metabolism(eqx.Module):

    def __call__(self, t, y, args):

        # unpack parameters
        # metabolic params
        kGly        = args[0]
        kHydro      = args[1]
        VforAK      = args[2]
        KeqAK       = args[3]
        kmm         = args[4]
        kmd         = args[5]
        kmt         = args[6]
        VmaxOxPhos  = args[7]
        Kadp        = args[8]
        n           = args[9]

        # unpack state variables
        AMP = y[0]
        ADP = y[1]
        ATP = y[2]

        # Metabolic fluxes
        # glycolysis
        Jgly = 2*kGly*ADP*ADP

        # ATP hydrolysis
        Jhydro = kHydro*ATP
        num_for = (VforAK*ATP*AMP)/(kmt*kmm)
        den = (1 + (ATP/kmt) + (AMP/kmm) + ((ATP*AMP)/(kmt*kmm)) + 
                    ((2*ADP)/kmd) + ((ADP**2)/(kmd**2)))
        VrevAK = (VforAK*(kmd**2))/(KeqAK*kmt*kmm)
        num_rev = (VrevAK*(ADP**2))/(kmd**2)
        JAK = (num_for - num_rev)/den
        
        # Oxidative Phos
        Joxphos = (VmaxOxPhos * ((ADP/Kadp)**n))/(1 + ((ADP/Kadp)**n))

        # return the odes for each state variable
        dydt = jnp.zeros((3,)) # 53 state variables jax array
        dydt = dydt.at[0].set(-JAK) # AMP
        dydt = dydt.at[1].set(-Jgly+2*JAK+Jhydro-Joxphos) # ADP
        dydt = dydt.at[2].set(Jgly-JAK-Jhydro+Joxphos) # ATP

        return dydt