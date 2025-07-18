import jax.numpy as jnp
import equinox as eqx

class metabolism_Cocci(eqx.Module):

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
        TCr        = args[10]
        VforCK      = args[10]
        Kia         = args[11]
        Kib         = args[12]
        Kiq         = args[13]
        Kb          = args[14]
        Kp          = args[15]
        KeqCK       = args[16]

        # unpack state variables
        AMP = y[0] # x(3) in Cocimiglio et al
        ADP = y[1] # x(2) in Cocimiglio et al
        ATP = y[2] # x(1) in Cocimiglio et al
        PCr = y[3] # x(4) in Cocimiglio et al
        Pi = y[4] # x(5) in Cocimiglio et al

        # Metabolic fluxes
        # glycolysis
        Jgly = kGly*ADP #2*kGly*ADP*ADP

        # ATP hydrolysis
        Jhydro = kHydro*ATP

        # Adenylate Kinase
        # written as (VforAK*ATP)/(kmt*kmm) in cocci, but units dont make sense
        num_for = (VforAK*ATP*AMP)/(kmt*kmm)

        den_ak = (1 + (ATP/kmt) + (AMP/kmm) + ((ATP*AMP)/(kmt*kmm)) + 
                    ((2*ADP)/kmd) + ((ADP**2)/(kmd**2)))
    
        VrevAK = (VforAK*(kmd**2))/(KeqAK*kmt*kmm)
        num_rev = (VrevAK*(ADP**2))/(kmd**2)
        JAK = (num_for - num_rev)/den_ak # ADP forming direction 

        # Oxidative Phos
        Joxphos = (VmaxOxPhos * ((ADP/Kadp)**n))/(1 + ((ADP/Kadp)**n))

        # Creatine kinase
        den_ck = 1 + (ADP/Kia) + (PCr/Kib) + (ATP/Kiq) + ((ADP*PCr)/(Kia*Kb)) + (((TCr - PCr)*ATP)/(Kiq*Kp))
        num_forCK = ((VforCK*ADP*PCr)/(Kia*Kb))
        VrevCK = (VforCK*Kiq*Kp)/(KeqCK*Kia*Kb)
        num_revCK = ((VrevCK*ATP*(TCr - PCr))/(Kiq*Kp))

        JCK = (num_revCK - num_forCK)/den_ck # Pi forming direction

        # return the odes for each state variable
        dydt = jnp.zeros((4,)) # 53 state variables jax array
        dydt = dydt.at[0].set(-JAK) # AMP
        dydt = dydt.at[1].set(-Jgly+2*JAK+Jhydro-Joxphos+JCK) # ADP
        dydt = dydt.at[2].set(Jgly-JAK-Jhydro+Joxphos-JCK) # ATP
        dydt = dydt.at[3].set(JCK) # PCr
        # dydt = dydt.at[4].set(Jhydro - Joxphos) # Pi

        return dydt