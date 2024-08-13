"""
    Nathaniel Linden (UCSD MAE)
    Created: April 25th, 2023

    This file contains the functions for a model of MAPK activation. That model makes the 
    following high-level assumptions:
        - allow single adenine nucleotide AMPK binding
        - use mass action and Michaelis Menten kinetics
        - the reaction mechanism reflects specific activation and inhibition of
            AMPK and phos/dephos by AXPs

"""
import jax.numpy as jnp
import equinox as eqx

class newmech_MA_single(eqx.Module):
    """Right hand side of the AMPK_ma_double_mech regulation model.

    Written in the format required by the diffrax package
    """

    # fixed parameters
    # metabolic params
    kGly: float
    kHydro: float
    VforAK: float
    KeqAK: float
    kmm: float
    kmd: float
    kmt: float
    VmaxOxPhos: float
    Kadp: float
    n: float

    def __call__(self, t, y, args):
        """Right hand side of the AMPK_ma_double_mech regulation model.

        Written in the format required by the diffrax package
        """
        # unpack parameters
        kOffAMP    = args[0] # AMP binding
        kOnAMP     = args[1]
        kOnCaMKK   = args[2] # CaMKK
        kOffCaMKK  = args[3]
        kPhosCaMKK = args[4] 
        kOnLKB1    = args[5] # LKB1
        kOffLKB1   = args[6]
        kPhosLKB1  = args[7]
        kOnPP      = args[8] # AMPK Phosphatase
        kOffPP     = args[9]
        kDephosPP  = args[10]
        kOnAMPK    = args[11] # AMPK kinase
        kOffAMPK   = args[12]
        kPhosAMPK  = args[13]
        kOnPP1     = args[14] # pAMPKAR Phosphatase
        kOffPP1    = args[15]
        kDephosPP1 = args[16]
        alpha      = args[17]
        beta       = args[18]
        
        # FLUXES
        # single AXP complexing
        J1  = kOnAMP*y[0]*y[3]  - kOffAMP*y[5] 
        J2  = kOnAMP*y[0]*y[4]  - kOffAMP*y[6] 
        J3  = kOnAMP*y[0]*y[17]  - kOffAMP*y[18]
        J4  = kOnCaMKK*y[3]*y[7] - kOffCaMKK*y[8] 
        J5  = kPhosCaMKK*y[8] 
        J6  = kOnCaMKK*y[5]*y[7] - kOffCaMKK*y[9] 
        J7  = kPhosCaMKK*y[9] 
        J8  = kOnLKB1*y[3]*y[10] - kOffLKB1*y[11] 
        J9  = kPhosLKB1*y[11] 
        J10 = kOnLKB1*y[5]*y[10] - kOffLKB1*y[12] 
        J11 = kPhosLKB1*y[12] 
        J12 = kOnPP*y[13]*y[4] - kOffPP*y[14] 
        J13 = kDephosPP*y[14] 
        J14 = kOnAMPK*y[4]*y[15] - kOffAMPK*y[17] 
        J15 = kPhosAMPK*y[17] 
        J16 = kOnAMPK*y[6]*y[15] - alpha*kOffAMPK*y[18] 
        J17 = beta*kPhosAMPK*y[18] 
        J18 = kOnPP1*y[19]*y[16] - kOffPP1*y[20] 
        J19 = kDephosPP1*y[20]
        
        # Metabolic fluxes
        # glycolysis
        Jgly = 2*self.kGly*y[1]*y[1]
        # ATP hydrolysis
        Jhydro = self.kHydro*y[2]
        # Adenylate Kinase
        num_for = (self.VforAK*y[2]*y[0])/(self.kmt*self.kmm)
        den = (1 + (y[2]/self.kmt) + (y[0]/self.kmm) + ((y[2]*y[0])/(self.kmt*self.kmm)) + 
                   ((2*y[1])/self.kmd) + ((y[1]**2)/(self.kmd**2)))
        VrevAK = (self.VforAK*(self.kmd**2))/(self.KeqAK*self.kmt*self.kmm)
        num_rev = (VrevAK*(y[1]**2))/(self.kmd**2)
        JAK = (num_for - num_rev)/den
        # Oxidative Phos
        Joxphos = (self.VmaxOxPhos * ((y[1]/self.Kadp)**self.n))/(1 + ((y[1]/self.Kadp)**self.n))

        # now return the odes for each state variable
        dydt = jnp.zeros_like(y)
        dydt = dydt.at[0].set(-J1 - J2 - J3 -JAK) # AMP
        dydt = dydt.at[1].set(-Jgly + 2*JAK + Jhydro - Joxphos) # ADP
        dydt = dydt.at[2].set(Jgly - JAK - Jhydro + Joxphos) # ATP
        dydt = dydt.at[3].set(-J1 - J4 - J8 + J13) # AMPK
        dydt = dydt.at[4].set(-J2 + J5 + J9 - J12 - J14 + J15) #pAMPK
        dydt = dydt.at[5].set(J1 - J6 - J10) # AMP-AMPK
        dydt = dydt.at[6].set(J2 + J7 + J11 - J16 + J17) # AMP-pAMPK
        dydt = dydt.at[7].set(-J4 - J6 + J5 + J7) # CaMKK
        dydt = dydt.at[8].set(J4 - J5) # CaMKK-AMPK
        dydt = dydt.at[9].set(J6 - J7) # CaMKK-AMP-AMPK
        dydt = dydt.at[10].set(-J8 -J10 + J9 + J11) # LKB1
        dydt = dydt.at[11].set(J8 - J9) # LKB1-AMPK
        dydt = dydt.at[12].set(J10 - J11) # LKB1-AMP-AMPK
        dydt = dydt.at[13].set(-J12 + J13) # PP
        dydt = dydt.at[14].set(J12 - J13) # PP-pAMPK
        dydt = dydt.at[15].set(-J14 - J16 + J19) # AMPKAR
        dydt = dydt.at[16].set(J15 + J17 - J18) # pAMPKAR
        dydt = dydt.at[17].set(-J3 + J14 - J15) # AMPKAR-pAMPK
        dydt = dydt.at[18].set(J3 + J16 - J17) # AMPKAR-AMP-pAMPK
        dydt = dydt.at[19].set(-J18 + J19) # PP1
        dydt = dydt.at[20].set(J18 - J19) # PP1-pAMPKAR

        return dydt
    
    
    def set_kGly(self, kGly):
            """Set the glycolysis rate parameter."""
            self.kGly = kGly