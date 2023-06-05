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

class ampk_qss_single(eqx.Module):
    """Right hand side of the AMPK_ma_double_mech regulation model.

    Written in the format required by the diffrax package
    """

    # fixed parameters
    # metabolic params
    k kGly: float
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
        KdAMP      = args[0] # AMP binding
        kCaMKK      = args[1] # CaMKK
        KmCaMKK     = args[2]
        kLKB1       = args[3] # LKB1 binding
        KmLKB1      = args[4]
        kPP         = args[5] # AMPK Phosphatase
        KmPP        = args[6]
        kAMPK       = args[7] # AMPK kinase
        KmAMPK      = args[8]
        kPP1        = args[9] # pAMPKAR Phosphatase
        KmPP1       = args[10]
        alpha       = args[11]
        beta        = args[12]
        # external enzyme concentrations
        CaMKKtot    = args[13]
        LKB1tot     = args[14]
        PPtot       = args[15]
        PP1tot      = args[16]
        
        # substrate conc for pAMPK dephos
        S = 0.5*(jnp.sqrt(((y[0] - y[4] + KdAMP)**2) + 4*KdAMP*y[4]) - (y[0] - y[4] + KdAMP))

        # FLUXES
        # single AXP complexing
        J1 = (kCaMKK*CaMKKtot*y[3])/(KmCaMKK + y[3])
        J2 = (kLKB1*LKB1tot*y[3])/(KmLKB1 + y[3])
        J3 = (kPP*PPtot*S)/(KmPP + S) 
        J4 = (((kAMPK*y[4]*y[5])/KmAMPK) + ((beta*kAMPK*y[4]*y[5]*y[0])/(alpha*KmAMPK*KdAMP)))/(1 + (y[5]/KmAMPK) + (y[0]/KdAMP) + ((y[5]*y[0])/(alpha*KmAMPK*KdAMP)))
        J5 = (kPP1*PP1tot*y[6])/(KmPP1 + y[6]) 
        
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
        dydt = jnp.zeros((7,))
        dydt = dydt.at[0].set(-JAK) # AMP
        dydt = dydt.at[1].set(-2*Jgly + 2*JAK + Jhydro - Joxphos) # ADP
        dydt = dydt.at[2].set(2*Jgly -JAK - Jhydro + Joxphos) # ATP
        dydt = dydt.at[3].set(-J1 - J2 + J3) # AMPK
        dydt = dydt.at[4].set(J1 + J2 - J3) # pAMPK
        dydt = dydt.at[5].set(-J4 + J5) # AMPKAR
        dydt = dydt.at[6].set(J4 - J5) # pAMPKAR

        return dydt
    
    
    def set_kGly(self, kGly):
            """Set the glycolysis rate parameter."""
            self.kGly = kGly