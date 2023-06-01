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

class ampk_newmech_MA_single(eqx.Module):
    """Right hand side of the AMPK_ma_double_mech regulation model.

    Written in the format required by the diffrax package
    """

    # fixed parameters
    # metabolic params
    kGly: float
    kHydro: float
    kForAK: float
    kRevAK: float
    VmaxOxPhos: float
    Kadp: float
    n: float


    def __init__(self, kGly, kHydro, kForAK, kRevAK, VmaxOxPhos, Kadp, n):
        """Initialize the model. Set fixed parameters."""
        # TODO: expand docstring
        # TODO: add default values
        self.kGly = kGly
        self.kHydro = kHydro
        self.kForAK = kForAK
        self.kRevAK = kRevAK
        self.VmaxOxPhos = VmaxOxPhos
        self.Kadp = Kadp
        self.n = n

    def __call__(self, t, y, args):
        """Right hand side of the AMPK_ma_double_mech regulation model.

        Written in the format required by the diffrax package
        """
        # unpack parameters
        KdAMP      = args[0] # AMP binding
        kCaMKK     = args[1] # CaMKK
        konCaMKK   = args[2]
        koffCaMKK  = args[3]
        kLKB1      = args[4] # LKB1 
        konLKB1    = args[5]
        koffLKB1   = args[6]
        kPP        = args[7] # AMPK Phosphatase
        konPP      = args[8]
        koffPP     = args[9]
        kAMPK      = args[10] # AMPK kinase
        konAMPK    = args[11]
        koffAMPK   = args[12]
        kPP1       = args[13] # pAMPKAR Phosphatase
        konPP1     = args[14]
        koffPP1    = args[15]
        alpha      = args[16]
        beta       = args[17]
        
        konAMP = 1.0
        
        # FLUXES
        # single AXP complexing
        J1  = konAMP*y[0]*y[3]  - KdAMP*y[5] 
        J2  = konAMP*y[0]*y[4]  - KdAMP*y[6] 
        J3  = konAMP*y[0]*y[17]  - KdAMP*y[18]
        J4  = konCaMKK*y[3]*y[7] - koffCaMKK*y[8] 
        J5  = kCaMKK*y[8] 
        J6  = konCaMKK*y[5]*y[7] - koffCaMKK*y[9] 
        J7  = kCaMKK*y[9] 
        J8  = konLKB1*y[3]*y[10] - koffLKB1*y[11] 
        J9  = kLKB1*y[11] 
        J10 = konLKB1*y[5]*y[10] - koffLKB1*y[12] 
        J11 = kLKB1*y[12] 
        J12 = konPP*y[13]*y[4] - koffPP*y[14] 
        J13 = kPP*y[14] 
        J14 = konAMPK*y[4]*y[15] - koffAMPK*y[17] 
        J15 = kAMPK*y[17] 
        J16 = konAMPK*y[6]*y[15] - alpha*koffAMPK*y[18] 
        J17 = beta*kAMPK*y[18] 
        J18 = konPP1*y[19]*y[16] - koffPP1*y[20] 
        J19 = kPP1*y[20]
        
        # Metabolic fluxes
        # glycolysis
        Jgly = 2*self.kGly*y[1]*y[1]
        # ATP hydrolysis
        Jhydro = self.kHydro*y[2]
        # Adenylate Kinase
        JAK = (self.kForAK*y[2]*y[0]) - (self.kRevAK*y[1]*y[1])
        # Oxidative Phos
        Joxphos = (self.VmaxOxPhos * ((y[1]/self.Kadp)**self.n))/(1 + ((y[1]/self.Kadp)**self.n))

        # now return the odes for each state variable
        dydt = jnp.zeros_like(y)
        dydt = dydt.at[0].set(-J1 - J2 - J3 -JAK) # AMP
        dydt = dydt.at[1].set(-2*Jgly+2*JAK+Jhydro-Joxphos) # ADP
        dydt = dydt.at[2].set(2*Jgly-JAK-Jhydro+Joxphos) # ATP
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
        dydt = dydt.at[15].set(J14 - J16 + J19) # AMPKAR
        dydt = dydt.at[16].set(J15 + J17 - J18) # pAMPKAR
        dydt = dydt.at[17].set(-J3 + J14 - J15) # AMPKAR-pAMPK
        dydt = dydt.at[18].set(J3 + J16 - J17) # AMPKAR-AMP-pAMPK
        dydt = dydt.at[19].set(-J18 + J19) # PP1
        dydt = dydt.at[20].set(J18 - J19) # PP1-pAMPKAR

        return dydt
    
    
    def set_kGly(self, kGly):
            """Set the glycolysis rate parameter."""
            self.kGly = kGly