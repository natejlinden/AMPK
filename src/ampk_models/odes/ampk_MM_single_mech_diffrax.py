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

class ampk_MM_single_mech(eqx.Module):
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
        kOnAMP      = args[0] # AMP binding
        kOffAMP     = args[1]
        kOnADP      = args[2] # ADP binding
        kOffADP     = args[3]
        kOnATP      = args[4] # ATP binding
        kOffATP     = args[5]
        kCaMKK      = args[6] # CaMKK
        KmCaMKK     = args[7]
        kLKB1       = args[8] # LKB1 binding
        KmLKB1      = args[9]
        kPP         = args[10] # AMPK Phosphatase
        KmPP        = args[11]
        kAMPK       = args[12] # AMPK kinase
        KmAMPK      = args[13]
        kPP1        = args[14] # pAMPKAR Phosphatase
        KmPP1       = args[15] 
        # external enzyme concentrations
        CaMKKtot    = args[16]
        LKB1tot     = args[17]
        PPtot       = args[18]
        PP1tot      = args[19]

        # FLUXES
        # single AXP complexing
        J1 = kOnAMP*y[0]*y[3] - kOffAMP*y[5]# AMPK
        J2 = kOnADP*y[1]*y[3] - kOffADP*y[6]
        J3 = kOnATP*y[2]*y[3] - kOffATP*y[7]
        J4 = kOnAMP*y[0]*y[4] - kOffAMP*y[8]# pAMPK
        J5 = kOnADP*y[1]*y[4] - kOffADP*y[9]
        J6 = kOnATP*y[2]*y[4] - kOffATP*y[10]
        J7 = (kCaMKK*CaMKKtot*y[3])/(KmCaMKK + y[3]) # CaMKK phosphorylation
        J8 = (kCaMKK*CaMKKtot*y[5])/(KmCaMKK + y[5])
        J9 = (kCaMKK*CaMKKtot*y[6])/(KmCaMKK + y[6])
        J10 = (kCaMKK*CaMKKtot*y[7])/(KmCaMKK + y[7])
        J11 = (kLKB1*LKB1tot*y[5])/(KmLKB1 + y[5])
        J12 = (kLKB1*LKB1tot*y[6])/(KmLKB1 + y[6])
        J13 = (kPP*PPtot*y[4])/(KmPP + y[4])
        J14 = (kPP*PPtot*y[10])/(KmPP + y[10])
        J15 = (kAMPK*y[8]*y[11])/(KmAMPK + y[11])
        J16 = (kPP1*PP1tot*y[12])/(KmPP1 + y[12])
        # Metabolic fluxes
        # glycolysis
        Jgly = 2*self.kGly*y[1]*y[1]
        # ATP hydrolysis
        Jhydro = self.kHydro*y[2]
        # Adenylate Kinase
        Jak = (self.kForAK*y[2]*y[0]) - (self.kRevAK*y[1]*y[1]) # MASS ACTION KINETICS!
        # Oxidative Phos
        Joxphos = (self.VmaxOxPhos * ((y[1]/self.Kadp)**self.n))/(1 + ((y[1]/self.Kadp)**self.n))


        # now return the odes for each state variable
        dydt = jnp.zeros((13,)) # 53 state variables jax array
        dydt = dydt.at[0].set(-J1-Jak)
        dydt = dydt.at[1].set(-J2-Jgly+2*Jak+Jhydro-Joxphos)
        dydt = dydt.at[2].set(-J3+Jgly-Jak-Jhydro+Joxphos)
        # free AMPK
        dydt = dydt.at[3].set(-J1-J2-J3-J7)
        dydt = dydt.at[4].set(-J4-J5-J6+J7-J13)
        # single AXP-AMPK complexes
        dydt = dydt.at[5].set(J1-J8-J11)
        dydt = dydt.at[6].set(J2-J9-J12)
        dydt = dydt.at[7].set(J3-J10+J14)
        # single AXP-pAMPK complexes
        dydt = dydt.at[8].set(J4+J8+J11)
        dydt = dydt.at[9].set(J5+J9+J12)
        dydt = dydt.at[10].set(J6+J10-J14)
        # AMPKAR
        dydt = dydt.at[11].set(-J15+J16)
        dydt = dydt.at[12].set(J15-J16)

        return dydt
    
    


    def set_kGly(self, kGly):
            """Set the glycolysis rate parameter."""
            self.kGly = kGly