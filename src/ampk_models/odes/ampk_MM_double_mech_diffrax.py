"""
    Nathaniel Linden (UCSD MAE)
    Created: April 25th, 2023

    This file contains the functions for a model of MAPK activation. That model makes the 
    following high-level assumptions:
        - allow double adenine nucleotide AMPK binding
        - use mass action and Michaelis Menten kinetics
        - the reaction mechanism reflects specific activation and inhibition of
            AMPK and phos/dephos by AXPs

"""
import jax.numpy as jnp
import equinox as eqx

class ampk_MM_double_mech(eqx.Module):
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
        J1 = kOnAMP*y[0]*y[3] - kOffAMP*y[5] # AMPK
        J2 = kOnADP*y[1]*y[3] - kOffADP*y[6]
        J3 = kOnATP*y[2]*y[3] - kOffATP*y[7]
        J4 = kOnAMP*y[0]*y[4]  - kOffAMP*y[8]# pAMPK
        J5 = kOnADP*y[1]*y[4] - kOffADP*y[9]
        J6 = kOnATP*y[2]*y[4] - kOffATP*y[10]
        # double AXP complexing
        J7 = kOnAMP*y[0]*y[5] - kOffAMP*y[11]# AMPK
        J8 = kOnAMP*y[0]*y[6] - kOffAMP*y[12]
        J9 = kOnAMP*y[0]*y[7] - kOffAMP*y[13]
        J10 = kOnADP*y[1]*y[5] - kOffADP*y[12]
        J11 = kOnADP*y[1]*y[6] - kOffADP*y[14]
        J12 = kOnADP*y[1]*y[7] - kOffADP*y[15]
        J13 = kOnATP*y[2]*y[5] - kOffATP*y[13]
        J14 = kOnATP*y[2]*y[6] -  kOffATP*y[15]
        J15 = kOnATP*y[2]*y[7] - kOffATP*y[16]
        J16 = kOnAMP*y[0]*y[8] -  kOffAMP*y[17] # pAMPK
        J17 = kOnAMP*y[0]*y[9] -  kOffAMP*y[18]
        J18 = kOnAMP*y[0]*y[10] - kOffAMP*y[19]
        J19 = kOnADP*y[1]*y[8] - kOffADP*y[18]
        J20 = kOnADP*y[1]*y[9] -  kOffADP*y[20]
        J21 = kOnADP*y[1]*y[10] -  kOffADP*y[21]
        J22 = kOnATP*y[2]*y[8] - kOffATP*y[19]
        J23 = kOnATP*y[2]*y[9] - kOffATP*y[21]
        J24 = kOnATP*y[2]*y[10] - kOffATP*y[22]
        J25 = (kCaMKK*CaMKKtot*y[3])/(KmCaMKK + y[3]) # CaMKK phosphorylation
        J26 = (kCaMKK*CaMKKtot*y[5])/(KmCaMKK + y[5])
        J27 = (kCaMKK*CaMKKtot*y[6])/(KmCaMKK + y[6])
        J28 = (kCaMKK*CaMKKtot*y[7])/(KmCaMKK + y[7])
        J29 = (kCaMKK*CaMKKtot*y[11])/(KmCaMKK + y[11])
        J30 = (kCaMKK*CaMKKtot*y[12])/(KmCaMKK + y[12])
        J31 = (kCaMKK*CaMKKtot*y[13])/(KmCaMKK + y[13])
        J32 = (kCaMKK*CaMKKtot*y[14])/(KmCaMKK + y[14])
        J33 = (kCaMKK*CaMKKtot*y[15])/(KmCaMKK + y[15])
        J34 = (kCaMKK*CaMKKtot*y[16])/(KmCaMKK + y[16])
        J35 = (kLKB1*LKB1tot*y[5])/(KmLKB1 + y[5])
        J36 = (kLKB1*LKB1tot*y[6])/(KmLKB1 + y[6])
        J37 = (kLKB1*LKB1tot*y[11])/(KmLKB1 + y[11])
        J38 = (kLKB1*LKB1tot*y[12])/(KmLKB1 + y[12])
        J39 = (kLKB1*LKB1tot*y[14])/(KmLKB1 + y[14])
        J40 = (kPP*PPtot*y[4])/(KmPP + y[4])
        J41 = (kPP*PPtot*y[10])/(KmPP + y[10])
        J42 = (kPP*PPtot*y[22])/(KmPP + y[22])
        J43 = (kPP*PPtot*y[19])/(KmPP + y[19])
        J44 = (kPP*PPtot*y[21])/(KmPP + y[21])
        J45 = (kAMPK*y[8]*y[23])/(KmAMPK + y[23])
        J46 = (kAMPK*y[17]*y[23])/(KmAMPK + y[23])
        J47 = (kAMPK*y[18]*y[23])/(KmAMPK + y[23])
        J48 = (kPP1*PP1tot*y[24])/(KmPP1 + y[24])
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
        dydt = jnp.zeros((25,)) # 53 state variables jax array
        dydt = dydt.at[0].set(-J1-J4-J7-J8-J9-J16-J17-J18-Jak)
        dydt = dydt.at[1].set(-J2-J5-J10-J11-J12-J19-J20-J21-Jgly+2*Jak+Jhydro-Joxphos)
        dydt = dydt.at[2].set(-J3-J6-J13-J14-J15-J22-J23-J24+Jgly-Jak-Jhydro+Joxphos)
        # free AMPK
        dydt = dydt.at[3].set(-J1-J2-J3-J25+J40)
        dydt = dydt.at[4].set(-J4-J5-J6+J25-J40)
        # single AXP-AMPK complexes
        dydt = dydt.at[5].set(J1-J7-J10-J13-J26-J35)
        dydt = dydt.at[6].set(J2-J8-J11-J14-J27-J36)
        dydt = dydt.at[7].set(J3-J9-J12-J15-J28+J41)
        # single AXP-pAMPK complexes
        dydt = dydt.at[8].set(J4-J16-J19-J22+J26+J35)
        dydt = dydt.at[9].set(J5-J17-J20-J23+J27+J36)
        dydt = dydt.at[10].set(J6-J18-J21-J24+J28-J41)
        # double AXP-AMPK complexes
        dydt = dydt.at[11].set(J7-J29-J37)
        dydt = dydt.at[12].set(J8+J10-J30-J38)
        dydt = dydt.at[13].set(J9+J13-J31+J43)
        dydt = dydt.at[14].set(J11-J32-J39)
        dydt = dydt.at[15].set(J12+J14-J33+J44)
        dydt = dydt.at[16].set(J15-J34+J42)
        # double AXP-pAMPK complexes
        dydt = dydt.at[17].set(J16+J29-J37)
        dydt = dydt.at[18].set(J17+J19-J30+J38)
        dydt = dydt.at[19].set(J18+J22+J31-J43)
        dydt = dydt.at[20].set(J20+J32+J39)
        dydt = dydt.at[21].set(J21+J33-J44)
        dydt = dydt.at[22].set(J24+J34-J42)
        # AMPKAR
        dydt = dydt.at[23].set(-J45-J46-J47+J48)
        dydt = dydt.at[24].set(J45+J46+J47-J48)

        return dydt
    
    


    def set_kGly(self, kGly):
            """Set the glycolysis rate parameter."""
            self.kGly = kGly