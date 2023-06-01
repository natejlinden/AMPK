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
        KmCaMKK    = args[2]
        kLKB1      = args[3] # LKB1 binding
        KmLKB1     = args[4]
        kPP        = args[5] # AMPK Phosphatase
        KmPP       = args[6]
        kAMPK      = args[7] # AMPK kinase
        KmAMPK     = args[8]
        kPP1       = args[9] # pAMPKAR Phosphatase
        KmPP1      = args[10]
        alpha      = args[11]
        beta       = args[12]
        alpha2     = args[13]
        beta2      = args[14]
        
        # external enzyme concentrations
        CaMKKtot   = args[15]
        LKB1tot    = args[16]
        PPtot      = args[17]
        PP1tot     = args[18]
        
        # substrate conc for pAMPK dephos
        S = 0.5*(jnp.sqrt(((y[0] - y[4] + KdAMP)**2) + 4*KdAMP*y[4]) - (y[0] - y[4] + KdAMP))

        # FLUXES
        # single AXP complexing
        J1 = (kCaMKK*CaMKKtot*y[3])/(KmCaMKK + y[3])
        J2 = (((kLKB1*LKB1tot*y[3])/KmLKB1) + ((beta2*kLKB1*LKB1tot*y[3]*y[0])/(alpha2*KmLKB1*KdAMP)))/(1 + (y[3]/KmLKB1) + (y[0]/KdAMP) + ((y[3]*y[0])/(alpha2*KmLKB1*KdAMP)))
        J3 = (kPP*PPtot*S)/(KmPP + S) 
        J4 = (((kAMPK*y[4]*y[5])/KmAMPK) + ((beta*kAMPK*y[4]*y[5]*y[0])/(alpha*KmAMPK*KdAMP)))/(1 + (y[5]/KmAMPK) + (y[0]/KdAMP) + ((y[5]*y[0])/(alpha*KmAMPK*KdAMP)))
        J5 = (kPP1*PP1tot*y[6])/(KmPP1 + y[6]) 
        
        # Metabolic fluxes
        # glycolysis
        Jgly = 2*self.kGly*y[1]*y[1]
        # ATP hydrolysis
        Jhydro = self.kHydro*y[2]
        # Adenylate Kinase
        JakFor = (self.kForAK*y[2]*y[0]) # MASS ACTION KINETICS!
        JakRev = (self.kRevAK*y[1]*y[1])
        # Oxidative Phos
        Joxphos = (self.VmaxOxPhos * ((y[1]/self.Kadp)**self.n))/(1 + ((y[1]/self.Kadp)**self.n))


        # now return the odes for each state variable
        dydt = jnp.zeros((7,))
        dydt = dydt.at[0].set(-JakFor + JakRev) # AMP
        dydt = dydt.at[1].set(-2*Jgly + 2*JakFor - 2*JakRev + Jhydro - Joxphos) # ADP
        dydt = dydt.at[2].set(2*Jgly - JakFor + JakRev - Jhydro + Joxphos) # ATP
        dydt = dydt.at[3].set(-J1 - J2 + J3) # AMPK
        dydt = dydt.at[4].set(J1 + J2 - J3) # pAMPK
        dydt = dydt.at[5].set(-J4 + J5) # AMPKAR
        dydt = dydt.at[6].set(J4 - J5) # pAMPKAR

        return dydt
    
    
    def set_kGly(self, kGly):
            """Set the glycolysis rate parameter."""
            self.kGly = kGly