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

class MM_single(eqx.Module):
    """Right hand side of the AMPK_ma_double_mech regulation model.

    Written in the format required by the diffrax package
    """

    # fixed metabolism parameters
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
    VforCK: float
    Kb: float
    Kia: float
    Kib: float
    Kiq: float
    Kp: float
    KeqCK: float
    TCr: float


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

        # unpack state variables
        AMP = y[0]
        ADP = y[1]
        ATP = y[2]
        PCr = y[3]
        AMPK = y[4]
        pAMPK = y[5]
        AMP_AMPK = y[6]
        ADP_AMPK = y[7]
        ATP_AMPK = y[8]
        AMP_pAMPK = y[9]
        ADP_pAMPK = y[10]
        ATP_pAMPK = y[11]
        AMPKAR = y[12]
        pAMPKAR = y[13]

        # FLUXES
        # single AXP complexing
        J1 = kOnAMP*AMP*AMPK - kOffAMP*AMP_AMPK # AMPK
        J2 = kOnADP*ADP*AMPK - kOffADP*ADP_AMPK
        J3 = kOnATP*ATP*AMPK - kOffATP*ATP_AMPK
        J4 = kOnAMP*AMP*pAMPK - kOffAMP*AMP_pAMPK # pAMPK
        J5 = kOnADP*ADP*pAMPK - kOffADP*ADP_pAMPK
        J6 = kOnATP*ATP*pAMPK - kOffATP*ATP_pAMPK
        J7 = (kCaMKK*CaMKKtot*AMPK)/(KmCaMKK + AMPK) # CaMKK phosphorylation
        J8 = (kCaMKK*CaMKKtot*AMP_AMPK)/(KmCaMKK + AMP_AMPK)
        J9 = (kCaMKK*CaMKKtot*ADP_AMPK)/(KmCaMKK + ADP_AMPK)
        J10 = (kCaMKK*CaMKKtot*ATP_AMPK)/(KmCaMKK + ATP_AMPK)
        J11 = (kLKB1*LKB1tot*AMP_AMPK)/(KmLKB1 + AMP_AMPK) # LKB1 phos
        J12 = (kLKB1*LKB1tot*ADP_AMPK)/(KmLKB1 + ADP_AMPK)
        J13 = (kPP*PPtot*pAMPK)/(KmPP + pAMPK) # PP dephos
        J14 = (kPP*PPtot*ATP_pAMPK)/(KmPP + ATP_pAMPK) # PP dephos
        J15 = (kAMPK*AMP_pAMPK*AMPKAR)/(KmAMPK + AMPKAR) # AMPKAR phos
        J16 = (kPP1*PP1tot*pAMPKAR)/(KmPP1 + pAMPKAR) # PP1 dephos
        
        # Metabolic fluxes
        # glycolysis
        Jgly = self.kGly*ADP #2*kGly*ADP*ADP
        # ATP hydrolysis
        Jhydro = self.kHydro*ATP
        # Adenylate Kinase
        # written as (VforAK*ATP)/(kmt*kmm) in cocci, but units dont make sense
        num_for = (self.VforAK*ATP*AMP)/(self.kmt*self.kmm)
        den_ak = (1 + (ATP/self.kmt) + (AMP/self.kmm) + ((ATP*AMP)/(self.kmt*self.kmm)) + 
                    ((2*ADP)/self.kmd) + ((ADP**2)/(self.kmd**2)))
        VrevAK = (self.VforAK*(self.kmd**2))/(self.KeqAK*self.kmt*self.kmm)
        num_rev = (VrevAK*(ADP**2))/(self.kmd**2)
        JAK = (num_for - num_rev)/den_ak # ADP forming direction 
        # Oxidative Phos
        Joxphos = (self.VmaxOxPhos * ((ADP/self.Kadp)**self.n))/(1 + ((ADP/self.Kadp)**self.n))
        # Creatine kinase
        den_ck = 1 + (ADP/self.Kia) + (PCr/self.Kib) + (ATP/self.Kiq) + ((ADP*PCr)/(self.Kia*self.Kb)) + (((self.TCr - PCr)*ATP)/(self.Kiq*self.Kp))
        num_forCK = ((self.VforCK*ADP*PCr)/(self.Kia*self.Kb))
        VrevCK = (self.VforCK*self.Kiq*self.Kp)/(self.KeqCK*self.Kia*self.Kb)
        num_revCK = ((VrevCK*ATP*(self.TCr - PCr))/(self.Kiq*self.Kp))
        JCK = (num_revCK - num_forCK)/den_ck # Pi forming direction


        # now return the odes for each state variable
        d_AMP = -J1-J4-JAK # AMP
        d_ADP = -J2-J5-Jgly+2*JAK+Jhydro-Joxphos+JCK # ADP
        d_ATP = -J3-J6+Jgly-JAK-Jhydro+Joxphos-JCK # ATP
        d_PCr = JCK
        # free AMPK
        d_AMPK = -J1-J2-J3-J7+J13 # AMPK
        d_pAMPK = -J4-J5-J6+J7-J13 # pAMPK
        # single AXP-AMPK complexes
        d_AMP_AMPK = J1-J8-J11 # AMP_AMPK
        d_ADP_AMPK = J2-J9-J12 # ADP_AMPK
        d_ATP_AMPK = J3-J10+J14 # ATP_AMPK
        # single AXP-pAMPK complexes
        d_AMP_pAMPK = J4+J8+J11 # AMP_pAMPK
        d_ADP_pAMPK = J5+J9+J12 # ADP_pAMPK
        d_ATP_pAMPK = J6+J10-J14 # ATP_pAMPK
        # AMPKAR
        d_AMPKAR = -J15+J16 # AMPKAR
        d_pAMPKAR = J15-J16 # pAMPKAR

        return [d_AMP, d_ADP, d_ATP, d_PCr, d_AMPK, d_pAMPK, d_AMP_AMPK, d_ADP_AMPK, d_ATP_AMPK, d_AMP_pAMPK, d_ADP_pAMPK, d_ATP_pAMPK, d_AMPKAR, d_pAMPKAR]

    def set_kGly(self, kGly):
            """Set the glycolysis rate parameter."""
            self.kGly = kGly


def MM_single_numpyro_model(data=None, data_std=None, solver=None):

    """Returns a numpyro model for the AMPK activation model.

    Args:
        data (np.ndarray): The data to fit the model to.
        data_std (np.ndarray): The standard deviation of the data.
        solver (wrapper around a dfrx.Solver): The solver to use for the model.
    """
    # TODO: update params to match the model and identifiability results
    # fixed parameters
    kOnAMP = numpyro.deterministic('kOnAMP', 1.0)
    kOnADP = numpyro.deterministic('kOnADP', 1.0)
    kOnATP = numpyro.deterministic('kOnATP', 1.0)
    kOnCaMKK = numpyro.deterministic('kOnCaMKK', 1.0)
    kOnLKB1 = numpyro.deterministic('kOnLKB1', 1.0)
    kOnPP = numpyro.deterministic('kOnPP', 1.0)
    kOnAMPK = numpyro.deterministic('kOnAMPK', 1.0)
    kOnPP1 = numpyro.deterministic('kOnPP1', 1.0)
    
    # PRIORS
    kOffAMP = numpyro.sample('kOffAMP', dist.Gamma(2.301, rate=209.419))
    kOffADP = numpyro.sample('kOffADP', dist.Gamma(3.781, rate=496.016))
    kOffATP = numpyro.sample('kOffATP', dist.Gamma(2.302, 29.088))
    kOffCaMKK = numpyro.sample('kOffCaMKK', dist.Gamma(2.302, 39.669))
    kPhosCaMKK = numpyro.sample('kPhosCaMKK', dist.Gamma(3.781, 918.55))
    kOffLKB1 = numpyro.sample('kOffLKB1', dist.Gamma(2.302, 0.375))
    kPhosLKB1 = numpyro.sample('kPhosLKB1', dist.Gamma(2.302, 133.573))
    kOffPP = numpyro.sample('kOffPP', dist.Gamma(2.302, 0.935))
    kDephosPP = numpyro.sample('kDephosPP', dist.Gamma(3.781, 6763.874))
    kOffAMPK = numpyro.sample('kOffAMPK', dist.Gamma(2.302, 6.167))
    kPhosAMPK = numpyro.sample('kPhosAMPK', dist.Gamma(3.781, 38751.36))
    kOffPP1 = numpyro.sample('kOffPP1', dist.Gamma(2.302, 0.935))
    kDephosPP1 = numpyro.sample('kDephosPP1', dist.Gamma(3.781, 6763.874))

    # std of likelihood
    if data_std is None:
        data_std = numpyro.sample('data_sigma', dist.LogNormal(0, 0.01))

    # run solver
    params = (kOnAMP, kOffAMP, kOnADP, kOffADP, kOnATP, kOffATP, kOnCaMKK, kOffCaMKK, kPhosCaMKK, kOnLKB1, kOffLKB1, kPhosLKB1, kOnPP, kOffPP, kDephosPP, kOnAMPK, kOffAMPK, kPhosAMPK, kOnPP1, kOffPP1, kDephosPP1)
    if solver is not None:
        predict = solver(params)
    else:
        raise Exception("Solver is not defined")

    # likelihood conditioned on the observations
    numpyro.sample('obs', dist.Normal(data, data_std), obs=data)