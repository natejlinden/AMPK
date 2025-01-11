"""
    Nathaniel Linden (UCSD MAE)
    Created: April 24th, 2023

    This file contains the functions for a model of MAPK activation. That model makes the 
    following high-level assumptions:
        - allow double adenine nucleotide AMPK binding
        - use only mass action kinetics
        - the reaction mechanism reflects specific activation and inhibition of
            AMPK and phos/dephos by AXPs

    The get_params functions returns a dictionary of the parameters in the correct
    format.

    The get_states function returns a dictionary od the parameters in the correct
    format.

    The RHS function is written following the syntax specified for the pymc/sunode
    package. See docs here https://sunode.readthedocs.io/en/latest/without_pymc.html
"""
import jax.numpy as jnp
import equinox as eqx

class MA_single(eqx.Module):
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
        """Right hand side of the AMPK_ma_single_mech regulation model.

        Written in the format required by the diffrax package
        """
        # unpack parameters
        kOnAMP      = args[0] # AMP binding
        kOffAMP     = args[1]
        kOnADP      = args[2] # ADP binding
        kOffADP     = args[3]
        kOnATP      = args[4] # ATP binding
        kOffATP     = args[5]
        kOnCaMKK    = args[6] # CaMKK binding
        kOffCaMKK   = args[7]
        kPhosCaMKK  = args[8] # phosphorylation
        kOnLKB1     = args[9] # LKB1 binding
        kOffLKB1    = args[10]
        kPhosLKB1   = args[11] # phosphorylation
        kOnPP       = args[12] # Phosphatase AMPK binding
        kOffPP      = args[13]
        kDephosPP   = args[14] # dephosphorylation
        kOnAMPK     = args[15] # pAMPK binds AMPKAR
        kOffAMPK    = args[16]
        kPhosAMPK   = args[17] # pAMPK phosphorylates AMPKAR
        kOnPP1      = args[18] # Phosphatase AMPKAR binding
        kOffPP1     = args[19] 
        kDephosPP1  = args[20]

        # unpack states
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
        CaMKK = y[12]
        CaMKK_AMPK = y[13]
        CaMKK_AMP_AMPK = y[14]
        CaMKK_ADP_AMPK = y[15]
        CaMKK_ATP_AMPK = y[16]
        LKB1 = y[17]
        LKB1_AMP_AMPK = y[18]
        LKB1_ADP_AMPK = y[19]
        PP = y[20]
        PP_pAMPK = y[21]
        PP_ATP_pAMPK = y[22]
        AMPKAR = y[23]
        pAMPKAR = y[24]
        AMPKAR_AMP_pAMPK = y[25]
        PP1 = y[26]
        PP1_pAMPKAR = y[27]

        # FLUXES
        # single AXP complexing
        J1 = kOnAMP*AMP*AMPK - kOffAMP*AMP_AMPK # AMPK
        J2 = kOnADP*ADP*AMPK - kOffADP*ADP_AMPK
        J3 = kOnATP*ATP*AMPK - kOffATP*ATP_AMPK
        J4 = kOnAMP*AMP*pAMPK - kOffAMP*AMP_pAMPK # pAMPK
        J5 = kOnADP*ADP*pAMPK - kOffADP*ADP_pAMPK
        J6 = kOnATP*ATP*pAMPK - kOffATP*ATP_pAMPK
        # CaMKK complexing and phosphorylation
        J7 = kOnCaMKK*CaMKK*AMPK - kOffCaMKK*CaMKK_AMPK
        J8 = kPhosCaMKK*CaMKK_AMPK
        J9 = kOnCaMKK*CaMKK*AMP_AMPK - kOffCaMKK*CaMKK_AMP_AMPK  
        J10 = kPhosCaMKK*CaMKK_AMP_AMPK 
        J11 = kOnCaMKK*CaMKK*ADP_AMPK - kOffCaMKK*CaMKK_ADP_AMPK
        J12 = kPhosCaMKK*CaMKK_ADP_AMPK
        J13 = kOnCaMKK*CaMKK*ATP_AMPK - kOffCaMKK*CaMKK_ATP_AMPK
        J14 = kPhosCaMKK*CaMKK_ATP_AMPK
        # LKB1 complexing and phosphorylation
        J15 = kOnLKB1*LKB1*AMP_AMPK - kOffLKB1*LKB1_AMP_AMPK
        J16 = kPhosLKB1*LKB1_AMP_AMPK
        J17 = kOnLKB1*LKB1*ADP_AMPK -  kOffLKB1*LKB1_ADP_AMPK
        J18 = kPhosLKB1*LKB1_ADP_AMPK
        # phosphatase binding and dephosphorylation
        J19 = kOnPP*PP*pAMPK - kOffPP*PP_pAMPK
        J20 =  kDephosPP*PP_pAMPK
        J21 = kOnPP*PP*ATP_pAMPK - kOffPP*PP_ATP_pAMPK
        J22 =  kDephosPP*PP_ATP_pAMPK
        # AMPK binding to AMPAKAR and phosphorylation
        J23 = kOnAMPK*AMPKAR*AMP_pAMPK - kOffAMPK*AMPKAR_AMP_pAMPK
        J24 = kPhosAMPK*AMPKAR_AMP_pAMPK
        # PP1 binding to AMPKAR and dephosphorylation  
        J25 = kOnPP1*PP1*pAMPKAR - kOffPP1*PP1_pAMPKAR
        J26 = kDephosPP1*PP1_pAMPKAR

        # additional fluxes to allow AXP to bind/unbind enzyme--AMPK complexes
        # Ja = kOnAMP*AMP*CaMKK_AMPK - kOffADP*CaMKK_AMP_AMPK
        # Jb = kOnADP*ADP*CaMKK_AMPK - kOffADP*CaMKK_ADP_AMPK
        # Jc = kOnATP*ATP*CaMKK_AMPK - kOffATP*CaMKK_ATP_AMPK
        # Jd = kOnATP*ATP*PP_pAMPK - kOffATP*PP_ATP_pAMPK
        Ja = 0
        Jb = 0
        Jc = 0
        Jd = 0

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
        d_AMP = -J1-J4-JAK-Ja # AMP
        d_ADP = -J2-J5-Jgly+2*JAK+Jhydro-Joxphos+JCK-Jb # ADP
        d_ATP = -J3-J6+Jgly-JAK-Jhydro+Joxphos-JCK-Jc-Jd # ATP
        d_PCr = JCK
        # free AMPK
        d_AMPK = -J1-J2-J3-J7+J20 # AMPK
        d_pAMPK = -J4-J5-J6+J8-J19 # pAMPK
        # single AXP-AMPK complexes
        d_AMP_AMPK = J1-J9-J15 # AMP_AMPK
        d_ADP_AMPK = J2-J11-J17 # ADP_AMPK
        d_ATP_AMPK = J3-J13+J22 # ATP_AMPK
        # single AXP-pAMPK complexes
        d_AMP_pAMPK = J4+J10+J16-J23+J24 # AMP_pAMPK
        d_ADP_pAMPK = J5+J12+J18 # ADP_pAMPK
        d_ATP_pAMPK = J6+J14-J21 #  ATP_pAMPK
        # CaMKK complexes
        d_CaMKK = -J7+J8-J9+J10-J11+J12-J13+J14 # CaMKK
        d_CaMKK_AMPK = J7-J8-Ja-Jb-Jc # CaMKK_AMPK
        d_CaMKK_AMP_AMPK = J9-J10+Ja # CaMKK_AMP_AMPK
        d_CaMKK_ADP_AMPK = J11-J12+Jb # CaMKK_ADP_AMPK
        d_CaMKK_ATP_AMPK = J13-J14+Jc # CaMKK_ATP_AMPK
        # LKB1 complexes
        d_LKB1 = -J15+J16-J17+J18 # LKB1
        d_LKB1_AMP_AMPK = J15-J16 # LKB1_AMP_AMPK
        d_LKB1_ADP_AMPK = J17-J18 # LKB1_ADP_AMPK
        # AMPK phosphatase complexes
        d_PP = -J19+J20-J21+J22 # PP
        d_PP_pAMPK = J19-J20-Jd # PP_pAMPK
        d_PP_ATP_pAMPK = J21-J22+Jd # PP_ATP_pAMPK
        # free AMPKAR
        d_AMPKAR = -J23+J26 # AMPKAR
        d_pAMPKAR = J24-J25 # pAMPKAR
        # AMPKAR-pAMPK complexes
        d_AMPKAR_AMP_pAMPK = J23-J24 # AMPKAR_AMP_pAMPK
        # AMPKAR phosphatase complexes
        d_PP1 = -J25+J26 # PP1
        d_PP1_pAMPKAR = J25-J26 # PP1_pAMPKAR

        return [d_AMP, d_ADP, d_ATP, d_PCr, d_AMPK, d_pAMPK, d_AMP_AMPK, d_ADP_AMPK, d_ATP_AMPK, d_AMP_pAMPK, d_ADP_pAMPK, d_ATP_pAMPK, d_CaMKK, d_CaMKK_AMPK, d_CaMKK_AMP_AMPK, d_CaMKK_ADP_AMPK, d_CaMKK_ATP_AMPK, d_LKB1, d_LKB1_AMP_AMPK, d_LKB1_ADP_AMPK, d_PP, d_PP_pAMPK, d_PP_ATP_pAMPK, d_AMPKAR, d_pAMPKAR, d_AMPKAR_AMP_pAMPK, d_PP1, d_PP1_pAMPKAR]


    def set_kGly(self, kGly):
            """Set the glycolysis rate parameter."""
            self.kGly = kGly