"""
    Nathaniel Linden (UCSD MAE)
    Created: Oct 29, 2024
"""
import jax.numpy as jnp
import equinox as eqx

class ampk_Coccimiglio(eqx.Module):
    """Right hand side of the Coccimiglio et al AMPK model.

    From: Ian F.Coccimiglio ID and David C. ClarkeID. "ADP is the dominant controller of AMP-activated protein kinase activity dynamics in skeletal muscle during exercise." PLOS Computational Biology. 2020. 6(7): e1008079.

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
        # unpack parameters
        k6f           = args[0]
        k6r           = args[1]
        k7f           = args[2]
        k7r           = args[3]
        k8f           = args[4]
        k8r           = args[5]
        k9f           = args[6]
        k9r           = args[7]
        k10f          = args[8]
        k10r          = args[9]
        k11f          = args[10]
        k11r          = args[11]
        Km12          = args[12]
        Km13          = args[13]
        Km14          = args[14]
        Km15          = args[15]
        Km16          = args[16]
        Km17          = args[17]
        Km18          = args[18]
        Km19          = args[19]
        Vmaxkinase    = args[20]
        VmaxkinaseATP = args[21]
        VmaxkinaseADP = args[22]
        VmaxkinaseAMP = args[23]
        Vmaxppase     = args[24]
        VmaxppaseATP  = args[25]
        VmaxppaseADP  = args[26]
        VmaxppaseAMP  = args[27]
        # added parameters for AMPK phos of AMPKAR
        Km_pAMPK        = args[28]
        k_pAMPK      = args[29]
        Km_AMP_pAMPK    = args[30]
        k_AMP_pAMPK  = args[31]
        Km_ADP_pAMPK    = args[32]
        k_ADP_pAMPK  = args[33]
        Km_ATP_pAMPK    = args[34]
        k_ATP_pAMPK  = args[35]
        Km_AMPKAR_PP    = args[36]
        Vmax_AMPKAR_PP  = args[37]

        # state variables
        ATP = y[0]
        ADP = y[1]
        AMP = y[2]
        PCr = y[3]
        Pi = y[4]
        ATP_AMPK = y[5]
        ADP_AMPK = y[6]
        AMP_AMPK = y[7]
        ATP_p_AMPK = y[8]
        ADP_p_AMPK = y[9]
        AMP_p_AMPK = y[10]
        AMPK = y[11]
        p_AMPK = y[12]
        # added state variables for AMPK phos of AMPKAR
        AMPKAR = y[13]
        p_AMPKAR = y[14]
        
        # FLUXES
        r6  = k6f*ATP*AMPK              - k6r*ATP_AMPK # ATP binding to AMPK
        r7  = k7f*ADP*AMPK              - k7r*ADP_AMPK # ADP binding to AMPK
        r8  = k8f*AMP*AMPK              - k8r*AMP_AMPK # AMP binding to AMPK
        r9  = k9f*ATP*p_AMPK            - k9r*ATP_p_AMPK # ATP binding to p_AMPK
        r10 = k10f*ADP*p_AMPK           - k10r*ADP_p_AMPK # ADP binding to p_AMPK
        r11 = k11f*AMP*p_AMPK           - k11r*AMP_p_AMPK # AMP binding to p_AMPK
        r12 = (Vmaxkinase*AMPK)         / (Km12 + AMPK) # AMPK phosphorylation
        r13 = (Vmaxppase*p_AMPK)        / (Km13 + p_AMPK) # p_AMPK dephosphorylation
        r14 = (VmaxkinaseATP*ATP_AMPK)  / (Km14 + ATP_AMPK) # ATP_AMPK phosphorylation
        r15 = (VmaxppaseATP*ATP_p_AMPK) / (Km15 + ATP_p_AMPK) # ATP_p_AMPK dephosphorylation
        r16 = (VmaxkinaseADP*ADP_AMPK)  / (Km16 + ADP_AMPK) # ADP_AMPK phosphorylation
        r17 = (VmaxppaseADP*ADP_p_AMPK) / (Km17 + ADP_p_AMPK) # ADP_p_AMPK dephosphorylation
        r18 = (VmaxkinaseAMP*AMP_AMPK)  / (Km18 + AMP_AMPK) # AMP_AMPK phosphorylation
        r19 = (VmaxppaseAMP*AMP_p_AMPK) / (Km19 + AMP_p_AMPK) # AMP_p_AMPK dephosphorylation
        # added fluxes for AMPK phos of AMPKAR
        r20 = (k_pAMPK*AMPKAR*p_AMPK)             / (Km_pAMPK + p_AMPK)
        r21 = (k_AMP_pAMPK*AMPKAR*AMP_p_AMPK)     / (Km_AMP_pAMPK + AMP_p_AMPK)
        r22 = (k_ADP_pAMPK*AMPKAR*ADP_p_AMPK)     / (Km_ADP_pAMPK + ADP_p_AMPK)
        r23 = (k_ATP_pAMPK*AMPKAR*ATP_p_AMPK)     / (Km_ATP_pAMPK + ATP_p_AMPK)
        r24 = (Vmax_AMPKAR_PP*p_AMPKAR)           / (Km_AMPKAR_PP + p_AMPKAR)
        
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
        d_ATP = Jgly - JAK - Jhydro + Joxphos - JCK - r6 - r9
        d_ADP = -Jgly + 2*JAK + Jhydro - Joxphos + JCK - r7 - r10
        d_AMP = -JAK -r8 - r11
        d_PCr = JCK
        d_Pi = -Jgly + Joxphos
        d_ATP_AMPK = r6 - r14 + r15
        d_ADP_AMPK = r7 - r16 + r17
        d_AMP_AMPK = r8 - r18 + r19
        d_ATP_p_AMPK = r9 - r14 + r15
        d_ADP_p_AMPK = r10 - r16 + r17
        d_AMP_p_AMPK = r11 - r18 + r19
        d_AMPK = -r6 - r7 - r8 - r12 + r13
        d_p_AMPK = -r9 - r10 - r11 + r12 - r13
        d_AMPKAR = -r20 - r21 - r22 - r23 + r24
        d_p_AMPKAR = r20 + r21 + r22 + r23 - r24

        return jnp.array([d_ATP, d_ADP, d_AMP, d_PCr, d_Pi, d_ATP_AMPK, 
                          d_ADP_AMPK, d_AMP_AMPK, d_ATP_p_AMPK, d_ADP_p_AMPK, 
                          d_AMP_p_AMPK, d_AMPK, d_p_AMPK, d_AMPKAR, d_p_AMPKAR])

    def set_kGly(self, kGly):
            """Set the glycolysis rate parameter."""
            self.kGly = kGly
