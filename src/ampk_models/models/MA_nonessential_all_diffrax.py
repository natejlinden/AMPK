"""
    - AMP/ADP binding protect from pAMPK dephosphorylation 
        by a factor of alphaPP < 1.0
    - AMP binding promotes pAMPK activity by a factor of betaAMP > 1.0
    - AMP/ADP binding promotes LKB1/CaMKK activity by a factor of betaKinase > 1.0

"""
import jax.numpy as jnp
import equinox as eqx

class MA_nonessential_all(eqx.Module):
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
        # updated params for Ca -> CaCaM -> CaMKK activation
        kOnCaM      = args[21] # Ca binding
        kOffCaM     = args[22]
        kPhosCaM    = args[23] # CaMKK phosphorylation (using MM here b/c refs)
        KmCaM      = args[24] # Km for CaMKK activation
        kDephosCaMKK = args[25] # dephosphorylation of CaMKK
        alphaPP = args[26] # < 1; dephos reduction factor due to AMP/ADP
        betaAMP = args[27] # >1 pAMPK phos increase factor due to AMP 
        betaKinase = args[28] # >1 AMPK phos increase factor due to AMP/ADP

        # unpack states
        AMP                = y[0]
        ADP                = y[1]
        ATP                = y[2]
        PCr                = y[3]
        Ca                 = y[4] # free calcium
        AMPK               = y[5]
        pAMPK              = y[6]
        AMP_AMPK           = y[7]
        ADP_AMPK           = y[8]
        ATP_AMPK           = y[9]
        AMP_pAMPK          = y[10]
        ADP_pAMPK          = y[11]
        ATP_pAMPK          = y[12]
        CaM                = y[13] # calmodulin
        CaCaM              = y[14]
        CaMKK              = y[15]
        CaMKK_act          = y[16]
        CaMKK_act_AMPK     = y[17] # only active CaMKK can bind AMPK
        CaMKK_act_AMP_AMPK = y[18]
        CaMKK_act_ADP_AMPK = y[19]
        CaMKK_act_ATP_AMPK = y[20]
        LKB1               = y[21]
        LKB1_AMPK          = y[22]
        LKB1_AMP_AMPK      = y[23]
        LKB1_ADP_AMPK      = y[24]
        LKB1_ATP_AMPK      = y[25]
        PP                 = y[26] # generic phosphatase that acts on AMPK
        PP_pAMPK           = y[27]
        PP_AMP_pAMPK       = y[28]
        PP_ADP_pAMPK       = y[29]
        PP_ATP_pAMPK       = y[30]
        AMPKAR             = y[31] # AMPK activity sensor
        pAMPKAR            = y[32]
        AMPKAR_pAMPK       = y[33]
        AMPKAR_AMP_pAMPK   = y[34]
        AMPKAR_ADP_pAMPK   = y[35]
        AMPKAR_ATP_pAMPK   = y[36]
        PP1                = y[37]
        PP1_pAMPKAR        = y[38]

        # FLUXES
        # single AXP complexing
        J1 = kOnAMP*AMP*AMPK - kOffAMP*AMP_AMPK # AMPK
        J2 = kOnADP*ADP*AMPK - kOffADP*ADP_AMPK
        J3 = kOnATP*ATP*AMPK - kOffATP*ATP_AMPK
        J4 = kOnAMP*AMP*pAMPK - kOffAMP*AMP_pAMPK # pAMPK
        J5 = kOnADP*ADP*pAMPK - kOffADP*ADP_pAMPK
        J6 = kOnATP*ATP*pAMPK - kOffATP*ATP_pAMPK

        # Ca -> CaM -> CaMKK activation
        JCa = kOnCaM*(Ca**3)*CaM - kOffCaM*CaCaM
        JCaMKK_act = (kPhosCaM*(CaCaM**4)*CaMKK)/(KmCaM**4 + CaCaM**4) # CaMKK activation
        JCaMKK_dephos = kDephosCaMKK*CaMKK_act

        # CaMKK complexing and phosphorylation
        J7 = kOnCaMKK*CaMKK_act*AMPK - kOffCaMKK*CaMKK_act_AMPK
        J8 = kPhosCaMKK*CaMKK_act_AMPK
        J9 = kOnCaMKK*CaMKK_act*AMP_AMPK - kOffCaMKK*CaMKK_act_AMP_AMPK  
        J10 = betaKinase*kPhosCaMKK*CaMKK_act_AMP_AMPK 
        J11 = kOnCaMKK*CaMKK_act*ADP_AMPK - kOffCaMKK*CaMKK_act_ADP_AMPK
        J12 = betaKinase*kPhosCaMKK*CaMKK_act_ADP_AMPK
        J13 = kOnCaMKK*CaMKK_act*ATP_AMPK - kOffCaMKK*CaMKK_act_ATP_AMPK
        J14 = kPhosCaMKK*CaMKK_act_ATP_AMPK

        # LKB1 complexing and phosphorylation
        J15 = kOnLKB1*LKB1*AMP_AMPK - kOffLKB1*LKB1_AMP_AMPK
        J16 = betaKinase*kPhosLKB1*LKB1_AMP_AMPK
        J17 = kOnLKB1*LKB1*ADP_AMPK -  kOffLKB1*LKB1_ADP_AMPK
        J18 = betaKinase*kPhosLKB1*LKB1_ADP_AMPK
        J19 = kOnLKB1*LKB1*ATP_AMPK -  kOffLKB1*LKB1_ATP_AMPK
        J20 = kPhosLKB1*LKB1_ATP_AMPK
        J21 = kOnLKB1*LKB1*AMPK -  kOffLKB1*LKB1_AMPK
        J22 = kPhosLKB1*LKB1_AMPK
        
        # phosphatase binding and dephosphorylation
        J23 = kOnPP*PP*pAMPK - kOffPP*PP_pAMPK
        J24 = kDephosPP*PP_pAMPK
        J25 = kOnPP*PP*ATP_pAMPK - kOffPP*PP_ATP_pAMPK
        J26 =  kDephosPP*PP_ATP_pAMPK
        J27 = kOnPP*PP*AMP_pAMPK - kOffPP*PP_AMP_pAMPK
        J28 = alphaPP*kDephosPP*PP_AMP_pAMPK
        J29 = kOnPP*PP*ADP_pAMPK - kOffPP*PP_ADP_pAMPK
        J30 = alphaPP*kDephosPP*PP_ADP_pAMPK

        # AMPK binding to AMPAKAR and phosphorylation
        J31 = kOnAMPK*AMPKAR*AMP_pAMPK - kOffAMPK*AMPKAR_AMP_pAMPK
        J32 = betaAMP*kPhosAMPK*AMPKAR_AMP_pAMPK
        J33 = kOnAMPK*AMPKAR*ADP_pAMPK - kOffAMPK*AMPKAR_ADP_pAMPK
        J34 = kPhosAMPK*AMPKAR_ADP_pAMPK
        J35 = kOnAMPK*AMPKAR*ATP_pAMPK - kOffAMPK*AMPKAR_ATP_pAMPK
        J36 = kPhosAMPK*AMPKAR_ATP_pAMPK
        J37 = kOnAMPK*AMPKAR*pAMPK - kOffAMPK*AMPKAR_pAMPK
        J38 = kPhosAMPK*AMPKAR_pAMPK

        # PP1 binding to AMPKAR and dephosphorylation  
        J39 = kOnPP1*PP1*pAMPKAR - kOffPP1*PP1_pAMPKAR
        J40 = kDephosPP1*PP1_pAMPKAR

        # additional fluxes to allow AXP to bind/unbind enzyme--AMPK complexes
        Ja = kOnAMP*AMP*CaMKK_act_AMPK - kOffADP*CaMKK_act_AMP_AMPK
        Jb = kOnADP*ADP*CaMKK_act_AMPK - kOffADP*CaMKK_act_ADP_AMPK
        Jc = kOnATP*ATP*CaMKK_act_AMPK - kOffATP*CaMKK_act_ATP_AMPK
        Jd = kOnAMP*AMP*LKB1_AMPK - kOffAMP*LKB1_AMP_AMPK
        Je = kOnADP*ADP*LKB1_AMPK - kOffADP*LKB1_ADP_AMPK
        Jf = kOnATP*ATP*LKB1_AMPK - kOffATP*LKB1_ATP_AMPK
        Jg = kOnAMP*AMP*PP_pAMPK - kOffAMP*PP_AMP_pAMPK
        Jh = kOnADP*ADP*PP_pAMPK - kOffADP*PP_ADP_pAMPK
        Ji = kOnATP*ATP*PP_pAMPK - kOffATP*PP_ATP_pAMPK
        Jj = kOnAMP*AMP*AMPKAR_pAMPK - kOffAMP*AMPKAR_AMP_pAMPK
        Jk = kOnADP*ADP*AMPKAR_pAMPK - kOffADP*AMPKAR_ADP_pAMPK
        Jl = kOnATP*ATP*AMPKAR_pAMPK - kOffATP*AMPKAR_ATP_pAMPK

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
        d_AMP = -J1-J4-JAK-Ja-Jd-Jg-Jj # AMP
        d_ADP = -J2-J5-Jgly+2*JAK+Jhydro-Joxphos+JCK-Jb-Je-Jh-Jk # ADP
        d_ATP = -J3-J6+Jgly-JAK-Jhydro+Joxphos-JCK-Jc-Jf-Ji-Jl # ATP
        d_PCr = JCK
        # free AMPK
        d_AMPK = -J1-J2-J3-J7+J24-J21 # AMPK
        d_pAMPK = -J4-J5-J6+J8-J23+J22-J37+J38 # pAMPK
        # single AXP-AMPK complexes
        d_AMP_AMPK = J1-J9-J15+J28 # AMP_AMPK
        d_ADP_AMPK = J2-J11-J17+J30 # ADP_AMPK
        d_ATP_AMPK = J3-J13+J26-J19 # ATP_AMPK
        # single AXP-pAMPK complexes
        d_AMP_pAMPK = J4+J10+J16-J31+J32-J27 # AMP_pAMPK
        d_ADP_pAMPK = J5+J12+J18-J30-J33+J34 # ADP_pAMPK
        d_ATP_pAMPK = J6+J14-J25+J20-J35+J36 #  ATP_pAMPK
        # CaMKK complexes
        d_Ca = -JCa
        d_CaM = -JCa
        d_CaCaM = JCa
        d_CaMKK = -JCaMKK_act + JCaMKK_dephos # CaMKK
        d_CaMKK_act = JCaMKK_act - JCaMKK_dephos-J7+J8-J9+J10-J11+J12-J13+J14 # CaMKK_act
        d_CaMKK_act_AMPK = J7-J8-Ja-Jb-Jc # CaMKK_AMPK
        d_CaMKK_act_AMP_AMPK = J9-J10+Ja # CaMKK_act_AMP_AMPK
        d_CaMKK_act_ADP_AMPK = J11-J12+Jb # CaMKK_act_ADP_AMPK
        d_CaMKK_act_ATP_AMPK = J13-J14+Jc # CaMKK_act_ATP_AMPK

        # LKB1 complexes
        d_LKB1 = -J15+J16-J17+J18-J19+J20-J21+J22 # LKB1
        d_LKB1_AMP_AMPK = J15-J16+Jd # LKB1_AMP_AMPK
        d_LKB1_ADP_AMPK = J17-J18+Je # LKB1_ADP_AMPK
        d_LKB1_ATP_AMPK = J19-J20+Jf # LKB1_ATP_AMPK
        d_LKB1_AMPK = J21-J22-Jd-Je-Jf # LKB1_AMPK

        # AMPK phosphatase complexes
        d_PP = -J23+J24-J25+J26-J27+J28-J29+J30 # PP
        d_PP_pAMPK = J23-J24-Jg-Ji-Jh # PP_pAMPK
        d_PP_ATP_pAMPK = J25-J26+Ji # PP_ATP_pAMPK
        d_PP_AMP_pAMPK = J27-J28+Jg # PP_AMP_pAMPK
        d_PP_ADP_pAMPK = J29-J30+Jh # PP_ADP_pAMPK

        # free AMPKAR
        d_AMPKAR = -J31+J40-J33-J35-J37 # AMPKAR
        d_pAMPKAR = J32-J39+J34+J36+J38 # pAMPKAR

        # AMPKAR-pAMPK complexes
        d_AMPKAR_pAMPK = J37-J38-Jj-Jk-Jl # AMPKAR_pAMPK
        d_AMPKAR_AMP_pAMPK = J31-J32+Jj # AMPKAR_AMP_pAMPK
        d_AMPKAR_ADP_pAMPK = J33-J34+Jk
        d_AMPKAR_ATP_pAMPK = J35-J36+Jl

        # AMPKAR phosphatase complexes
        d_PP1 = -J39+J40 # PP1
        d_PP1_pAMPKAR = J39-J40 # PP1_pAMPKAR

        return [d_AMP, d_ADP, d_ATP, d_PCr, d_Ca, 
                d_AMPK, d_pAMPK, d_AMP_AMPK, d_ADP_AMPK, d_ATP_AMPK, d_AMP_pAMPK, d_ADP_pAMPK, d_ATP_pAMPK, 
                d_CaM, d_CaCaM, d_CaMKK, d_CaMKK_act, d_CaMKK_act_AMPK, 
                d_CaMKK_act_AMP_AMPK, d_CaMKK_act_ADP_AMPK, d_CaMKK_act_ATP_AMPK, 
                d_LKB1, d_LKB1_AMPK, d_LKB1_AMP_AMPK, d_LKB1_ADP_AMPK, d_LKB1_ATP_AMPK,
                d_PP, d_PP_pAMPK, d_PP_AMP_pAMPK, d_PP_ADP_pAMPK, d_PP_ATP_pAMPK,
                d_AMPKAR, d_pAMPKAR, d_AMPKAR_pAMPK,
                d_AMPKAR_AMP_pAMPK, d_AMPKAR_ADP_pAMPK, d_AMPKAR_ATP_pAMPK,
                d_PP1, d_PP1_pAMPKAR]

    def set_kGly(self, kGly):
            """Set the glycolysis rate parameter."""
            self.kGly = kGly