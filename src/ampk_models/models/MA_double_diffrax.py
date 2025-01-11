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

class MA_double(eqx.Module):
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
        
        # FLUXES
        # single AXP complexing
        J1  = kOnAMP*y[0]*y[3]      - kOffAMP*y[5] # AMPK
        J2  = kOnADP*y[1]*y[3]      - kOffADP*y[6]
        J3  = kOnATP*y[2]*y[3]      - kOffATP*y[7]
        J4  = kOnAMP*y[0]*y[4]      - kOffAMP*y[8] # pAMPK
        J5  = kOnADP*y[1]*y[4]      - kOffADP*y[9]
        J6  = kOnATP*y[2]*y[4]      - kOffATP*y[10]
        # double AXP complexing
        J7  = kOnAMP*y[0]*y[5]      - kOffAMP*y[11] # AMPK
        J8  = kOnAMP*y[0]*y[6]      - kOffAMP*y[12]
        J9  = kOnAMP*y[0]*y[7]      - kOffAMP*y[13]
        J10 = kOnADP*y[1]*y[5]      - kOffADP*y[12]
        J11 = kOnADP*y[1]*y[6]      - kOffADP*y[14]
        J12 = kOnADP*y[1]*y[7]      - kOffADP*y[15]
        J13 = kOnATP*y[2]*y[5]      -  kOffATP*y[13]
        J14 = kOnATP*y[2]*y[6]      - kOffATP*y[15]
        J15 = kOnATP*y[2]*y[7]      - kOffATP*y[16]
        J16 = kOnAMP*y[0]*y[8]      - kOffAMP*y[17] # pAMPK
        J17 = kOnAMP*y[0]*y[9]      - kOffAMP*y[18]
        J18 = kOnAMP*y[0]*y[10]     - kOffAMP*y[19]
        J19 = kOnADP*y[1]*y[8]      - kOffADP*y[18]
        J20 = kOnADP*y[1]*y[9]      - kOffADP*y[20]
        J21 = kOnADP*y[1]*y[10]     - kOffADP*y[21]
        J22 = kOnATP*y[2]*y[8]      - kOffATP*y[19]
        J23 = kOnATP*y[2]*y[9]      - kOffATP*y[21]
        J24 = kOnATP*y[2]*y[10]     - kOffATP*y[22]
        # CaMKK complexing and phosphorylation    
        J25 = kOnCaMKK*y[23]*y[3]   - kOffCaMKK*y[24]
        J26 = kPhosCaMKK*y[24]
        J27 = kOnCaMKK*y[23]*y[5]   - kOffCaMKK*y[25]   
        J28 = kPhosCaMKK*y[25] 
        J29 = kOnCaMKK*y[23]*y[6]   - kOffCaMKK*y[26]   
        J30 = kPhosCaMKK*y[26]  
        J31 = kOnCaMKK*y[23]*y[7]   - kOffCaMKK*y[27]   
        J32 = kPhosCaMKK*y[27] 
        J33 = kOnCaMKK*y[23]*y[11]   - kOffCaMKK*y[28]   
        J34 = kPhosCaMKK*y[28]  
        J35 = kOnCaMKK*y[23]*y[12]  - kOffCaMKK*y[29]   
        J36 = kPhosCaMKK*y[29]  
        J37 = kOnCaMKK*y[23]*y[13]  - kOffCaMKK*y[30]   
        J38 = kPhosCaMKK*y[30]  
        J39 = kOnCaMKK*y[23]*y[14]  - kOffCaMKK*y[31]   
        J40 = kPhosCaMKK*y[31]  
        J41 = kOnCaMKK*y[23]*y[15]  - kOffCaMKK*y[32]   
        J42 = kPhosCaMKK*y[32]  
        J43 = kOnCaMKK*y[23]*y[16]  - kOffCaMKK*y[33]   
        J44 = kPhosCaMKK*y[33]
        # LKB1 complexing and phosphorylation
        J45 = kOnLKB1*y[34]*y[5]    - kOffLKB1*y[35]   
        J46 = kPhosLKB1*y[35]  
        J47 = kOnLKB1*y[34]*y[6]    - kOffLKB1*y[36]   
        J48 = kPhosLKB1*y[36]  
        J49 = kOnLKB1*y[34]*y[11]   - kOffLKB1*y[37]   
        J50 = kPhosLKB1*y[37]  
        J51 = kOnLKB1*y[34]*y[12]   - kOffLKB1*y[38]   
        J52 = kPhosLKB1*y[38]  
        J53 = kOnLKB1*y[34]*y[14]   - kOffLKB1*y[39]   
        J54 = kPhosLKB1*y[39]  
        # phosphatase binding and dephosphorylation
        J55 = kOnPP*y[40]*y[4]      - kOffPP*y[41]    
        J56 = kDephosPP*y[41]
        J57 = kOnPP*y[40]*y[10]     - kOffPP*y[42]    
        J58 = kDephosPP*y[42]
        J59 = kOnPP*y[40]*y[19]     - kOffPP*y[43]    
        J60 = kDephosPP*y[43]
        J61 = kOnPP*y[40]*y[21]     - kOffPP*y[44]    
        J62 = kDephosPP*y[44]
        J63 = kOnPP*y[40]*y[22]     - kOffPP*y[45]
        J64 = kDephosPP*y[45]
        # AMPK binding to AMPAKAR and phosphorylation
        J65 = kOnAMPK*y[46]*y[8]    - kOffAMPK*y[48]   
        J66 = kPhosAMPK*y[48]  
        J67 = kOnAMPK*y[46]*y[17]   - kOffAMPK*y[49]   
        J68 = kPhosAMPK*y[49]  
        J69 = kOnAMPK*y[46]*y[18]   - kOffAMPK*y[50]   
        J70 = kPhosAMPK*y[50]
        # PP1 binding to AMPKAR and dephosphorylation  
        J71 = kOnPP1*y[51]*y[47]    - kOffPP1*y[52]
        J72 = kDephosPP1*y[52]
        
        
        # Metabolic fluxes
        # glycolysis
        Jgly = self.kGly*y[1] #2*kGly*y[1]*y[1]
        # ATP hydrolysis
        Jhydro = self.kHydro*y[2]
        # Adenylate Kinase
        # written as (VforAK*y[2])/(kmt*kmm) in cocci, but units dont make sense
        num_for = (self.VforAK*y[2]*y[0])/(self.kmt*self.kmm)
        den_ak = (1 + (y[2]/self.kmt) + (y[0]/self.kmm) + ((y[2]*y[0])/(self.kmt*self.kmm)) + 
                    ((2*y[1])/self.kmd) + ((y[1]**2)/(self.kmd**2)))
        VrevAK = (self.VforAK*(self.kmd**2))/(self.KeqAK*self.kmt*self.kmm)
        num_rev = (VrevAK*(y[1]**2))/(self.kmd**2)
        JAK = (num_for - num_rev)/den_ak # y[1] forming direction 
        # Oxidative Phos
        Joxphos = (self.VmaxOxPhos * ((y[1]/self.Kadp)**self.n))/(1 + ((y[1]/self.Kadp)**self.n))
        # Creatine kinase
        den_ck = 1 + (y[1]/self.Kia) + (y[53]/self.Kib) + (y[2]/self.Kiq) + ((y[1]*y[53])/(self.Kia*self.Kb)) + (((self.TCr - y[53])*y[2])/(self.Kiq*self.Kp))
        num_forCK = ((self.VforCK*y[1]*y[53])/(self.Kia*self.Kb))
        VrevCK = (self.VforCK*self.Kiq*self.Kp)/(self.KeqCK*self.Kia*self.Kb)
        num_revCK = ((VrevCK*y[2]*(self.TCr - y[53]))/(self.Kiq*self.Kp))
        JCK = (num_revCK - num_forCK)/den_ck # Pi forming direction

        # now return the odes for each state variable
        return [-J1-J4-J7-J8-J9-J16-J17-J18-JAK,
        -J2-J5-J10-J11-J12-J19-J20-J21-Jgly+2*JAK+Jhydro-Joxphos,
        -J3-J6-J13-J14-J15-J22-J23-J24+Jgly-JAK-Jhydro+Joxphos,
        -J1-J2-J3-J25+J56,
        -J4-J5-J6+J26-J55,
        J1-J7-J10-J13-J27-J45,
        J2-J8-J11-J14-J29-J47,
        J3-J9-J12-J15-J31+J58,
        J4-J16-J19-J22+J28+J46-J65+J66,
        J5-J17-J20-J23+J30+J48,
        J6-J18-J21-J24+J32-J57,
        J7-J33-J49,
        J8+J10-J35-J51,
        J9+J13-J37+J60,
        J11-J39-J53,
        J12+J14-J41+J62,
        J15-J43+J64,
        J16+J34+J50-J67+J68,
        J17+J19+J36+J52-J69+J70,
        J18+J22+J38-J59,
        J20+J40+J54,
        J21+J23+J42-J61,
        J24+J44-J63,
        -J25+J26-J27+J28-J29+J30-J31+J32-J33+J34-J35+J36
                               -J37+J38-J39+J40-J41+J42-J43+J44,
        J25-J26,
        J27-J28,
        J29-J30,
        J31-J32,
        J33-J34,
        J35-J36,
        J37-J38,
        J39-J40,
        J41-J42,
        J43-J44,
        -J45+J46-J47+J48-J49+J50-J51+J52-J53+J54,
        J45-J46,
        J47-J48,
        J49-J50,
        J51-J52,
        J53-J54,
        -J55+J56-J57+J58-J59+J60-J61+J62-J63+J64,
        J55-J56,
        J57-J58,
        J59-J60,
        J61-J62,
        J63-J64,
        -J65-J67-J69+J72,
        J66+J68+J70-J71,
        J65-J66,
        J67-J68,
        J69-J70,
        J71+J72,
        J71-J72,
        JCK]


    def set_kGly(self, kGly):
            """Set the glycolysis rate parameter."""
            self.kGly = kGly