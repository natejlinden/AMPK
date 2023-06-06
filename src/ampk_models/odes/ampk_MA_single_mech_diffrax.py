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

class vector_field(eqx.Module):
    """Right hand side of the AMPK_ma_double_mech regulation model.

    Written in the format required by the diffrax package
    """

    # fixed parameters
    # metabolic params
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

        # FLUXES
        # single AXP complexing
        # single AXP complexing
        J1 = kOnAMP*y[0]*y[3] - kOffAMP*y[5] # AMPK
        J2 = kOnADP*y[1]*y[3] - kOffADP*y[6]
        J3 = kOnATP*y[2]*y[3] - kOffATP*y[7]
        J4 = kOnAMP*y[0]*y[4] - kOffAMP*y[8] # pAMPK
        J5 = kOnADP*y[1]*y[4] - kOffADP*y[9]
        J6 = kOnATP*y[2]*y[4] - kOffATP*y[10]
        # CaMKK complexing and phosphorylation
        J7 = kOnCaMKK*y[11]*y[3] - kOffCaMKK*y[12]
        J8 = kPhosCaMKK*y[12]
        J9 = kOnCaMKK*y[11]*y[5] - kOffCaMKK*y[13]  
        J10 = kPhosCaMKK*y[13] 
        J11 = kOnCaMKK*y[11]*y[6] - kOffCaMKK*y[14]
        J12 = kPhosCaMKK*y[14]
        J13 = kOnCaMKK*y[11]*y[7] - kOffCaMKK*y[15]
        J14 = kPhosCaMKK*y[15]
        # LKB1 complexing and phosphorylation
        J15 = kOnLKB1*y[16]*y[5] - kOffLKB1*y[17]
        J16 = kPhosLKB1*y[17]
        J17 = kOnLKB1*y[16]*y[6] -  kOffLKB1*y[18]
        J18 = kPhosLKB1*y[18]
        # phosphatase binding and dephosphorylation
        J19 = kOnPP*y[19]*y[4] - kOffPP*y[20]
        J20 =  kDephosPP*y[20]
        J21 = kOnPP*y[19]*y[10] - kOffPP*y[21]
        J22 =  kDephosPP*y[21]
        # AMPK binding to AMPAKAR and phosphorylation
        J23 = kOnAMPK*y[22]*y[8] - kOffAMPK*y[24]
        J24 = kPhosAMPK*y[24]
        # PP1 binding to AMPKAR and dephosphorylation  
        J25 = kOnPP1*y[25]*y[23] - kOffPP1*y[26]
        J26 = kDephosPP1*y[26]
        # Metabolic fluxes
        # glycolysis
        Jgly = 2*self.kGly*y[1]*y[1]
        # ATP hydrolysis
        Jhydro = self.kHydro*y[2]
        num_for = (self.VforAK*y[2]*y[0])/(self.kmt*self.kmm)
        den = (1 + (y[2]/self.kmt) + (y[0]/self.kmm) + ((y[2]*y[0])/(self.kmt*self.kmm)) + 
                    ((2*y[1])/self.kmd) + ((y[1]**2)/(self.kmd**2)))
        VrevAK = (self.VforAK*(self.kmd**2))/(self.KeqAK*self.kmt*self.kmm)
        num_rev = (VrevAK*(y[1]**2))/(self.kmd**2)
        JAK = (num_for - num_rev)/den
        # Oxidative Phos
        Joxphos = (self.VmaxOxPhos * ((y[1]/self.Kadp)**self.n))/(1 + ((y[1]/self.Kadp)**self.n))



        # now return the odes for each state variable
        dydt = jnp.zeros((27,)) # 53 state variables jax array
        dydt = dydt.at[0].set(-J1-J4-JAK) # AMP
        dydt = dydt.at[1].set(-J2-J5-Jgly+2*JAK+Jhydro-Joxphos) # ADP
        dydt = dydt.at[2].set(-J3-J6+Jgly-JAK-Jhydro+Joxphos) # ATP
        # free AMPK
        dydt = dydt.at[3].set(-J1-J2-J3-J7+J20) # AMPK
        dydt = dydt.at[4].set(-J4-J5-J6+J8-J19) # pAMPK
        # single AXP-AMPK complexes
        dydt = dydt.at[5].set(J1-J9-J15) # AMP_AMPK
        dydt = dydt.at[6].set(J2-J11-J17) # ADP_AMPK
        dydt = dydt.at[7].set(J3-J13+J22) # ATP_AMPK
        # single AXP-pAMPK complexes
        dydt = dydt.at[8].set(J4+J10+J16-J23+J24) # AMP_pAMPK
        dydt = dydt.at[9].set(J5+J12+J18) # ADP_pAMPK
        dydt = dydt.at[10].set(J6+J14-J21) #  ATP_pAMPK
        # CaMKK complexes
        dydt = dydt.at[11].set(-J7+J8-J9+J10-J11+J12-J13+J14) # CaMKK
        dydt = dydt.at[12].set(J7-J8) # CaMKK_AMPK
        dydt = dydt.at[13].set(J9-J10) # CaMKK_AMP_AMPK
        dydt = dydt.at[14].set(J11-J12) # CaMKK_ADP_AMPK
        dydt = dydt.at[15].set(J13-J14) # CaMKK_ATP_AMPK
        # LKB1 complexes
        dydt = dydt.at[16].set(-J15+J16-J17+J18) # LKB1
        dydt = dydt.at[17].set(J15-J16) # LKB1_AMP_AMPK
        dydt = dydt.at[18].set(J17-J18) # LKB1_ADP_AMPK
        # AMPK phosphatase complexes
        dydt = dydt.at[19].set(-J19+J20-J21+J22) # PP
        dydt = dydt.at[20].set(J19-J20) # PP_pAMPK
        dydt = dydt.at[21].set(J21-J22) # PP_ATP_pAMPK
        # free AMPKAR
        dydt = dydt.at[22].set(-J23+J26) # AMPKAR
        dydt = dydt.at[23].set(J24-J25) # pAMPKAR
        # AMPKAR-pAMPK complexes
        dydt = dydt.at[24].set(J23-J24) # pAMPKAR_AMP_pAMPK
        # AMPKAR phosphatase complexes
        dydt = dydt.at[25].set(-J25+J26) # PP1
        dydt = dydt.at[26].set(J25-J26) # PP1_pAMPKAR

        return dydt


    def set_kGly(self, kGly):
            """Set the glycolysis rate parameter."""
            self.kGly = kGly