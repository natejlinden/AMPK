import numpy as np

def Henri_Michaelis_Menten_irreversible_(substrate, Km, Vmax):
    """Michaelis Menten Kinetics
    
    This function evaluates the reversible Michaelis-Menten eqn.
    """
    return (Vmax * substrate) /(Km + substrate)

def FunctionForOxidativePhosphorylation(substrate, K_ADP_OxPhos, Vmax,  n_OxPhos):
    """Custom function to represent oxidative phosphorylation
    
    This function captures oxidative phosphorylation in a single equation.
    """
    return (Vmax * ((substrate/K_ADP_OxPhos)**n_OxPhos)) / (1.0 + ((substrate/K_ADP_OxPhos)**n_OxPhos))

def ampk_rhs(t, x, p):
    """Function for the ODE of the MAPK model developed by Allen Lueng. 
    
    This function evaluates the right-hand-side of the ODE model. It is based-on
    the model in COPASI file: mito_cyto.cps.
    
    Parameters
    ----------
    t : float
        time 
    x : numpy array of floats, shape=()
    current state at time t. y has the following organization:
        x = ["ADP", "ADP_m", "p_AMPK", "AMPK_m", "AMPK", "p_AMPK_m", "AMP_AMPK", "AMP_AMPK_m", "ATP_p_AMPK", "ATP_p_AMPK_m", "ADP_AMPK", "ADP_AMPK_m", "AMP", "AMP_p_AMPK_m", "ADP_p_AMPK", "ADP_p_AMPK_m", "ATP_AMPK", "ATP_m", "ADP_lysosome", "AMP_lysosome", "ATP_lysosome", "Pi"]

    p : numpy array of floats, shape=()
        parameter vector with the following ordering
        p = ["glucose", "pyruvate", "pyruvate_m", "lysosome", "cytosol", "mitochondria", "mito_cyto", "n", "total_AMPK", "n1", "_T_", "_PI_", "_F_", "_F_nmol_", "_N_pmol_", "_K_GHK_", "_R_", "K_millivolts_per_volt", "Kr_rate_of_ATP_hydrolysis", "Kf_ATP_binding_AMPK", "Kr_ATP_binding_AMPK", "Kf_ADP_binding_AMPK", "Kr_ADP_binding_AMPK", "Kf_AMP_binding_AMPK", "Kr_AMP_binding_AMPK", "Kf_ATP_binding_pAMPK", "Kr_ATP_binding_pAMPK", "Kf_ADP_binding_pAMPK", "Kr_ADP_binding_pAMPK", "Kf_AMP_binding_pAMPK", "Kr_AMP_binding_pAMPK", "Km_phosphorylation_of_AMPK", "Vmax_phosphorylation_of_AMPK", "Km_dephosphorylation_of_AMPK", "Vmax_dephosphorylation_of_AMPK", "Km_phosphorylation_of_ATP_AMPK", "Vmax_phosphorylation_of_ATP_AMPK", "Km_phosphorylation_of_AMP_AMPK", "Vmax_phosphorylation_of_AMP_AMPK", "Km_phosphorylation_of_ADP_AMPK", "Vmax_phosphorylation_of_ADP_AMPK", "Km_dephosphorylation_of_ATP_p_AMPK", "Vmax_dephosphorylation_of_ATP_p_AMPK", "Km_dephosphorylation_of_ADP_p_AMPK", "Vmax_dephosphorylation_of_ADP_p_AMPK", "Km_dephosphorylation_of_AMP_p_AMPK", "Vmax_dephosphorylation_of_AMP_p_AMPK", "Kr_glycolysis", "XAK_simple_adenylate_kinase", "KAK_simple_adenylate_kinase", "KADP_Oxidative_Phosphorylation", "Vmaxoxphos_Oxidative_Phosphorylation", "nH_Oxidative_Phosphorylation", "Kf_ANT", "Kr_ANT", "Kf_pyruvate_transfer", "Kr_pyruvate_transfer", "Kf_basal_ATP_production", "Kr_basal_ATP_production", "activity_scale", "baseline_activity", "k1", "k2", "k1", "k2", ct_AMPK_m, ct_AMP_p_AMPK,ct_ATP, ct_AMP_m]

    Returns
    -------
    dxdt : numpy array of floats, same shape as x
        time derivative of x
    
    """

    # algebraic expressions: reactions
    y = np.zeros(15)
    y[0] = p[65]-x[3]-x[5]-x[7]-x[9]-1*x[11]-x[13]-x[15]	#metabolite 'ATP_AMPK_m': reactions
    y[1] = p[66]-1*x[2]-x[4]-1*x[6]-1*x[8]-1*x[10]-x[14]-1*x[16]	#metabolite 'AMP_p_AMPK': reactions
    y[2] = p[67]-x[0]+x[2]+x[4]-x[12]	#metabolite 'ATP': reactions
    y[3] = p[68]-x[1]+x[3]+x[5]-x[17]	#metabolite 'AMP_m': reactions
    y[4] = 0.05000000000000000*p[7]	#model entity 'Kf_rate_of_ATP_hydrolysis':assignment
    y[5] = 0.00498161634948818*p[9]	#model entity 'Kf_glycolysis':assignment
    y[6] = p[48]*p[49]	#model entity 'Kf_simple_adenylate_kinase':assignment
    y[7] = p[48]	#model entity 'Kr_simple_adenylate_kinase':assignment

    #  algebraic expressions: scaled concentrations
    x_c = np.zeros(22)
    p_c = np.zeros(3)
    y_c = np.zeros(4)
    x_c[0] = x[0]/p[4]	#concentration of metabolite 'ADP': reactions
    x_c[1] = x[1]/p[5]	#concentration of metabolite 'ADP_m': reactions
    x_c[2] = x[2]/p[4]	#concentration of metabolite 'p_AMPK': reactions
    x_c[3] = x[3]/p[5]	#concentration of metabolite 'AMPK_m': reactions
    x_c[4] = x[4]/p[4]	#concentration of metabolite 'AMPK': reactions
    x_c[5] = x[5]/p[5]	#concentration of metabolite 'p_AMPK_m': reactions
    x_c[6] = x[6]/p[4]	#concentration of metabolite 'AMP_AMPK': reactions
    x_c[7] = x[7]/p[5]	#concentration of metabolite 'AMP_AMPK_m': reactions
    x_c[8] = x[8]/p[4]	#concentration of metabolite 'ATP_p_AMPK': reactions
    x_c[9] = x[9]/p[5]	#concentration of metabolite 'ATP_p_AMPK_m': reactions
    x_c[10] = x[10]/p[4]	#concentration of metabolite 'ADP_AMPK': reactions
    x_c[11] = x[11]/p[5]	#concentration of metabolite 'ADP_AMPK_m': reactions
    x_c[12] = x[12]/p[4]	#concentration of metabolite 'AMP': reactions
    x_c[13] = x[13]/p[5]	#concentration of metabolite 'AMP_p_AMPK_m': reactions
    x_c[14] = x[14]/p[4]	#concentration of metabolite 'ADP_p_AMPK': reactions
    x_c[15] = x[15]/p[5]	#concentration of metabolite 'ADP_p_AMPK_m': reactions
    x_c[16] = x[16]/p[4]	#concentration of metabolite 'ATP_AMPK': reactions
    x_c[17] = x[17]/p[5]	#concentration of metabolite 'ATP_m': reactions
    x_c[18] = x[18]/p[3]	#concentration of metabolite 'ADP_lysosome': reactions
    x_c[19] = x[19]/p[3]	#concentration of metabolite 'AMP_lysosome': reactions
    x_c[20] = x[20]/p[3]	#concentration of metabolite 'ATP_lysosome': reactions
    x_c[21] = x[21]/p[4]	#concentration of metabolite 'Pi': reactions

    y_c[0] = y[0]/p[5]	#concentration of metabolite 'ATP_AMPK_m': reactions
    y_c[1] = y[1]/p[4]	#concentration of metabolite 'AMP_p_AMPK': reactions
    y_c[2] = y[2]/p[4]	#concentration of metabolite 'ATP': reactions
    y_c[3] = y[3]/p[5]	#concentration of metabolite 'AMP_m': reactions
    
    p_c[0] = p[0]/p[4]	#concentration of metabolite 'glucose': fixed
    p_c[1] = p[1]/p[4]	#concentration of metabolite 'pyruvate': fixed
    p_c[2] = p[2]/p[5]	#concentration of metabolite 'pyruvate_m': fixed

    y[8] = p[60]+(y_c[1]+x_c[14]+x_c[8]+x_c[2])/(x_c[6]+x_c[10]+x_c[16]+x_c[4])*p[59]	#model entity 'FitValue':assignment
    y[9] = p[60]+(x_c[5]+x_c[15]+x_c[13]+x_c[9])/(x_c[3]+x_c[7]+x_c[11]+y_c[0])*p[59]	#model entity 'mito_fitval':assignment
    y[10] = x_c[2]+y_c[1]+x_c[14]+x_c[8]	#model entity 'All pAMPK':assignment
    y[11] = x_c[4]+x_c[6]+x_c[10]+x_c[16]	#model entity 'All AMPK':assignment
    y[12] = x_c[5]+x_c[13]+x_c[15]+x_c[9]	#model entity 'All pAMPK mito':assignment
    y[13] = x_c[3]+x_c[7]+x_c[11]+y_c[0]	#model entity 'All AMPK mito':assignment
    y[14] = y[13]+y[11]+y[12]+y[10]	#model entity 'total AMPK_pools':assignment
    

    # define output variable
    dx = np.zeros(shape=x.shape)
    dx[0] = (y[4] * y_c[2]) *p[4]-(p[21] * x_c[4] * x_c[0] - p[22] * x_c[10]) *p[4]-(p[27] * x_c[2] * x_c[0] - p[28] * x_c[14]) *p[4]-(y[5] * x_c[0] * p_c[0]) *p[4]-2*(p[61] * x_c[0] * x_c[0] - p[62] * x_c[12] * y_c[2]) *p[4]-(p[57] * x_c[0]) *p[4]
    dx[1] = -FunctionForOxidativePhosphorylation(p_c[2], p[50], p[51], p[52])*p[5]-(p[21] * x_c[3] * x_c[1] - p[22] * x_c[11]) *p[5]-(p[27] * x_c[5] * x_c[1] - p[28] * x_c[15]) *p[5]-2*(p[63] * x_c[1] * x_c[1] - p[64] * y_c[3] * x_c[17]) *p[5]
    dx[2] = -(p[25] * x_c[2] * y_c[2] - p[26] * x_c[8]) *p[4]-(p[27] * x_c[2] * x_c[0] - p[28] * x_c[14]) *p[4]-(p[29] * x_c[2] * x_c[12] - p[30] * y_c[1]) *p[4]+Henri_Michaelis_Menten_irreversible_(x_c[4], p[31], p[32])*p[4]-Henri_Michaelis_Menten_irreversible_(x_c[2], p[33], p[34])*p[4]
    dx[3] = -(p[25] * x_c[3] * x_c[17] - p[20] * y_c[0]) *p[5]-(p[21] * x_c[3] * x_c[1] - p[22] * x_c[11]) *p[5]-(p[23] * y_c[3] * x_c[3] - p[24] * x_c[7]) *p[5]+Henri_Michaelis_Menten_irreversible_(x_c[5], p[33], p[34])*p[5]-Henri_Michaelis_Menten_irreversible_(x_c[3], p[31], p[32])*p[5]
    dx[4] = -(p[25] * x_c[4] * y_c[2] - p[20] * x_c[16]) *p[4]-(p[21] * x_c[4] * x_c[0] - p[22] * x_c[10]) *p[4]-(p[23] * x_c[12] * x_c[4] - p[24] * x_c[6]) *p[4]-Henri_Michaelis_Menten_irreversible_(x_c[4], p[31], p[32])*p[4]+Henri_Michaelis_Menten_irreversible_(x_c[2], p[33], p[34])*p[4]
    dx[5] = -(p[27] * x_c[5] * x_c[1] - p[28] * x_c[15]) *p[5]-(p[25] * x_c[5] * x_c[17] - p[26] * x_c[9]) *p[5]-Henri_Michaelis_Menten_irreversible_(x_c[5], p[33], p[34])*p[5]+Henri_Michaelis_Menten_irreversible_(x_c[3], p[31], p[32])*p[5]-(p[29] * x_c[5] * y_c[3] - p[30] * x_c[13]) *p[5]
    dx[6] = (p[23] * x_c[12] * x_c[4] - p[24] * x_c[6]) *p[4]-Henri_Michaelis_Menten_irreversible_(x_c[6], p[37], p[38])*p[4]+Henri_Michaelis_Menten_irreversible_(y_c[1], p[45], p[46])*p[4]
    dx[7] = (p[23] * y_c[3] * x_c[3] - p[24] * x_c[7]) *p[5]+Henri_Michaelis_Menten_irreversible_(x_c[13], p[45], p[46])*p[5]-Henri_Michaelis_Menten_irreversible_(x_c[7], p[37], p[38])*p[5]
    dx[8] = (p[25] * x_c[2] * y_c[2] - p[26] * x_c[8]) *p[4]+Henri_Michaelis_Menten_irreversible_(x_c[16], p[41], p[36])*p[4]-Henri_Michaelis_Menten_irreversible_(x_c[8], p[41], p[42])*p[4]
    dx[9] = (p[25] * x_c[5] * x_c[17] - p[26] * x_c[9]) *p[5]-Henri_Michaelis_Menten_irreversible_(x_c[9], p[41], p[42])*p[5]+Henri_Michaelis_Menten_irreversible_(y_c[0], p[41], p[36])*p[5]
    dx[10] = (p[21] * x_c[4] * x_c[0] - p[22] * x_c[10]) *p[4]-Henri_Michaelis_Menten_irreversible_(x_c[10], p[39], p[40])*p[4]+Henri_Michaelis_Menten_irreversible_(x_c[14], p[43], p[44])*p[4]
    dx[11] = (p[21] * x_c[3] * x_c[1] - p[22] * x_c[11]) *p[5]+Henri_Michaelis_Menten_irreversible_(x_c[15], p[43], p[44])*p[5]-Henri_Michaelis_Menten_irreversible_(x_c[11], p[39], p[40])*p[5]
    dx[12] = -(p[23] * x_c[12] * x_c[4] - p[24] * x_c[6]) *p[4]-(p[29] * x_c[2] * x_c[12] - p[30] * y_c[1]) *p[4]+(p[61] * x_c[0] * x_c[0] - p[62] * x_c[12] * y_c[2]) *p[4]
    dx[13] = -Henri_Michaelis_Menten_irreversible_(x_c[13], p[45], p[46])*p[5]+Henri_Michaelis_Menten_irreversible_(x_c[7], p[37], p[38])*p[5]+(p[29] * x_c[5] * y_c[3] - p[30] * x_c[13]) *p[5]
    dx[14] = (p[27] * x_c[2] * x_c[0] - p[28] * x_c[14]) *p[4]+Henri_Michaelis_Menten_irreversible_(x_c[10], p[39], p[40])*p[4]-Henri_Michaelis_Menten_irreversible_(x_c[14], p[43], p[44])*p[4]
    dx[15] = (p[27] * x_c[5] * x_c[1] - p[28] * x_c[15]) *p[5]-Henri_Michaelis_Menten_irreversible_(x_c[15], p[43], p[44])*p[5]+Henri_Michaelis_Menten_irreversible_(x_c[11], p[39], p[40])*p[5]
    dx[16] = (p[25] * x_c[4] * y_c[2] - p[20] * x_c[16]) *p[4]-Henri_Michaelis_Menten_irreversible_(x_c[16], p[41], p[36])*p[4]+Henri_Michaelis_Menten_irreversible_(x_c[8], p[41], p[42])*p[4]
    dx[17] = FunctionForOxidativePhosphorylation(p_c[2], p[50], p[51], p[52])*p[5]-(p[25] * x_c[3] * x_c[17] - p[20] * y_c[0]) *p[5]-(p[25] * x_c[5] * x_c[17] - p[26] * x_c[9]) *p[5]+(p[63] * x_c[1] * x_c[1] - p[64] * y_c[3] * x_c[17]) *p[5]
    dx[18] = 0
    dx[19] = 0
    dx[20] = 0
    dx[21] = 0

    return dx

