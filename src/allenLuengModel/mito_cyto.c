#ifdef SIZE_DEFINITIONS
#define N_METABS 29
#define N_ODE_METABS 0
#define N_INDEP_METABS 18
#define N_COMPARTMENTS 4
#define N_GLOBAL_PARAMS 65
#define N_KIN_PARAMS 4
#define N_REACTIONS 35

#define N_ARRAY_SIZE_P  65	// number of parameters
#define N_ARRAY_SIZE_X  22	// number of initials
#define N_ARRAY_SIZE_Y  15	// number of assigned elements
#define N_ARRAY_SIZE_XC 22	// number of x concentration
#define N_ARRAY_SIZE_PC 3	// number of p concentration
#define N_ARRAY_SIZE_YC 4	// number of y concentration
#define N_ARRAY_SIZE_DX 22	// number of ODEs 
#define N_ARRAY_SIZE_CT 4	// number of conserved totals

#endif // SIZE_DEFINITIONS

#ifdef TIME
#define T  <set here a user name for the time variable> 
#endif // TIME

#ifdef NAME_ARRAYS
const char* p_names[] = {"glucose", "pyruvate", "pyruvate_m", "lysosome", "cytosol", "mitochondria", "mito_cyto", "n", "total_AMPK", "n1", "_T_", "_PI_", "_F_", "_F_nmol_", "_N_pmol_", "_K_GHK_", "_R_", "K_millivolts_per_volt", "Kr_rate_of_ATP_hydrolysis", "Kf_ATP_binding_AMPK", "Kr_ATP_binding_AMPK", "Kf_ADP_binding_AMPK", "Kr_ADP_binding_AMPK", "Kf_AMP_binding_AMPK", "Kr_AMP_binding_AMPK", "Kf_ATP_binding_pAMPK", "Kr_ATP_binding_pAMPK", "Kf_ADP_binding_pAMPK", "Kr_ADP_binding_pAMPK", "Kf_AMP_binding_pAMPK", "Kr_AMP_binding_pAMPK", "Km_phosphorylation_of_AMPK", "Vmax_phosphorylation_of_AMPK", "Km_dephosphorylation_of_AMPK", "Vmax_dephosphorylation_of_AMPK", "Km_phosphorylation_of_ATP_AMPK", "Vmax_phosphorylation_of_ATP_AMPK", "Km_phosphorylation_of_AMP_AMPK", "Vmax_phosphorylation_of_AMP_AMPK", "Km_phosphorylation_of_ADP_AMPK", "Vmax_phosphorylation_of_ADP_AMPK", "Km_dephosphorylation_of_ATP_p_AMPK", "Vmax_dephosphorylation_of_ATP_p_AMPK", "Km_dephosphorylation_of_ADP_p_AMPK", "Vmax_dephosphorylation_of_ADP_p_AMPK", "Km_dephosphorylation_of_AMP_p_AMPK", "Vmax_dephosphorylation_of_AMP_p_AMPK", "Kr_glycolysis", "XAK_simple_adenylate_kinase", "KAK_simple_adenylate_kinase", "KADP_Oxidative_Phosphorylation", "Vmaxoxphos_Oxidative_Phosphorylation", "nH_Oxidative_Phosphorylation", "Kf_ANT", "Kr_ANT", "Kf_pyruvate_transfer", "Kr_pyruvate_transfer", "Kf_basal_ATP_production", "Kr_basal_ATP_production", "activity_scale", "baseline_activity", "k1", "k2", "k1", "k2",  "" };
const char* x_names[] = {"ADP", "ADP_m", "p_AMPK", "AMPK_m", "AMPK", "p_AMPK_m", "AMP_AMPK", "AMP_AMPK_m", "ATP_p_AMPK", "ATP_p_AMPK_m", "ADP_AMPK", "ADP_AMPK_m", "AMP", "AMP_p_AMPK_m", "ADP_p_AMPK", "ADP_p_AMPK_m", "ATP_AMPK", "ATP_m", "ADP_lysosome", "AMP_lysosome", "ATP_lysosome", "Pi",  "" };
const char* y_names[] = {"ATP_AMPK_m", "AMP_p_AMPK", "ATP", "AMP_m", "Kf_rate_of_ATP_hydrolysis", "Kf_glycolysis", "Kf_simple_adenylate_kinase", "Kr_simple_adenylate_kinase", "FitValue", "mito_fitval", "All pAMPK", "All AMPK", "All pAMPK mito", "All AMPK mito", "total AMPK_pools",  "" };
const char* xc_names[] = {"ADP", "ADP_m", "p_AMPK", "AMPK_m", "AMPK", "p_AMPK_m", "AMP_AMPK", "AMP_AMPK_m", "ATP_p_AMPK", "ATP_p_AMPK_m", "ADP_AMPK", "ADP_AMPK_m", "AMP", "AMP_p_AMPK_m", "ADP_p_AMPK", "ADP_p_AMPK_m", "ATP_AMPK", "ATP_m", "ADP_lysosome", "AMP_lysosome", "ATP_lysosome", "Pi",  "" };
const char* pc_names[] = {"glucose", "pyruvate", "pyruvate_m",  "" };
const char* yc_names[] = {"ATP_AMPK_m", "AMP_p_AMPK", "ATP", "AMP_m",  "" };
const char* dx_names[] = {"ODE ADP", "ODE ADP_m", "ODE p_AMPK", "ODE AMPK_m", "ODE AMPK", "ODE p_AMPK_m", "ODE AMP_AMPK", "ODE AMP_AMPK_m", "ODE ATP_p_AMPK", "ODE ATP_p_AMPK_m", "ODE ADP_AMPK", "ODE ADP_AMPK_m", "ODE AMP", "ODE AMP_p_AMPK_m", "ODE ADP_p_AMPK", "ODE ADP_p_AMPK_m", "ODE ATP_AMPK", "ODE ATP_m", "ODE ADP_lysosome", "ODE AMP_lysosome", "ODE ATP_lysosome", "ODE Pi",  "" };
const char* ct_names[] = {"CT ATP_AMPK_m", "CT AMP_p_AMPK", "CT ATP", "CT AMP_m",  "" };
#endif // NAME_ARRAYS

#ifdef INITIAL
x[0] = 2.12416e-13;	//metabolite 'ADP': reactions
x[1] = 1.8223e-12;	//metabolite 'ADP_m': reactions
x[2] = 1.00023e-16;	//metabolite 'p_AMPK': reactions
x[3] = 1.00005e-16;	//metabolite 'AMPK_m': reactions
x[4] = 9.63666e-16;	//metabolite 'AMPK': reactions
x[5] = 1.00005e-16;	//metabolite 'p_AMPK_m': reactions
x[6] = 9.98667e-16;	//metabolite 'AMP_AMPK': reactions
x[7] = 1.00002e-16;	//metabolite 'AMP_AMPK_m': reactions
x[8] = 1.49186e-17;	//metabolite 'ATP_p_AMPK': reactions
x[9] = 1.00007e-16;	//metabolite 'ATP_p_AMPK_m': reactions
x[10] = 1.00009e-17;	//metabolite 'ADP_AMPK': reactions
x[11] = 9.99908e-16;	//metabolite 'ADP_AMPK_m': reactions
x[12] = 1e-14;	//metabolite 'AMP': reactions
x[13] = 1.00002e-16;	//metabolite 'AMP_p_AMPK_m': reactions
x[14] = 1.0003e-17;	//metabolite 'ADP_p_AMPK': reactions
x[15] = 1e-16;	//metabolite 'ADP_p_AMPK_m': reactions
x[16] = 1e-15;	//metabolite 'ATP_AMPK': reactions
x[17] = 2.50004e-12;	//metabolite 'ATP_m': reactions
x[18] = 0;	//metabolite 'ADP_lysosome': reactions
x[19] = 0;	//metabolite 'AMP_lysosome': reactions
x[20] = 0;	//metabolite 'ATP_lysosome': reactions
x[21] = 3e-12;	//metabolite 'Pi': reactions
#endif /* INITIAL */

#ifdef FIXED
// ct[0] = 1.7172224382999999e-15;	//ct[0] conserved total for 'ATP_AMPK_m'
// ct[1] = 3.7367737269944892e-15;	//ct[1] conserved total for 'AMP_p_AMPK'
// ct[2] = 9.2213347519424476e-12;	//ct[2] conserved total for 'ATP'
// ct[3] = 4.4221399788238003e-12;	//ct[3] conserved total for 'AMP_m'
p[0] = 1e-15;	//metabolite 'glucose': fixed
p[1] = 1e-15;	//metabolite 'pyruvate': fixed
p[2] = 1e-15;	//metabolite 'pyruvate_m': fixed
p[3] = 1e-15;	//compartment 'lysosome':fixed
p[4] = 1e-15;	//compartment 'cytosol':fixed
p[5] = 1e-15;	//compartment 'mitochondria':fixed
p[6] = 2e-11;	//compartment 'mito_cyto':fixed
p[7] = 1;	//global quantity 'n':fixed
p[8] = 450;	//global quantity 'total_AMPK':fixed
p[9] = 1;	//global quantity 'n1':fixed
p[10] = 300;	//global quantity '_T_':fixed
p[11] = 3.14159;	//global quantity '_PI_':fixed
p[12] = 96485.3;	//global quantity '_F_':fixed
p[13] = 9.64853e-05;	//global quantity '_F_nmol_':fixed
p[14] = 6.02214e+11;	//global quantity '_N_pmol_':fixed
p[15] = 1e-09;	//global quantity '_K_GHK_':fixed
p[16] = 8314.46;	//global quantity '_R_':fixed
p[17] = 1000;	//global quantity 'K_millivolts_per_volt':fixed
p[18] = 0;	//global quantity 'Kr_rate_of_ATP_hydrolysis':fixed
p[19] = 0.000831589;	//global quantity 'Kf_ATP_binding_AMPK':fixed
p[20] = 0.01;	//global quantity 'Kr_ATP_binding_AMPK':fixed
p[21] = 0.00250488;	//global quantity 'Kf_ADP_binding_AMPK':fixed
p[22] = 0.0001;	//global quantity 'Kr_ADP_binding_AMPK':fixed
p[23] = 0.0001;	//global quantity 'Kf_AMP_binding_AMPK':fixed
p[24] = 0.01;	//global quantity 'Kr_AMP_binding_AMPK':fixed
p[25] = 0.000774664;	//global quantity 'Kf_ATP_binding_pAMPK':fixed
p[26] = 0.000100331;	//global quantity 'Kr_ATP_binding_pAMPK':fixed
p[27] = 0.00999998;	//global quantity 'Kf_ADP_binding_pAMPK':fixed
p[28] = 0.000100004;	//global quantity 'Kr_ADP_binding_pAMPK':fixed
p[29] = 0.000100059;	//global quantity 'Kf_AMP_binding_pAMPK':fixed
p[30] = 0.000396674;	//global quantity 'Kr_AMP_binding_pAMPK':fixed
p[31] = 1475.78;	//global quantity 'Km_phosphorylation_of_AMPK':fixed
p[32] = 0.121172;	//global quantity 'Vmax_phosphorylation_of_AMPK':fixed
p[33] = 111.473;	//global quantity 'Km_dephosphorylation_of_AMPK':fixed
p[34] = 10.8709;	//global quantity 'Vmax_dephosphorylation_of_AMPK':fixed
p[35] = 1053.26;	//global quantity 'Km_phosphorylation_of_ATP_AMPK':fixed
p[36] = 0.162046;	//global quantity 'Vmax_phosphorylation_of_ATP_AMPK':fixed
p[37] = 140.741;	//global quantity 'Km_phosphorylation_of_AMP_AMPK':fixed
p[38] = 2.02627;	//global quantity 'Vmax_phosphorylation_of_AMP_AMPK':fixed
p[39] = 215.351;	//global quantity 'Km_phosphorylation_of_ADP_AMPK':fixed
p[40] = 0.697823;	//global quantity 'Vmax_phosphorylation_of_ADP_AMPK':fixed
p[41] = 6.1571;	//global quantity 'Km_dephosphorylation_of_ATP_p_AMPK':fixed
p[42] = 72.508;	//global quantity 'Vmax_dephosphorylation_of_ATP_p_AMPK':fixed
p[43] = 17.48;	//global quantity 'Km_dephosphorylation_of_ADP_p_AMPK':fixed
p[44] = 0.0358125;	//global quantity 'Vmax_dephosphorylation_of_ADP_p_AMPK':fixed
p[45] = 6.36734;	//global quantity 'Km_dephosphorylation_of_AMP_p_AMPK':fixed
p[46] = 0.146208;	//global quantity 'Vmax_dephosphorylation_of_AMP_p_AMPK':fixed
p[47] = 0;	//global quantity 'Kr_glycolysis':fixed
p[48] = 9.24309e-06;	//global quantity 'XAK_simple_adenylate_kinase':fixed
p[49] = 0.536266;	//global quantity 'KAK_simple_adenylate_kinase':fixed
p[50] = 58;	//global quantity 'KADP_Oxidative_Phosphorylation':fixed
p[51] = 500;	//global quantity 'Vmaxoxphos_Oxidative_Phosphorylation':fixed
p[52] = 2.568;	//global quantity 'nH_Oxidative_Phosphorylation':fixed
p[53] = 1e-06;	//global quantity 'Kf_ANT':fixed
p[54] = 1e-06;	//global quantity 'Kr_ANT':fixed
p[55] = 1e-06;	//global quantity 'Kf_pyruvate_transfer':fixed
p[56] = 1e-06;	//global quantity 'Kr_pyruvate_transfer':fixed
p[57] = 0.1;	//global quantity 'Kf_basal_ATP_production':fixed
p[58] = 0;	//global quantity 'Kr_basal_ATP_production':fixed
p[59] = 1.5;	//global quantity 'activity_scale':fixed
p[60] = 0.663742;	//global quantity 'baseline_activity':fixed
p[61] = 0.1;	//reaction 'simple adenylate kinase':  kinetic parameter 'k1'
p[62] = 0.1;	//reaction 'simple adenylate kinase':  kinetic parameter 'k2'
p[63] = 0.1;	//reaction 'simple adenylate kinase_copy':  kinetic parameter 'k1'
p[64] = 0.1;	//reaction 'simple adenylate kinase_copy':  kinetic parameter 'k2'
p[65] = 1.7172224382999999e-15;	//ct[0] conserved total for 'ATP_AMPK_m'
p[66] = 3.7367737269944892e-15;	//ct[1] conserved total for 'AMP_p_AMPK'
p[67] = 9.2213347519424476e-12;	//ct[2] conserved total for 'ATP'
p[68] = 4.4221399788238003e-12;	//ct[3] conserved total for 'AMP_m'
#endif /* FIXED */

#ifdef ASSIGNMENT
y[0] = p[65]-x[3]-x[5]-x[7]-x[9]-1*x[11]-x[13]-x[15];	//metabolite 'ATP_AMPK_m': reactions
y[1] = p[66]-1*x[2]-x[4]-1*x[6]-1*x[8]-1*x[10]-x[14]-1*x[16];	//metabolite 'AMP_p_AMPK': reactions
y[2] = p[67]-x[0]+x[2]+x[4]-x[12];	//metabolite 'ATP': reactions
y[3] = p[68]-x[1]+x[3]+x[5]-x[17];	//metabolite 'AMP_m': reactions
y[4] = 0.05000000000000000*p[7];	//model entity 'Kf_rate_of_ATP_hydrolysis':assignment
y[5] = 0.00498161634948818*p[9];	//model entity 'Kf_glycolysis':assignment
y[6] = p[48]*p[49];	//model entity 'Kf_simple_adenylate_kinase':assignment
y[7] = p[48];	//model entity 'Kr_simple_adenylate_kinase':assignment
y[8] = p[60]+(y_c[1]+x_c[14]+x_c[8]+x_c[2])/(x_c[6]+x_c[10]+x_c[16]+x_c[4])*p[59];	//model entity 'FitValue':assignment
y[9] = p[60]+(x_c[5]+x_c[15]+x_c[13]+x_c[9])/(x_c[3]+x_c[7]+x_c[11]+y_c[0])*p[59];	//model entity 'mito_fitval':assignment
y[10] = x_c[2]+y_c[1]+x_c[14]+x_c[8];	//model entity 'All pAMPK':assignment
y[11] = x_c[4]+x_c[6]+x_c[10]+x_c[16];	//model entity 'All AMPK':assignment
y[12] = x_c[5]+x_c[13]+x_c[15]+x_c[9];	//model entity 'All pAMPK mito':assignment
y[13] = x_c[3]+x_c[7]+x_c[11]+y_c[0];	//model entity 'All AMPK mito':assignment
y[14] = y[13]+y[11]+y[12]+y[10];	//model entity 'total AMPK_pools':assignment
x_c[0] = x[0]/p[4];	//concentration of metabolite 'ADP': reactions
x_c[1] = x[1]/p[5];	//concentration of metabolite 'ADP_m': reactions
x_c[2] = x[2]/p[4];	//concentration of metabolite 'p_AMPK': reactions
x_c[3] = x[3]/p[5];	//concentration of metabolite 'AMPK_m': reactions
x_c[4] = x[4]/p[4];	//concentration of metabolite 'AMPK': reactions
x_c[5] = x[5]/p[5];	//concentration of metabolite 'p_AMPK_m': reactions
x_c[6] = x[6]/p[4];	//concentration of metabolite 'AMP_AMPK': reactions
x_c[7] = x[7]/p[5];	//concentration of metabolite 'AMP_AMPK_m': reactions
x_c[8] = x[8]/p[4];	//concentration of metabolite 'ATP_p_AMPK': reactions
x_c[9] = x[9]/p[5];	//concentration of metabolite 'ATP_p_AMPK_m': reactions
x_c[10] = x[10]/p[4];	//concentration of metabolite 'ADP_AMPK': reactions
x_c[11] = x[11]/p[5];	//concentration of metabolite 'ADP_AMPK_m': reactions
x_c[12] = x[12]/p[4];	//concentration of metabolite 'AMP': reactions
x_c[13] = x[13]/p[5];	//concentration of metabolite 'AMP_p_AMPK_m': reactions
x_c[14] = x[14]/p[4];	//concentration of metabolite 'ADP_p_AMPK': reactions
x_c[15] = x[15]/p[5];	//concentration of metabolite 'ADP_p_AMPK_m': reactions
x_c[16] = x[16]/p[4];	//concentration of metabolite 'ATP_AMPK': reactions
x_c[17] = x[17]/p[5];	//concentration of metabolite 'ATP_m': reactions
y_c[0] = y[0]/p[5];	//concentration of metabolite 'ATP_AMPK_m': reactions
y_c[1] = y[1]/p[4];	//concentration of metabolite 'AMP_p_AMPK': reactions
y_c[2] = y[2]/p[4];	//concentration of metabolite 'ATP': reactions
y_c[3] = y[3]/p[5];	//concentration of metabolite 'AMP_m': reactions
p_c[0] = p[0]/p[4];	//concentration of metabolite 'glucose': fixed
p_c[1] = p[1]/p[4];	//concentration of metabolite 'pyruvate': fixed
p_c[2] = p[2]/p[5];	//concentration of metabolite 'pyruvate_m': fixed
x_c[18] = x[18]/p[3];	//concentration of metabolite 'ADP_lysosome': reactions
x_c[19] = x[19]/p[3];	//concentration of metabolite 'AMP_lysosome': reactions
x_c[20] = x[20]/p[3];	//concentration of metabolite 'ATP_lysosome': reactions
x_c[21] = x[21]/p[4];	//concentration of metabolite 'Pi': reactions
#endif /* ASSIGNMENT */

#ifdef FUNCTIONS_HEADERS
double Henri_Michaelis_Menten_irreversible_(double sub_0, double param_0, double param_1); 
double FunctionForOxidativePhosphorylation(double sub_0, double param_0, double param_1, double param_2); 
#endif /* FUNCTIONS_HEADERS */

#ifdef FUNCTIONS
double Henri_Michaelis_Menten_irreversible_(double sub_0, double param_0, double param_1) 	//Henri-Michaelis-Menten (irreversible)
{return  param_1*sub_0/(param_0+sub_0);} 
double FunctionForOxidativePhosphorylation(double sub_0, double param_0, double param_1, double param_2) 	//Function for Oxidative Phosphorylation
{return  param_1*pow((sub_0/param_0),param_2)/(1.00000000000000000+pow((sub_0/param_0),param_2));} 
#endif /* FUNCTIONS */

#ifdef ODEs
dx[0] = (y[4] * y_c[2]) *p[4]-(p[21] * x_c[4] * x_c[0] - p[22] * x_c[10]) *p[4]-(p[27] * x_c[2] * x_c[0] - p[28] * x_c[14]) *p[4]-(y[5] * x_c[0] * p_c[0]) *p[4]-2*(p[61] * x_c[0] * x_c[0] - p[62] * x_c[12] * y_c[2]) *p[4]-(p[57] * x_c[0]) *p[4];
dx[1] = -FunctionForOxidativePhosphorylation(p_c[2], p[50], p[51], p[52])*p[5]-(p[21] * x_c[3] * x_c[1] - p[22] * x_c[11]) *p[5]-(p[27] * x_c[5] * x_c[1] - p[28] * x_c[15]) *p[5]-2*(p[63] * x_c[1] * x_c[1] - p[64] * y_c[3] * x_c[17]) *p[5];
dx[2] = -(p[25] * x_c[2] * y_c[2] - p[26] * x_c[8]) *p[4]-(p[27] * x_c[2] * x_c[0] - p[28] * x_c[14]) *p[4]-(p[29] * x_c[2] * x_c[12] - p[30] * y_c[1]) *p[4]+Henri_Michaelis_Menten_irreversible_(x_c[4], p[31], p[32])*p[4]-Henri_Michaelis_Menten_irreversible_(x_c[2], p[33], p[34])*p[4];
dx[3] = -(p[25] * x_c[3] * x_c[17] - p[20] * y_c[0]) *p[5]-(p[21] * x_c[3] * x_c[1] - p[22] * x_c[11]) *p[5]-(p[23] * y_c[3] * x_c[3] - p[24] * x_c[7]) *p[5]+Henri_Michaelis_Menten_irreversible_(x_c[5], p[33], p[34])*p[5]-Henri_Michaelis_Menten_irreversible_(x_c[3], p[31], p[32])*p[5];
dx[4] = -(p[25] * x_c[4] * y_c[2] - p[20] * x_c[16]) *p[4]-(p[21] * x_c[4] * x_c[0] - p[22] * x_c[10]) *p[4]-(p[23] * x_c[12] * x_c[4] - p[24] * x_c[6]) *p[4]-Henri_Michaelis_Menten_irreversible_(x_c[4], p[31], p[32])*p[4]+Henri_Michaelis_Menten_irreversible_(x_c[2], p[33], p[34])*p[4];
dx[5] = -(p[27] * x_c[5] * x_c[1] - p[28] * x_c[15]) *p[5]-(p[25] * x_c[5] * x_c[17] - p[26] * x_c[9]) *p[5]-Henri_Michaelis_Menten_irreversible_(x_c[5], p[33], p[34])*p[5]+Henri_Michaelis_Menten_irreversible_(x_c[3], p[31], p[32])*p[5]-(p[29] * x_c[5] * y_c[3] - p[30] * x_c[13]) *p[5];
dx[6] = (p[23] * x_c[12] * x_c[4] - p[24] * x_c[6]) *p[4]-Henri_Michaelis_Menten_irreversible_(x_c[6], p[37], p[38])*p[4]+Henri_Michaelis_Menten_irreversible_(y_c[1], p[45], p[46])*p[4];
dx[7] = (p[23] * y_c[3] * x_c[3] - p[24] * x_c[7]) *p[5]+Henri_Michaelis_Menten_irreversible_(x_c[13], p[45], p[46])*p[5]-Henri_Michaelis_Menten_irreversible_(x_c[7], p[37], p[38])*p[5];
dx[8] = (p[25] * x_c[2] * y_c[2] - p[26] * x_c[8]) *p[4]+Henri_Michaelis_Menten_irreversible_(x_c[16], p[41], p[36])*p[4]-Henri_Michaelis_Menten_irreversible_(x_c[8], p[41], p[42])*p[4];
dx[9] = (p[25] * x_c[5] * x_c[17] - p[26] * x_c[9]) *p[5]-Henri_Michaelis_Menten_irreversible_(x_c[9], p[41], p[42])*p[5]+Henri_Michaelis_Menten_irreversible_(y_c[0], p[41], p[36])*p[5];
dx[10] = (p[21] * x_c[4] * x_c[0] - p[22] * x_c[10]) *p[4]-Henri_Michaelis_Menten_irreversible_(x_c[10], p[39], p[40])*p[4]+Henri_Michaelis_Menten_irreversible_(x_c[14], p[43], p[44])*p[4];
dx[11] = (p[21] * x_c[3] * x_c[1] - p[22] * x_c[11]) *p[5]+Henri_Michaelis_Menten_irreversible_(x_c[15], p[43], p[44])*p[5]-Henri_Michaelis_Menten_irreversible_(x_c[11], p[39], p[40])*p[5];
dx[12] = -(p[23] * x_c[12] * x_c[4] - p[24] * x_c[6]) *p[4]-(p[29] * x_c[2] * x_c[12] - p[30] * y_c[1]) *p[4]+(p[61] * x_c[0] * x_c[0] - p[62] * x_c[12] * y_c[2]) *p[4];
dx[13] = -Henri_Michaelis_Menten_irreversible_(x_c[13], p[45], p[46])*p[5]+Henri_Michaelis_Menten_irreversible_(x_c[7], p[37], p[38])*p[5]+(p[29] * x_c[5] * y_c[3] - p[30] * x_c[13]) *p[5];
dx[14] = (p[27] * x_c[2] * x_c[0] - p[28] * x_c[14]) *p[4]+Henri_Michaelis_Menten_irreversible_(x_c[10], p[39], p[40])*p[4]-Henri_Michaelis_Menten_irreversible_(x_c[14], p[43], p[44])*p[4];
dx[15] = (p[27] * x_c[5] * x_c[1] - p[28] * x_c[15]) *p[5]-Henri_Michaelis_Menten_irreversible_(x_c[15], p[43], p[44])*p[5]+Henri_Michaelis_Menten_irreversible_(x_c[11], p[39], p[40])*p[5];
dx[16] = (p[25] * x_c[4] * y_c[2] - p[20] * x_c[16]) *p[4]-Henri_Michaelis_Menten_irreversible_(x_c[16], p[41], p[36])*p[4]+Henri_Michaelis_Menten_irreversible_(x_c[8], p[41], p[42])*p[4];
dx[17] = FunctionForOxidativePhosphorylation(p_c[2], p[50], p[51], p[52])*p[5]-(p[25] * x_c[3] * x_c[17] - p[20] * y_c[0]) *p[5]-(p[25] * x_c[5] * x_c[17] - p[26] * x_c[9]) *p[5]+(p[63] * x_c[1] * x_c[1] - p[64] * y_c[3] * x_c[17]) *p[5];
dx[18] = 0;
dx[19] = 0;
dx[20] = 0;
dx[21] = 0;
#endif /* ODEs */
