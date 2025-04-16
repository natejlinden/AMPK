using StructuralIdentifiability, ModelingToolkit

# model parameters
@parameters kGly, kHydro, VforAK, KeqAK, kmm, kmd, kmt, VmaxOxPhos, Kadp, n, VforCK, Kb, Kia, Kib, Kiq, Kp, KeqCK, TCr, kOnAMP, kOffAMP, kOnADP, kOffADP, kOnATP, kOffATP, kOnCaMKK, kOffCaMKK, kPhosCaMKK, kOnLKB1, kOffLKB1, kPhosLKB1, kOnPP, kOffPP, kDephosPP, kOnAMPK, kOffAMPK, kPhosAMPK, kOnPP1, kOffPP1, kDephosPP1, kOnCaM, kOffCaM, kPhosCaM, KmCaM, kDephosCaMKK, alphaPP, betaAMP, betaCaMKK, betaLKB1

@independent_variables t

# state variables
@variables AMP(t), ADP(t), ATP(t), PCr(t), Ca(t), AMPK(t), pAMPK(t), AMP_AMPK(t), ADP_AMPK(t), ATP_AMPK(t), AMP_pAMPK(t), ADP_pAMPK(t), ATP_pAMPK(t), CaM(t), CaCaM(t), CaMKK(t), CaMKK_act(t), CaMKK_act_AMPK(t), CaMKK_act_AMP_AMPK(t), CaMKK_act_ADP_AMPK(t), CaMKK_act_ATP_AMPK(t), LKB1(t), LKB1_AMPK(t), LKB1_AMP_AMPK(t), LKB1_ADP_AMPK(t), LKB1_ATP_AMPK(t), PP(t), PP_pAMPK(t), PP_AMP_pAMPK(t), PP_ADP_pAMPK(t), PP_ATP_pAMPK(t), AMPKAR(t), pAMPKAR(t), AMPKAR_pAMPK(t), AMPKAR_AMP_pAMPK(t), AMPKAR_ADP_pAMPK(t), AMPKAR_ATP_pAMPK(t), PP1(t), PP1_pAMPKAR(t), y(t)

# differential operator
D = Differential(t)

# model equations
eqns = [
    D(AMP) ~ -(kOnAMP*AMP*AMPK - kOffAMP*AMP_AMPK)-(kOnAMP*AMP*pAMPK - kOffAMP*AMP_pAMPK)-((((VforAK*ATP*AMP)/(kmt*kmm)) - ((((VforAK*(kmd*kmd))/(KeqAK*kmt*kmm))*(ADP*ADP))/(kmd*kmd)))/(1 + (ATP/kmt) + (AMP/kmm) + ((ATP*AMP)/(kmt*kmm)) + ((2*ADP)/kmd) + ((ADP*ADP)/(kmd*kmd))))-(kOnAMP*AMP*CaMKK_act_AMPK - kOffADP*CaMKK_act_AMP_AMPK)-(kOnAMP*AMP*LKB1_AMPK - kOffAMP*LKB1_AMP_AMPK)-(kOnAMP*AMP*PP_pAMPK - kOffAMP*PP_AMP_pAMPK)-(kOnAMP*AMP*AMPKAR_pAMPK - kOffAMP*AMPKAR_AMP_pAMPK),
    D(ADP) ~ -(kOnADP*ADP*AMPK - kOffADP*ADP_AMPK)-(kOnADP*ADP*pAMPK - kOffADP*ADP_pAMPK)-(kGly*ADP)+2*((((VforAK*ATP*AMP)/(kmt*kmm)) - ((((VforAK*(kmd*kmd))/(KeqAK*kmt*kmm))*(ADP*ADP))/(kmd*kmd)))/(1 + (ATP/kmt) + (AMP/kmm) + ((ATP*AMP)/(kmt*kmm)) + ((2*ADP)/kmd) + ((ADP*ADP)/(kmd*kmd))))+(kHydro*ATP)-((VmaxOxPhos * ((ADP/Kadp)*(ADP/Kadp)))/(1 + ((ADP/Kadp)*(ADP/Kadp))))+((((((VforCK*Kiq*Kp)/(KeqCK*Kia*Kb))*ATP*(TCr - PCr))/(Kiq*Kp)) - ((VforCK*ADP*PCr)/(Kia*Kb)))/(1 + (ADP/Kia) + (PCr/Kib) + (ATP/Kiq) + ((ADP*PCr)/(Kia*Kb)) + (((TCr - PCr)*ATP)/(Kiq*Kp))))-(kOnADP*ADP*CaMKK_act_AMPK - kOffADP*CaMKK_act_ADP_AMPK)-(kOnADP*ADP*LKB1_AMPK - kOffADP*LKB1_ADP_AMPK)-(kOnADP*ADP*PP_pAMPK - kOffADP*PP_ADP_pAMPK)-(kOnADP*ADP*AMPKAR_pAMPK - kOffADP*AMPKAR_ADP_pAMPK),
    D(ATP) ~ -(kOnATP*ATP*AMPK - kOffATP*ATP_AMPK)-(kOnATP*ATP*pAMPK - kOffATP*ATP_pAMPK)+(kGly*ADP)-((((VforAK*ATP*AMP)/(kmt*kmm)) - ((((VforAK*(kmd*kmd))/(KeqAK*kmt*kmm))*(ADP*ADP))/(kmd*kmd)))/(1 + (ATP/kmt) + (AMP/kmm) + ((ATP*AMP)/(kmt*kmm)) + ((2*ADP)/kmd) + ((ADP*ADP)/(kmd*kmd))))-(kHydro*ATP)+((VmaxOxPhos * ((ADP/Kadp)*(ADP/Kadp)))/(1 + ((ADP/Kadp)*(ADP/Kadp))))-((((((VforCK*Kiq*Kp)/(KeqCK*Kia*Kb))*ATP*(TCr - PCr))/(Kiq*Kp)) - ((VforCK*ADP*PCr)/(Kia*Kb)))/(1 + (ADP/Kia) + (PCr/Kib) + (ATP/Kiq) + ((ADP*PCr)/(Kia*Kb)) + (((TCr - PCr)*ATP)/(Kiq*Kp))))-(kOnATP*ATP*CaMKK_act_AMPK - kOffATP*CaMKK_act_ATP_AMPK)-(kOnATP*ATP*LKB1_AMPK - kOffATP*LKB1_ATP_AMPK)-(kOnATP*ATP*PP_pAMPK - kOffATP*PP_ATP_pAMPK)-(kOnATP*ATP*AMPKAR_pAMPK - kOffATP*AMPKAR_ATP_pAMPK),
    D(PCr) ~ ((((((VforCK*Kiq*Kp)/(KeqCK*Kia*Kb))*ATP*(TCr - PCr))/(Kiq*Kp)) - ((VforCK*ADP*PCr)/(Kia*Kb)))/(1 + (ADP/Kia) + (PCr/Kib) + (ATP/Kiq) + ((ADP*PCr)/(Kia*Kb)) + (((TCr - PCr)*ATP)/(Kiq*Kp)))),
    D(AMPK) ~ -(kOnAMP*AMP*AMPK - kOffAMP*AMP_AMPK)-(kOnADP*ADP*AMPK - kOffADP*ADP_AMPK)-(kOnATP*ATP*AMPK - kOffATP*ATP_AMPK)-(kOnCaMKK*CaMKK_act*AMPK - kOffCaMKK*CaMKK_act_AMPK)+(kDephosPP*PP_pAMPK)-(kOnLKB1*LKB1*AMPK -  kOffLKB1*LKB1_AMPK),
    D(pAMPK) ~ -(kOnAMP*AMP*pAMPK - kOffAMP*AMP_pAMPK)-(kOnADP*ADP*pAMPK - kOffADP*ADP_pAMPK)-(kOnATP*ATP*pAMPK - kOffATP*ATP_pAMPK)+(kPhosCaMKK*CaMKK_act_AMPK)-(kOnPP*PP*pAMPK - kOffPP*PP_pAMPK)+(kPhosLKB1*LKB1_AMPK)-(kOnAMPK*AMPKAR*pAMPK - kOffAMPK*AMPKAR_pAMPK)+(kPhosAMPK*AMPKAR_pAMPK),
    D(AMP_AMPK) ~ (kOnAMP*AMP*AMPK - kOffAMP*AMP_AMPK)-(kOnCaMKK*CaMKK_act*AMP_AMPK - kOffCaMKK*CaMKK_act_AMP_AMPK)-(kOnLKB1*LKB1*AMP_AMPK - kOffLKB1*LKB1_AMP_AMPK)+(alphaPP*kDephosPP*PP_AMP_pAMPK),
    D(ADP_AMPK) ~ (kOnADP*ADP*AMPK - kOffADP*ADP_AMPK)-(kOnCaMKK*CaMKK_act*ADP_AMPK - kOffCaMKK*CaMKK_act_ADP_AMPK)-(kOnLKB1*LKB1*ADP_AMPK -  kOffLKB1*LKB1_ADP_AMPK)+(alphaPP*kDephosPP*PP_ADP_pAMPK),
    D(ATP_AMPK) ~ (kOnATP*ATP*AMPK - kOffATP*ATP_AMPK)-(kOnCaMKK*CaMKK_act*ATP_AMPK - kOffCaMKK*CaMKK_act_ATP_AMPK)+(kDephosPP*PP_ATP_pAMPK)-(kOnLKB1*LKB1*ATP_AMPK -  kOffLKB1*LKB1_ATP_AMPK),
    D(AMP_pAMPK) ~ (kOnAMP*AMP*pAMPK - kOffAMP*AMP_pAMPK)+(betaCaMKK*kPhosCaMKK*CaMKK_act_AMP_AMPK)+(betaLKB1*kPhosLKB1*LKB1_AMP_AMPK)-(kOnAMPK*AMPKAR*AMP_pAMPK - kOffAMPK*AMPKAR_AMP_pAMPK)+(betaAMP*kPhosAMPK*AMPKAR_AMP_pAMPK)-(kOnPP*PP*AMP_pAMPK - kOffPP*PP_AMP_pAMPK),
    D(ADP_pAMPK) ~ (kOnADP*ADP*pAMPK - kOffADP*ADP_pAMPK)+(betaCaMKK*kPhosCaMKK*CaMKK_act_ADP_AMPK)+(betaLKB1*kPhosLKB1*LKB1_ADP_AMPK)-(alphaPP*kDephosPP*PP_ADP_pAMPK)-(kOnAMPK*AMPKAR*ADP_pAMPK - kOffAMPK*AMPKAR_ADP_pAMPK)+(kPhosAMPK*AMPKAR_ADP_pAMPK),
    D(ATP_pAMPK) ~ (kOnATP*ATP*pAMPK - kOffATP*ATP_pAMPK)+(kPhosCaMKK*CaMKK_act_ATP_AMPK)-(kOnPP*PP*ATP_pAMPK - kOffPP*PP_ATP_pAMPK)+(kPhosLKB1*LKB1_ATP_AMPK)-(kOnAMPK*AMPKAR*ATP_pAMPK - kOffAMPK*AMPKAR_ATP_pAMPK)+(kPhosAMPK*AMPKAR_ATP_pAMPK),
    D(Ca) ~ -(kOnCaM*(Ca*Ca*Ca)*CaM - kOffCaM*CaCaM),
    D(CaM) ~ -(kOnCaM*(Ca*Ca*Ca)*CaM - kOffCaM*CaCaM),
    D(CaCaM) ~ (kOnCaM*(Ca*Ca*Ca)*CaM - kOffCaM*CaCaM),
    D(CaMKK) ~ -((kPhosCaM*(CaCaM*CaCaM*CaCaM*CaCaM)*CaMKK)/(KmCaM*KmCaM*KmCaM*KmCaM + CaCaM*CaCaM*CaCaM*CaCaM)) + (kDephosCaMKK*CaMKK_act),
    D(CaMKK_act) ~ ((kPhosCaM*(CaCaM*CaCaM*CaCaM*CaCaM)*CaMKK)/(KmCaM*KmCaM*KmCaM*KmCaM + CaCaM*CaCaM*CaCaM*CaCaM)) - (kDephosCaMKK*CaMKK_act)-(kOnCaMKK*CaMKK_act*AMPK - kOffCaMKK*CaMKK_act_AMPK)+(kPhosCaMKK*CaMKK_act_AMPK)-(kOnCaMKK*CaMKK_act*AMP_AMPK - kOffCaMKK*CaMKK_act_AMP_AMPK)+(betaCaMKK*kPhosCaMKK*CaMKK_act_AMP_AMPK)-(kOnCaMKK*CaMKK_act*ADP_AMPK - kOffCaMKK*CaMKK_act_ADP_AMPK)+(betaCaMKK*kPhosCaMKK*CaMKK_act_ADP_AMPK)-(kOnCaMKK*CaMKK_act*ATP_AMPK - kOffCaMKK*CaMKK_act_ATP_AMPK)+(kPhosCaMKK*CaMKK_act_ATP_AMPK),
    D(CaMKK_act_AMPK) ~ (kOnCaMKK*CaMKK_act*AMPK - kOffCaMKK*CaMKK_act_AMPK)-(kPhosCaMKK*CaMKK_act_AMPK)-(kOnAMP*AMP*CaMKK_act_AMPK - kOffADP*CaMKK_act_AMP_AMPK)-(kOnADP*ADP*CaMKK_act_AMPK - kOffADP*CaMKK_act_ADP_AMPK)-(kOnATP*ATP*CaMKK_act_AMPK - kOffATP*CaMKK_act_ATP_AMPK),
    D(CaMKK_act_AMP_AMPK) ~ (kOnCaMKK*CaMKK_act*AMP_AMPK - kOffCaMKK*CaMKK_act_AMP_AMPK)-(betaCaMKK*kPhosCaMKK*CaMKK_act_AMP_AMPK)+(kOnAMP*AMP*CaMKK_act_AMPK - kOffADP*CaMKK_act_AMP_AMPK),
    D(CaMKK_act_ADP_AMPK) ~ (kOnCaMKK*CaMKK_act*ADP_AMPK - kOffCaMKK*CaMKK_act_ADP_AMPK)-(betaCaMKK*kPhosCaMKK*CaMKK_act_ADP_AMPK)+(kOnADP*ADP*CaMKK_act_AMPK - kOffADP*CaMKK_act_ADP_AMPK),
    D(CaMKK_act_ATP_AMPK) ~ (kOnCaMKK*CaMKK_act*ATP_AMPK - kOffCaMKK*CaMKK_act_ATP_AMPK)-(kPhosCaMKK*CaMKK_act_ATP_AMPK)+(kOnATP*ATP*CaMKK_act_AMPK - kOffATP*CaMKK_act_ATP_AMPK),
    D(LKB1) ~ -(kOnLKB1*LKB1*AMP_AMPK - kOffLKB1*LKB1_AMP_AMPK)+(betaLKB1*kPhosLKB1*LKB1_AMP_AMPK)-(kOnLKB1*LKB1*ADP_AMPK -  kOffLKB1*LKB1_ADP_AMPK)+(betaLKB1*kPhosLKB1*LKB1_ADP_AMPK)-(kOnLKB1*LKB1*ATP_AMPK -  kOffLKB1*LKB1_ATP_AMPK)+(kPhosLKB1*LKB1_ATP_AMPK)-(kOnLKB1*LKB1*AMPK -  kOffLKB1*LKB1_AMPK)+(kPhosLKB1*LKB1_AMPK),
    D(LKB1_AMP_AMPK) ~ (kOnLKB1*LKB1*AMP_AMPK - kOffLKB1*LKB1_AMP_AMPK)-(betaLKB1*kPhosLKB1*LKB1_AMP_AMPK)+(kOnAMP*AMP*LKB1_AMPK - kOffAMP*LKB1_AMP_AMPK),
    D(LKB1_ADP_AMPK) ~ (kOnLKB1*LKB1*ADP_AMPK -  kOffLKB1*LKB1_ADP_AMPK)-(betaLKB1*kPhosLKB1*LKB1_ADP_AMPK)+(kOnADP*ADP*LKB1_AMPK - kOffADP*LKB1_ADP_AMPK),
    D(LKB1_ATP_AMPK) ~ (kOnLKB1*LKB1*ATP_AMPK -  kOffLKB1*LKB1_ATP_AMPK)-(kPhosLKB1*LKB1_ATP_AMPK)+(kOnATP*ATP*LKB1_AMPK - kOffATP*LKB1_ATP_AMPK),
    D(LKB1_AMPK) ~ (kOnLKB1*LKB1*AMPK -  kOffLKB1*LKB1_AMPK)-(kPhosLKB1*LKB1_AMPK)-(kOnAMP*AMP*LKB1_AMPK - kOffAMP*LKB1_AMP_AMPK)-(kOnADP*ADP*LKB1_AMPK - kOffADP*LKB1_ADP_AMPK)-(kOnATP*ATP*LKB1_AMPK - kOffATP*LKB1_ATP_AMPK),
    D(PP) ~ -(kOnPP*PP*pAMPK - kOffPP*PP_pAMPK)+(kDephosPP*PP_pAMPK)-(kOnPP*PP*ATP_pAMPK - kOffPP*PP_ATP_pAMPK)+(kDephosPP*PP_ATP_pAMPK)-(kOnPP*PP*AMP_pAMPK - kOffPP*PP_AMP_pAMPK)+(alphaPP*kDephosPP*PP_AMP_pAMPK)-(kOnPP*PP*ADP_pAMPK - kOffPP*PP_ADP_pAMPK)+(alphaPP*kDephosPP*PP_ADP_pAMPK),
    D(PP_pAMPK) ~ (kOnPP*PP*pAMPK - kOffPP*PP_pAMPK)-(kDephosPP*PP_pAMPK)-(kOnAMP*AMP*PP_pAMPK - kOffAMP*PP_AMP_pAMPK)-(kOnATP*ATP*PP_pAMPK - kOffATP*PP_ATP_pAMPK)-(kOnADP*ADP*PP_pAMPK - kOffADP*PP_ADP_pAMPK),
    D(PP_ATP_pAMPK) ~ (kOnPP*PP*ATP_pAMPK - kOffPP*PP_ATP_pAMPK)-(kDephosPP*PP_ATP_pAMPK)+(kOnATP*ATP*PP_pAMPK - kOffATP*PP_ATP_pAMPK),
    D(PP_AMP_pAMPK) ~ (kOnPP*PP*AMP_pAMPK - kOffPP*PP_AMP_pAMPK)-(alphaPP*kDephosPP*PP_AMP_pAMPK)+(kOnAMP*AMP*PP_pAMPK - kOffAMP*PP_AMP_pAMPK),
    D(PP_ADP_pAMPK) ~ (kOnPP*PP*ADP_pAMPK - kOffPP*PP_ADP_pAMPK)-(alphaPP*kDephosPP*PP_ADP_pAMPK)+(kOnADP*ADP*PP_pAMPK - kOffADP*PP_ADP_pAMPK),
    D(AMPKAR) ~ -(kOnAMPK*AMPKAR*AMP_pAMPK - kOffAMPK*AMPKAR_AMP_pAMPK)+(kDephosPP1*PP1_pAMPKAR)-(kOnAMPK*AMPKAR*ADP_pAMPK - kOffAMPK*AMPKAR_ADP_pAMPK)-(kOnAMPK*AMPKAR*ATP_pAMPK - kOffAMPK*AMPKAR_ATP_pAMPK)-(kOnAMPK*AMPKAR*pAMPK - kOffAMPK*AMPKAR_pAMPK),
    D(pAMPKAR) ~ (betaAMP*kPhosAMPK*AMPKAR_AMP_pAMPK)-(kOnPP1*PP1*pAMPKAR - kOffPP1*PP1_pAMPKAR)+(kPhosAMPK*AMPKAR_ADP_pAMPK)+(kPhosAMPK*AMPKAR_ATP_pAMPK)+(kPhosAMPK*AMPKAR_pAMPK),
    D(AMPKAR_pAMPK) ~ (kOnAMPK*AMPKAR*pAMPK - kOffAMPK*AMPKAR_pAMPK)-(kPhosAMPK*AMPKAR_pAMPK)-(kOnAMP*AMP*AMPKAR_pAMPK - kOffAMP*AMPKAR_AMP_pAMPK)-(kOnADP*ADP*AMPKAR_pAMPK - kOffADP*AMPKAR_ADP_pAMPK)-(kOnATP*ATP*AMPKAR_pAMPK - kOffATP*AMPKAR_ATP_pAMPK),
    D(AMPKAR_AMP_pAMPK) ~ (kOnAMPK*AMPKAR*AMP_pAMPK - kOffAMPK*AMPKAR_AMP_pAMPK)-(betaAMP*kPhosAMPK*AMPKAR_AMP_pAMPK)+(kOnAMP*AMP*AMPKAR_pAMPK - kOffAMP*AMPKAR_AMP_pAMPK),
    D(AMPKAR_ADP_pAMPK) ~ (kOnAMPK*AMPKAR*ADP_pAMPK - kOffAMPK*AMPKAR_ADP_pAMPK)-(kPhosAMPK*AMPKAR_ADP_pAMPK)+(kOnADP*ADP*AMPKAR_pAMPK - kOffADP*AMPKAR_ADP_pAMPK),
    D(AMPKAR_ATP_pAMPK) ~ (kOnAMPK*AMPKAR*ATP_pAMPK - kOffAMPK*AMPKAR_ATP_pAMPK)-(kPhosAMPK*AMPKAR_ATP_pAMPK)+(kOnATP*ATP*AMPKAR_pAMPK - kOffATP*AMPKAR_ATP_pAMPK),
    D(PP1) ~ -(kOnPP1*PP1*pAMPKAR - kOffPP1*PP1_pAMPKAR)+(kDephosPP1*PP1_pAMPKAR),
    D(PP1_pAMPKAR) ~ (kOnPP1*PP1*pAMPKAR - kOffPP1*PP1_pAMPKAR)-(kDephosPP1*PP1_pAMPKAR) 
]

measured_quantities = [
    y ~ (pAMPKAR + PP1_pAMPKAR) / (AMPKAR + AMPKAR_pAMPK + AMPKAR_AMP_pAMPK + AMPKAR_ADP_pAMPK + AMPKAR_ATP_pAMPK + pAMPKAR + PP1_pAMPKAR)
]

MA_nonessential_all = ODESystem(eqns, t, name = :MA_nonessential_all)

# Assess local identifiability with all parameters free, including metabolism parameters
local_id_all_free = assess_local_identifiability(MA_nonessential_all, funcs_to_check = [kOnAMP, kOffAMP, kOnADP, kOffADP, kOnATP, kOffATP, kOnCaMKK, kOffCaMKK, kPhosCaMKK, kOnLKB1, kOffLKB1, kPhosLKB1, kOnPP, kOffPP, kDephosPP, kOnAMPK, kOffAMPK, kPhosAMPK, kOnPP1, kOffPP1, kDephosPP1, alphaPP, betaAMP, betaCaMKK, betaLKB1],
measured_quantities = measured_quantities)

# now fix the metabolism parameters and reassess
# need to covnert MTK to SI system & then fix the values of the parameters
MA_nonessential_all_SI = mtk_to_si(MA_nonessential_all, measured_quantities)

# Note: n in the Oxphos eqn is fixed at 2 for all calculations bc the computer algebra system
# can't handle the symbolic calculations with exponents as a variable
# # set_parameters is a function from StructuralIdentifiability that fixes the 
# # specified values of the parameters in the model
# MA_nonessential_all_fixed_metab = set_parameter_values(MA_nonessential_all, known_parameters)
known_params = Dict(
    MA_nonessential_all_SI[2][kGly] => 0.5,
    MA_nonessential_all_SI[2][kHydro] => 0.15,
    MA_nonessential_all_SI[2][VforAK] => 14.66,
    MA_nonessential_all_SI[2][KeqAK] => 2.221,
    MA_nonessential_all_SI[2][kmm] => 0.32,
    MA_nonessential_all_SI[2][kmd] => 0.35,
    MA_nonessential_all_SI[2][kmt] => 0.27,
    MA_nonessential_all_SI[2][VmaxOxPhos] => 0.5,
    MA_nonessential_all_SI[2][Kadp] => 5.8e-2,
    MA_nonessential_all_SI[2][VforCK] => 1e2,
    MA_nonessential_all_SI[2][Kb] => 1.11,
    MA_nonessential_all_SI[2][Kia] => 0.135,
    MA_nonessential_all_SI[2][Kib] => 3.9,
    MA_nonessential_all_SI[2][Kiq] => 3.5,
    MA_nonessential_all_SI[2][Kp] => 3.8,
    MA_nonessential_all_SI[2][KeqCK] => 1.77e2,
    MA_nonessential_all_SI[2][TCr] => 39.0,
    MA_nonessential_all_SI[2][kOnCaM] => 7.75,
    MA_nonessential_all_SI[2][kOffCaM] => 1.0,
    MA_nonessential_all_SI[2][kPhosCaM] => 120.0,
    MA_nonessential_all_SI[2][KmCaM] => 4.0,
    MA_nonessential_all_SI[2][kDephosCaMKK] => 0.05 
)


# set_parameters is a function from StructuralIdentifiability that fixes the 
# # specified values of the parameters in the model
MA_nonessential_all_fixed_metab = set_parameter_values(MA_nonessential_all_SI[1], known_params)

funcs_to_check = [
    MA_nonessential_all_SI[2][kOnAMP],
    MA_nonessential_all_SI[2][kOffAMP],
    MA_nonessential_all_SI[2][kOnADP],
    MA_nonessential_all_SI[2][kOffADP],
    MA_nonessential_all_SI[2][kOnATP],
    MA_nonessential_all_SI[2][kOffATP],
    MA_nonessential_all_SI[2][kOnCaMKK],
    MA_nonessential_all_SI[2][kOffCaMKK],
    MA_nonessential_all_SI[2][kPhosCaMKK],
    MA_nonessential_all_SI[2][kOnLKB1],
    MA_nonessential_all_SI[2][kOffLKB1],
    MA_nonessential_all_SI[2][kPhosLKB1],
    MA_nonessential_all_SI[2][kOnPP],
    MA_nonessential_all_SI[2][kOffPP],
    MA_nonessential_all_SI[2][kDephosPP],
    MA_nonessential_all_SI[2][kOnAMPK],
    MA_nonessential_all_SI[2][kOffAMPK],
    MA_nonessential_all_SI[2][kPhosAMPK],
    MA_nonessential_all_SI[2][kOnPP1],
    MA_nonessential_all_SI[2][kOffPP1],
    MA_nonessential_all_SI[2][kDephosPP1],
    MA_nonessential_all_SI[2][alphaPP],
    MA_nonessential_all_SI[2][betaAMP],
    MA_nonessential_all_SI[2][betaCaMKK],
    MA_nonessential_all_SI[2][betaLKB1]
]

local_id_fixed_metab = assess_local_identifiability(MA_nonessential_all_fixed_metab, funcs_to_check = funcs_to_check)


# write everything to a file 
fname = "../../../results/identifiability/local_ID_MA_nonessential_all.txt"
if isfile(fname)
    rm(fname)
end

file = open(fname, "w")
println(file, "Locally Identifiable parameters with all parameters free:")
for (key, value) in local_id_all_free
    if value == 1
        print(file, "$key, ")
    end
end
println(file, "")
println(file, "Nonidentifiable parameters with all parameters free:")
for (key, value) in local_id_all_free
    if value == 0
        print(file, "$key, ")
    end
end
println(file, "")
println(file, "")
println(file, "")
println(file, "Locally Identifiable parameters with metabolism parameters fixed:")
for (key, value) in local_id_fixed_metab
    if value == 1
        print(file, "$key, ")
    end
end
println(file, "")
println(file, "Nonidentifiable parameters with metabolism parameters fixed:")
for (key, value) in local_id_fixed_metab
    if value == 0
        print(file, "$key, ")
    end
end

close(file)
