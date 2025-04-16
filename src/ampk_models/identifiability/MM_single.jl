using StructuralIdentifiability, ModelingToolkit

# model parameters
@parameters kGly, kHydro, VforAK, KeqAK, kmm, kmd, kmt, VmaxOxPhos, Kadp, n, VforCK, Kb, Kia, Kib, Kiq, Kp, KeqCK, TCr, kOnAMP, kOffAMP, kOnADP, kOffADP, kOnATP, kOffATP, kCaMKK, KmCaMKK, kLKB1, KmLKB1, kPP, KmPP, kAMPK, KmAMPK, kPP1, KmPP1, kOnCaM, kOffCaM, kPhosCaM, KmCaM, kDephosCaMKK

@independent_variables t

@variables AMP(t), ADP(t), ATP(t), PCr(t), Ca(t), CaM(t), CaCaM(t), CaMKK(t), CaMKK_act(t), AMPK(t), pAMPK(t), AMP_AMPK(t), ADP_AMPK(t), ATP_AMPK(t), AMP_pAMPK(t), ADP_pAMPK(t), ATP_pAMPK(t), AMPKAR(t), pAMPKAR(t), y(t)

# differential operator
D = Differential(t)

eqns = [
    D(AMP) ~ -(kOnAMP*AMP*AMPK - kOffAMP*AMP_AMPK)-(kOnAMP*AMP*pAMPK - kOffAMP*AMP_pAMPK)-((((VforAK*ATP*AMP)/(kmt*kmm)) - ((((VforAK*(kmd*kmd))/(KeqAK*kmt*kmm))*(ADP*ADP))/(kmd*kmd)))/(1 + (ATP/kmt) + (AMP/kmm) + ((ATP*AMP)/(kmt*kmm)) + ((2*ADP)/kmd) + ((ADP*ADP)/(kmd*kmd)))),
    D(ADP) ~ -(kOnADP*ADP*AMPK - kOffADP*ADP_AMPK)-(kOnADP*ADP*pAMPK - kOffADP*ADP_pAMPK)-kGly*ADP+2*((((VforAK*ATP*AMP)/(kmt*kmm)) - ((((VforAK*(kmd*kmd))/(KeqAK*kmt*kmm))*(ADP*ADP))/(kmd*kmd)))/(1 + (ATP/kmt) + (AMP/kmm) + ((ATP*AMP)/(kmt*kmm)) + ((2*ADP)/kmd) + ((ADP*ADP)/(kmd*kmd))))+kHydro*ATP-((VmaxOxPhos * ((ADP/Kadp)*(ADP/Kadp)))/(1 + ((ADP/Kadp)*(ADP/Kadp))))+(((((VforCK*Kiq*Kp)/(KeqCK*Kia*Kb))*ATP*(TCr - PCr))/(Kiq*Kp)) - ((VforCK*ADP*PCr)/(Kia*Kb)))/(1 + (ADP/Kia) + (PCr/Kib) + (ATP/Kiq) + ((ADP*PCr)/(Kia*Kb)) + (((TCr - PCr)*ATP)/(Kiq*Kp))),
    D(ATP) ~ -(kOnATP*ATP*AMPK - kOffATP*ATP_AMPK)-(kOnATP*ATP*pAMPK - kOffATP*ATP_pAMPK)+kGly*ADP-((((VforAK*ATP*AMP)/(kmt*kmm)) - ((((VforAK*(kmd*kmd))/(KeqAK*kmt*kmm))*(ADP*ADP))/(kmd*kmd)))/(1 + (ATP/kmt) + (AMP/kmm) + ((ATP*AMP)/(kmt*kmm)) + ((2*ADP)/kmd) + ((ADP*ADP)/(kmd*kmd))))-kHydro*ATP+((VmaxOxPhos * ((ADP/Kadp)*(ADP/Kadp)))/(1 + ((ADP/Kadp)*(ADP/Kadp))))-(((((VforCK*Kiq*Kp)/(KeqCK*Kia*Kb))*ATP*(TCr - PCr))/(Kiq*Kp)) - ((VforCK*ADP*PCr)/(Kia*Kb)))/(1 + (ADP/Kia) + (PCr/Kib) + (ATP/Kiq) + ((ADP*PCr)/(Kia*Kb)) + (((TCr - PCr)*ATP)/(Kiq*Kp))),
    D(PCr) ~ (((((VforCK*Kiq*Kp)/(KeqCK*Kia*Kb))*ATP*(TCr - PCr))/(Kiq*Kp)) - ((VforCK*ADP*PCr)/(Kia*Kb)))/(1 + (ADP/Kia) + (PCr/Kib) + (ATP/Kiq) + ((ADP*PCr)/(Kia*Kb)) + (((TCr - PCr)*ATP)/(Kiq*Kp))),
    D(Ca) ~ -(kOnCaM*(Ca*Ca*Ca)*CaM - kOffCaM*CaCaM),
    D(CaM) ~ -(kOnCaM*(Ca*Ca*Ca)*CaM - kOffCaM*CaCaM),
    D(CaCaM) ~ (kOnCaM*(Ca*Ca*Ca)*CaM - kOffCaM*CaCaM),
    D(CaMKK) ~ -((kPhosCaM*(CaCaM*CaCaM*CaCaM*CaCaM)*CaMKK)/(KmCaM*KmCaM*KmCaM*KmCaM + CaCaM*CaCaM*CaCaM*CaCaM)) + (kDephosCaMKK*CaMKK_act),
    D(CaMKK_act) ~ ((kPhosCaM*(CaCaM*CaCaM*CaCaM*CaCaM)*CaMKK)/(KmCaM*KmCaM*KmCaM*KmCaM + CaCaM*CaCaM*CaCaM*CaCaM)) - (kDephosCaMKK*CaMKK_act),
    D(AMPK) ~ -(kOnAMP*AMP*AMPK - kOffAMP*AMP_AMPK)-(kOnADP*ADP*AMPK - kOffADP*ADP_AMPK)-(kOnATP*ATP*AMPK - kOffATP*ATP_AMPK)-((kCaMKK*CaMKK_act*AMPK)/(KmCaMKK + AMPK))+((kPP*pAMPK)/(KmPP + pAMPK)),
    D(pAMPK) ~ -(kOnAMP*AMP*pAMPK - kOffAMP*AMP_pAMPK)-(kOnADP*ADP*pAMPK - kOffADP*ADP_pAMPK)-(kOnATP*ATP*pAMPK - kOffATP*ATP_pAMPK)+((kCaMKK*CaMKK_act*AMPK)/(KmCaMKK + AMPK))-((kPP*pAMPK)/(KmPP + pAMPK)),
    D(AMP_AMPK) ~ (kOnAMP*AMP*AMPK - kOffAMP*AMP_AMPK)-((kCaMKK*CaMKK_act*AMP_AMPK)/(KmCaMKK + AMP_AMPK))-((kLKB1*AMP_AMPK)/(KmLKB1 + AMP_AMPK)),
    D(ADP_AMPK) ~ (kOnADP*ADP*AMPK - kOffADP*ADP_AMPK)-((kCaMKK*CaMKK_act*ADP_AMPK)/(KmCaMKK + ADP_AMPK))-((kLKB1*ADP_AMPK)/(KmLKB1 + ADP_AMPK)),
    D(ATP_AMPK) ~ (kOnATP*ATP*AMPK - kOffATP*ATP_AMPK)-((kCaMKK*CaMKK_act*ATP_AMPK)/(KmCaMKK + ATP_AMPK))+((kPP*ATP_pAMPK)/(KmPP + ATP_pAMPK)),
    D(AMP_pAMPK) ~ (kOnAMP*AMP*pAMPK - kOffAMP*AMP_pAMPK)+((kCaMKK*CaMKK_act*AMP_AMPK)/(KmCaMKK + AMP_AMPK))+((kLKB1*AMP_AMPK)/(KmLKB1 + AMP_AMPK)),
    D(ADP_pAMPK) ~ (kOnADP*ADP*pAMPK - kOffADP*ADP_pAMPK)+((kCaMKK*CaMKK_act*ADP_AMPK)/(KmCaMKK + ADP_AMPK))+((kLKB1*ADP_AMPK)/(KmLKB1 + ADP_AMPK)),
    D(ATP_pAMPK) ~ (kOnATP*ATP*pAMPK - kOffATP*ATP_pAMPK)+((kCaMKK*CaMKK_act*ATP_AMPK)/(KmCaMKK + ATP_AMPK))-((kPP*ATP_pAMPK)/(KmPP + ATP_pAMPK)),
    D(AMPKAR) ~ -((kAMPK*AMP_pAMPK*AMPKAR)/(KmAMPK + AMPKAR))+((kPP1*pAMPKAR)/(KmPP1 + pAMPKAR)),
    D(pAMPKAR) ~ ((kAMPK*AMP_pAMPK*AMPKAR)/(KmAMPK + AMPKAR))-((kPP1*pAMPKAR)/(KmPP1 + pAMPKAR))
];

measured_quantities = [
    y ~ pAMPKAR / (AMPKAR + pAMPKAR)
]

MM_single = ODESystem(eqns, t, name = :MM_single)

# Assess local identifiability with all parameters free, including metabolism parameters
local_id_all_free = assess_local_identifiability(MM_single, funcs_to_check = [kOnAMP,kOffAMP,kOnADP,kOffADP,kOnATP,kOffATP,kCaMKK,KmCaMKK,kLKB1,KmLKB1,kPP,KmPP,kAMPK,KmAMPK,kPP1,KmPP1], measured_quantities = measured_quantities)

# now fix the calcium & metabolism parameters and reassess
# need to covnert MTK to SI system & then fix the values of the parameters
MM_single_SI = mtk_to_si(MM_single, measured_quantities)

# Note: n in the Oxphos eqn is fixed at 2 for all calculations bc the computer algebra system
# can't handle the symbolic calculations with exponents as a variable
# metabolism_parameters = Dict(kGly => 0.5,kHydro => 0.15,VforAK => 14.66,KeqAK => 2.221,kmm => 0.32,kmd => 0.35,kmt => 0.27,VmaxOxPhos => 0.5,Kadp => 5.8e-2,VforCK => 1e2,Kb => 1.11,Kia => 0.135,Kib => 3.9,Kiq => 3.5,Kp => 3.8,KeqCK => 1.77e2,TCr => 39.0)  
known_params = Dict(
    MM_single_SI[2][kGly] => 0.5,
    MM_single_SI[2][kHydro] => 0.15,
    MM_single_SI[2][VforAK] => 14.66,
    MM_single_SI[2][KeqAK] => 2.221,
    MM_single_SI[2][kmm] => 0.32,
    MM_single_SI[2][kmd] => 0.35,
    MM_single_SI[2][kmt] => 0.27,
    MM_single_SI[2][VmaxOxPhos] => 0.5,
    MM_single_SI[2][Kadp] => 5.8e-2,
    MM_single_SI[2][VforCK] => 1e2,
    MM_single_SI[2][Kb] => 1.11,
    MM_single_SI[2][Kia] => 0.135,
    MM_single_SI[2][Kib] => 3.9,
    MM_single_SI[2][Kiq] => 3.5,
    MM_single_SI[2][Kp] => 3.8,
    MM_single_SI[2][KeqCK] => 1.77e2,
    MM_single_SI[2][TCr] => 39.0,
    MM_single_SI[2][kOnCaM] => 7.75,
    MM_single_SI[2][kOffCaM] => 1.0,
    MM_single_SI[2][kPhosCaM] => 120.0,
    MM_single_SI[2][KmCaM] => 4.0,
    MM_single_SI[2][kDephosCaMKK] => 0.05 
)

# set_parameters is a function from StructuralIdentifiability that fixes the 
# # specified values of the parameters in the model
MM_single_fixed_metab = set_parameter_values(MM_single_SI[1], known_params)

funcs_to_check = [
    MM_single_SI[2][kOnAMP],
    MM_single_SI[2][kOffAMP],
    MM_single_SI[2][kOnADP],
    MM_single_SI[2][kOffADP],
    MM_single_SI[2][kOnATP],
    MM_single_SI[2][kOffATP],
    MM_single_SI[2][kCaMKK],
    MM_single_SI[2][KmCaMKK],
    MM_single_SI[2][kLKB1],
    MM_single_SI[2][KmLKB1],
    MM_single_SI[2][kPP],
    MM_single_SI[2][KmPP],
    MM_single_SI[2][kAMPK],
    MM_single_SI[2][KmAMPK],
    MM_single_SI[2][kPP1],
    MM_single_SI[2][KmPP1]
]

local_id_fixed_metab = assess_local_identifiability(MM_single_fixed_metab, funcs_to_check = funcs_to_check)

# write everything to a file 
fname = "../../../results/identifiability/local_ID_MM_single.txt"
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
