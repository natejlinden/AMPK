using StructuralIdentifiability

ampk_Coccimiglio=@ODEmodel(
    x1'(t)=(kGly*x2(t))-((((((((VforAK*(kmd*kmd))/(KeqAK*kmt*kmm))*(x2(t)*x2(t)))/(kmd*kmd))*x1(t)*x3(t))/(kmt*kmm))-((((VforAK*(kmd*kmd))/(KeqAK*kmt*kmm))*(x2(t)*x2(t)))/(kmd*kmd)))/(1+(x1(t)/kmt)+(x3(t)/kmm)+((x1(t)*x3(t))/(kmt*kmm))+((2*x2(t))/kmd)+((x2(t)*x2(t))/(kmd*kmd))))-(kHydro*x1(t))+((VmaxOxPhos*(((x2(t)*x2(t))/(Kadp*Kadp))))/(1+(((x2(t)*x2(t))/(Kadp*Kadp)))))-((((((VforCK*Kiq*Kp)/(KeqCK*Kia*Kb))*x1(t)*(TCr-x4(t)))/(Kiq*Kp))-((VforCK*x2(t)*x4(t))/(Kia*Kb)))/(1+(x2(t)/Kia)+(x4(t)/Kib)+(x1(t)/Kiq)+((x2(t)*x4(t))/(Kia*Kb))+(((TCr-x4(t))*x1(t))/(Kiq*Kp))))-(k6f*x1(t)*x12(t)-k6r*x6(t))-(k9f*x1(t)*x13(t)-k9r*x9(t)),
    x2'(t)=-(kGly*x2(t))+2*((((((((VforAK*(kmd*kmd))/(KeqAK*kmt*kmm))*(x2(t)*x2(t)))/(kmd*kmd))*x1(t)*x3(t))/(kmt*kmm))-((((VforAK*(kmd*kmd))/(KeqAK*kmt*kmm))*(x2(t)*x2(t)))/(kmd*kmd)))/(1+(x1(t)/kmt)+(x3(t)/kmm)+((x1(t)*x3(t))/(kmt*kmm))+((2*x2(t))/kmd)+((x2(t)*x2(t))/(kmd*kmd))))+(kHydro*x1(t))-((VmaxOxPhos*(((x2(t)*x2(t))/(Kadp*Kadp))))/(1+(((x2(t)*x2(t))/(Kadp*Kadp)))))+((((((VforCK*Kiq*Kp)/(KeqCK*Kia*Kb))*x1(t)*(TCr-x4(t)))/(Kiq*Kp))-((VforCK*x2(t)*x4(t))/(Kia*Kb)))/(1+(x2(t)/Kia)+(x4(t)/Kib)+(x1(t)/Kiq)+((x2(t)*x4(t))/(Kia*Kb))+(((TCr-x4(t))*x1(t))/(Kiq*Kp))))-(k7f*x2(t)*x12(t)-k7r*x7(t))-(k10f*x2(t)*x13(t)-k10r*x10(t)),
    x3'(t)=-((((((((VforAK*(kmd*kmd))/(KeqAK*kmt*kmm))*(x2(t)*x2(t)))/(kmd*kmd))*x1(t)*x3(t))/(kmt*kmm))-((((VforAK*(kmd*kmd))/(KeqAK*kmt*kmm))*(x2(t)*x2(t)))/(kmd*kmd)))/(1+(x1(t)/kmt)+(x3(t)/kmm)+((x1(t)*x3(t))/(kmt*kmm))+((2*x2(t))/kmd)+((x2(t)*x2(t))/(kmd*kmd))))-(k8f*x3(t)*x12(t)-k8r*x8(t))-(k11f*x3(t)*x13(t)-k11r*x11(t)),
    x4'(t)=((((((VforCK*Kiq*Kp)/(KeqCK*Kia*Kb))*x1(t)*(TCr-x4(t)))/(Kiq*Kp))-((VforCK*x2(t)*x4(t))/(Kia*Kb)))/(1+(x2(t)/Kia)+(x4(t)/Kib)+(x1(t)/Kiq)+((x2(t)*x4(t))/(Kia*Kb))+(((TCr-x4(t))*x1(t))/(Kiq*Kp)))),
    x5'(t)=-(kGly*x2(t))+((VmaxOxPhos*(((x2(t)*x2(t))/(Kadp*Kadp))))/(1+(((x2(t)*x2(t))/(Kadp*Kadp))))),
    x6'(t)=(k6f*x1(t)*x12(t)-k6r*x6(t))-((VmaxkinaseATP*x6(t))/(Km14+x6(t)))+((VmaxppaseATP*x9(t))/(Km15+x9(t))),
    x7'(t)=(k7f*x2(t)*x12(t)-k7r*x7(t))-((VmaxkinaseADP*x7(t))/(Km16+x7(t)))+((VmaxppaseADP*x10(t))/(Km17+x10(t))),
    x8'(t)=(k8f*x3(t)*x12(t)-k8r*x8(t))-((VmaxkinaseAMP*x8(t))/(Km18+x8(t)))+((VmaxppaseAMP*x11(t))/(Km19+x11(t))),
    x9'(t)=(k9f*x1(t)*x13(t)-k9r*x9(t))-((VmaxkinaseATP*x6(t))/(Km14+x6(t)))+((VmaxppaseATP*x9(t))/(Km15+x9(t))),
    x10'(t)=(k10f*x2(t)*x13(t)-k10r*x10(t))-((VmaxkinaseADP*x7(t))/(Km16+x7(t)))+((VmaxppaseADP*x10(t))/(Km17+x10(t))),
    x11'(t)=(k11f*x3(t)*x13(t)-k11r*x11(t))-((VmaxkinaseAMP*x8(t))/(Km18+x8(t)))+((VmaxppaseAMP*x11(t))/(Km19+x11(t))),
    x12'(t)=-(k6f*x1(t)*x12(t)-k6r*x6(t))-(k7f*x2(t)*x12(t)-k7r*x7(t))-(k8f*x3(t)*x12(t)-k8r*x8(t))-((Vmaxkinase*x12(t))/(Km12+x12(t)))+((Vmaxppase*x13(t))/(Km13+x13(t))),
    x13'(t)=-(k9f*x1(t)*x13(t)-k9r*x9(t))-(k10f*x2(t)*x13(t)-k10r*x10(t))-(k11f*x3(t)*x13(t)-k11r*x11(t))+((Vmaxkinase*x12(t))/(Km12+x12(t)))-((Vmaxppase*x13(t))/(Km13+x13(t))),
    x14'(t)=-((k_pAMPK*x14(t)*x13(t))/(Km_pAMPK+x13(t)))-((k_AMP_pAMPK*x14(t)*x11(t))/(Km_AMP_pAMPK+x11(t)))-((k_ADP_pAMPK*x14(t)*x10(t))/(Km_ADP_pAMPK+x10(t)))-((k_ATP_pAMPK*x14(t)*x9(t))/(Km_ATP_pAMPK+x9(t)))+((Vmax_AMPKAR_PP*x15(t))/(Km_AMPKAR_PP+x15(t))),
    x15'(t)=((k_pAMPK*x14(t)*x13(t))/(Km_pAMPK+x13(t)))+((k_AMP_pAMPK*x14(t)*x11(t))/(Km_AMP_pAMPK+x11(t)))+((k_ADP_pAMPK*x14(t)*x10(t))/(Km_ADP_pAMPK+x10(t)))+((k_ATP_pAMPK*x14(t)*x9(t))/(Km_ATP_pAMPK+x9(t)))-((Vmax_AMPKAR_PP*x15(t))/(Km_AMPKAR_PP+x15(t))),
    y1(t) = x15(t)/x14(t)
)

# Assess local identifiability with all parameters free, including metabolism parameters
local_id_all_free = assess_local_identifiability(ampk_Coccimiglio, funcs_to_check = [k6f,k6r,k7f,k7r,k8f,k8r,k9f,k9r,k10f,k10r,k11f,k11r,Km12,Km13,Km14,Km15,Km16,Km17,Km18,Km19,Vmaxkinase,VmaxkinaseATP,VmaxkinaseADP,VmaxkinaseAMP,Vmaxppase,VmaxppaseATP,VmaxppaseADP,VmaxppaseAMP,Km_pAMPK,k_pAMPK,Km_AMP_pAMPK,k_AMP_pAMPK,Km_ADP_pAMPK,k_ADP_pAMPK,Km_ATP_pAMPK,k_ATP_pAMPK,Km_AMPKAR_PP,Vmax_AMPKAR_PP])

# now fix the metabolism parameters and reassess
# Note: n in the Oxphos eqn is fixed at 2 for all calculations bc the computer algebra system
# can't handle the symbolic calculations with exponents as a variable
metabolism_parameters = Dict(kGly => 0.5,kHydro => 0.15,VforAK => 14.66,KeqAK => 2.221,kmm => 0.32,kmd => 0.35,kmt => 0.27,VmaxOxPhos => 0.5,Kadp => 5.8e-2,VforCK => 1e2,Kb => 1.11,Kia => 0.135,Kib => 3.9,Kiq => 3.5,Kp => 3.8,KeqCK => 1.77e2,TCr => 39.0)   

# set_parameters is a function from StructuralIdentifiability that fixes the 
# specified values of the parameters in the model
ampk_Coccimiglio_fixed_metab = set_parameter_values(ampk_Coccimiglio, metabolism_parameters)

local_id_fixed_metab = assess_local_identifiability(ampk_Coccimiglio_fixed_metab, funcs_to_check = [k6f,k6r,k7f,k7r,k8f,k8r,k9f,k9r,k10f,k10r,k11f,k11r,Km12,Km13,Km14,Km15,Km16,Km17,Km18,Km19,Vmaxkinase,VmaxkinaseATP,VmaxkinaseADP,VmaxkinaseAMP,Vmaxppase,VmaxppaseATP,VmaxppaseADP,VmaxppaseAMP,Km_pAMPK,k_pAMPK,Km_AMP_pAMPK,k_AMP_pAMPK,Km_ADP_pAMPK,k_ADP_pAMPK,Km_ATP_pAMPK,k_ATP_pAMPK,Km_AMPKAR_PP,Vmax_AMPKAR_PP])

# write everything to a file 
fname = "../../../results/identifiability/local_ID_ampk_Coccimiglio.txt"
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
