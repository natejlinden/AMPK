using StructuralIdentifiability


MM_nonessential = @ODEmodel(
    x1'(t) = -((((((((VforAK*(kmd*kmd))/(KeqAK*kmt*kmm))*(x2(t)*x2(t)))/(kmd*kmd))*x3(t)*x1(t))/(kmt*kmm))-((((VforAK*(kmd*kmd))/(KeqAK*kmt*kmm))*(x2(t)*x2(t)))/(kmd*kmd)))/(1+(x3(t)/kmt)+(x1(t)/kmm)+((x3(t)*x1(t))/(kmt*kmm))+((2*x2(t))/kmd)+((x2(t)*x2(t))/(kmd*kmd))))-(kOnAMP*x1(t)*x5(t)-kOffAMP*x7(t))-(kOnAMP*x1(t)*x6(t)-kOffAMP*x10(t)),
    x2'(t) = -(kGly*x2(t))+2*((((((((VforAK*(kmd*kmd))/(KeqAK*kmt*kmm))*(x2(t)*x2(t)))/(kmd*kmd))*x3(t)*x1(t))/(kmt*kmm))-((((VforAK*(kmd*kmd))/(KeqAK*kmt*kmm))*(x2(t)*x2(t)))/(kmd*kmd)))/(1+(x3(t)/kmt)+(x1(t)/kmm)+((x3(t)*x1(t))/(kmt*kmm))+((2*x2(t))/kmd)+((x2(t)*x2(t))/(kmd*kmd))))+(kHydro*x3(t))-((VmaxOxPhos*(((x2(t)*x2(t))/(Kadp*Kadp))))/(1+(((x2(t)*x2(t))/(Kadp*Kadp))))) + ((((((VforCK*Kiq*Kp)/(KeqCK*Kia*Kb))*x3(t)*(TCr-x4(t)))/(Kiq*Kp))-((VforCK*x2(t)*x4(t))/(Kia*Kb)))/(1+(x2(t)/Kia)+(x4(t)/Kib)+(x3(t)/Kiq)+((x2(t)*x4(t))/(Kia*Kb))+(((TCr-x4(t))*x3(t))/(Kiq*Kp))))-(kOnADP*x2(t)*x5(t)-kOffADP*x8(t))-(kOnADP*x2(t)*x6(t)-kOffADP*x11(t)),
    x3'(t) = (kGly*x2(t))-((((((((VforAK*(kmd*kmd))/(KeqAK*kmt*kmm))*(x2(t)*x2(t)))/(kmd*kmd))*x3(t)*x1(t))/(kmt*kmm))-((((VforAK*(kmd*kmd))/(KeqAK*kmt*kmm))*(x2(t)*x2(t)))/(kmd*kmd)))/(1+(x3(t)/kmt)+(x1(t)/kmm)+((x3(t)*x1(t))/(kmt*kmm))+((2*x2(t))/kmd)+((x2(t)*x2(t))/(kmd*kmd))))-(kHydro*x3(t))+((VmaxOxPhos*(((x2(t)*x2(t))/(Kadp*Kadp))))/(1+(((x2(t)*x2(t))/(Kadp*Kadp)))))-((((((VforCK*Kiq*Kp)/(KeqCK*Kia*Kb))*x3(t)*(TCr-x4(t)))/(Kiq*Kp))-((VforCK*x2(t)*x4(t))/(Kia*Kb)))/(1+(x2(t)/Kia)+(x4(t)/Kib)+(x3(t)/Kiq)+((x2(t)*x4(t))/(Kia*Kb))+(((TCr-x4(t))*x3(t))/(Kiq*Kp))))-(kOnATP*x3(t)*x5(t)-kOffATP*x9(t))-(kOnATP*x3(t)*x6(t)-kOffATP*x12(t)),
    x4'(t) = ((((((VforCK*Kiq*Kp)/(KeqCK*Kia*Kb))*x3(t)*(TCr-x4(t)))/(Kiq*Kp))-((VforCK*x2(t)*x4(t))/(Kia*Kb)))/(1+(x2(t)/Kia)+(x4(t)/Kib)+(x3(t)/Kiq)+((x2(t)*x4(t))/(Kia*Kb))+(((TCr-x4(t))*x3(t))/(Kiq*Kp)))),
    x5'(t) = -(kOnAMP*x1(t)*x5(t)-kOffAMP*x7(t))-(kOnADP*x2(t)*x5(t)-kOffADP*x8(t))-(kOnATP*x3(t)*x5(t)-kOffATP*x9(t))-((kPhosCaMKK*x5(t))/(KmCaMKK + x5(t)))-((kPhosLKB1*x5(t))/(KmLKB1 + x5(t)))+((kDephosPP*x6(t))/(KmPP + x6(t))),
    x6'(t) = -(kOnAMP*x1(t)*x6(t)-kOffAMP*x10(t))-(kOnADP*x2(t)*x6(t)-kOffADP*x11(t))-(kOnATP*x3(t)*x6(t)-kOffATP*x12(t))+((kPhosCaMKK*x5(t))/(KmCaMKK + x5(t)))+((kPhosLKB1*x5(t))/(KmLKB1 + x5(t)))-((kDephosPP*x6(t))/(KmPP + x6(t))),
    x7'(t) = (kOnAMP*x1(t)*x5(t)-kOffAMP*x7(t))-((kPhosCaMKK*x7(t))/(KmCaMKK + x7(t)))-((kPhosLKB1*x7(t))/(alphaLKB1*KmLKB1 + x7(t)))+((kDephosPP*x10(t))/(alphaPP*KmPP + x10(t))),
    x8'(t) = (kOnADP*x2(t)*x5(t)-kOffADP*x8(t))-((kPhosCaMKK*x8(t))/(KmCaMKK + x8(t)))-((kPhosLKB1*x8(t))/(alphaLKB1*KmLKB1 + x8(t)))+((kDephosPP*x11(t))/(alphaPP*KmPP + x11(t))),
    x9'(t) = (kOnATP*x3(t)*x5(t)-kOffATP*x9(t))-((kPhosCaMKK*x9(t))/(KmCaMKK + x9(t)))+((kDephosPP*x12(t))/(KmPP + x12(t))),
    x10'(t) = (kOnAMP*x1(t)*x6(t)-kOffAMP*x10(t))+((kPhosCaMKK*x7(t))/(KmCaMKK + x7(t)))+((kPhosLKB1*x7(t))/(alphaLKB1*KmLKB1 + x7(t)))-((kDephosPP*x10(t))/(alphaPP*KmPP + x10(t))),
    x11'(t) = (kOnADP*x2(t)*x6(t)-kOffADP*x11(t))+((kPhosCaMKK*x8(t))/(KmCaMKK + x8(t)))+((kPhosLKB1*x8(t))/(alphaLKB1*KmLKB1 + x8(t)))-((kDephosPP*x11(t))/(alphaPP*KmPP + x11(t))),
    x12'(t) = (kOnATP*x3(t)*x6(t)-kOffATP*x12(t))+((kPhosCaMKK*x9(t))/(KmCaMKK + x9(t)))-((kDephosPP*x12(t))/(KmPP + x12(t))),
    x13'(t) = -((kPhosAMPK*x6(t)*x13(t))/(KmAMPK + x13(t)))-((betaAMP*kPhosAMPK*x10(t)*x13(t))/(KmAMPK + x13(t)))-((kPhosAMPK*x11(t)*x13(t))/(KmAMPK + x13(t)))+((kDephosPP1*x14(t))/(KmPP1 + x14(t))),
    x14'(t) = ((kPhosAMPK*x6(t)*x13(t))/(KmAMPK + x13(t)))+((betaAMP*kPhosAMPK*x10(t)*x13(t))/(KmAMPK + x13(t)))+((kPhosAMPK*x11(t)*x13(t))/(KmAMPK + x13(t)))-((kDephosPP1*x14(t))/(KmPP1 + x14(t))),
    y1(t) = x14(t)/x13(t)  
)

# Assess local identifiability with all parameters free, including metabolism parameters
local_id_all_free = assess_local_identifiability(MM_nonessential, funcs_to_check = [kOnAMP,kOffAMP,kOnADP,kOffADP,kOnATP,kOffATP,kPhosCaMKK,KmCaMKK,kPhosLKB1,KmLKB1,kDephosPP,KmPP,kPhosAMPK,KmAMPK,kDephosPP1,KmPP1,alphaLKB1,alphaPP,betaAMP])

# now fix the metabolism parameters and reassess
# Note: n in the Oxphos eqn is fixed at 2 for all calculations bc the computer algebra system
# can't handle the symbolic calculations with exponents as a variable
metabolism_parameters = Dict(kGly => 0.5,kHydro => 0.15,VforAK => 14.66,KeqAK => 2.221,kmm => 0.32,kmd => 0.35,kmt => 0.27,VmaxOxPhos => 0.5,Kadp => 5.8e-2,VforCK => 1e2,Kb => 1.11,Kia => 0.135,Kib => 3.9,Kiq => 3.5,Kp => 3.8,KeqCK => 1.77e2,TCr => 39.0)   

# set_parameters is a function from StructuralIdentifiability that fixes the 
# specified values of the parameters in the model
MM_nonessential_fixed_metab = set_parameter_values(MM_nonessential, metabolism_parameters)

local_id_fixed_metab = assess_local_identifiability(MM_nonessential_fixed_metab, funcs_to_check = [kOnAMP,kOffAMP,kOnADP,kOffADP,kOnATP,kOffATP,kPhosCaMKK,KmCaMKK,kPhosLKB1,KmLKB1,kDephosPP,KmPP,kPhosAMPK,KmAMPK,kDephosPP1,KmPP1,alphaLKB1,alphaPP,betaAMP])

# write everything to a file 
fname = "../../../results/identifiability/local_ID_MM_nonessential.txt"
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