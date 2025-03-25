using StructuralIdentifiability

MA_amp_adp_dep = @ODEmodel(
        x1'(t) = -(kOnAMP*x1(t)*x5(t)-kOffAMP*x7(t))-(kOnAMP*x1(t)*x6(t)-kOffAMP*x10(t))-((((((((VforAK*(kmd*kmd))/(KeqAK*kmt*kmm))*(x2(t)*x2(t)))/(kmd*kmd))*x3(t)*x1(t))/(kmt*kmm))-((((VforAK*(kmd*kmd))/(KeqAK*kmt*kmm))*(x2(t)*x2(t)))/(kmd*kmd)))/(1+(x3(t)/kmt)+(x1(t)/kmm)+((x3(t)*x1(t))/(kmt*kmm))+((2*x2(t))/kmd)+((x2(t)*x2(t))/(kmd*kmd)))),
        x2'(t) = -(kOnADP*x2(t)*x5(t)-kOffADP*x8(t))-(kOnADP*x2(t)*x6(t)-kOffADP*x11(t))-(kGly*x2(t))+2*((((((((VforAK*(kmd*kmd))/(KeqAK*kmt*kmm))*(x2(t)*x2(t)))/(kmd*kmd))*x3(t)*x1(t))/(kmt*kmm))-((((VforAK*(kmd*kmd))/(KeqAK*kmt*kmm))*(x2(t)*x2(t)))/(kmd*kmd)))/(1+(x3(t)/kmt)+(x1(t)/kmm)+((x3(t)*x1(t))/(kmt*kmm))+((2*x2(t))/kmd)+((x2(t)*x2(t))/(kmd*kmd))))+(kHydro*x3(t))-((VmaxOxPhos*(((x2(t)*x2(t))/(Kadp*Kadp))))/(1+(((x2(t)*x2(t))/(Kadp*Kadp)))))+((((((VforCK*Kiq*Kp)/(KeqCK*Kia*Kb))*x3(t)*(TCr-x4(t)))/(Kiq*Kp))-((VforCK*x2(t)*x4(t))/(Kia*Kb)))/(1+(x2(t)/Kia)+(x4(t)/Kib)+(x3(t)/Kiq)+((x2(t)*x4(t))/(Kia*Kb))+(((TCr-x4(t))*x3(t))/(Kiq*Kp)))),
        x3'(t) = -(kOnATP*x3(t)*x5(t)-kOffATP*x9(t))-(kOnATP*x3(t)*x6(t)-kOffATP*x12(t))+(kGly*x2(t))-((((((((VforAK*(kmd*kmd))/(KeqAK*kmt*kmm))*(x2(t)*x2(t)))/(kmd*kmd))*x3(t)*x1(t))/(kmt*kmm))-((((VforAK*(kmd*kmd))/(KeqAK*kmt*kmm))*(x2(t)*x2(t)))/(kmd*kmd)))/(1+(x3(t)/kmt)+(x1(t)/kmm)+((x3(t)*x1(t))/(kmt*kmm))+((2*x2(t))/kmd)+((x2(t)*x2(t))/(kmd*kmd))))-(kHydro*x3(t))+((VmaxOxPhos*(((x2(t)*x2(t))/(Kadp*Kadp))))/(1+(((x2(t)*x2(t))/(Kadp*Kadp)))))-((((((VforCK*Kiq*Kp)/(KeqCK*Kia*Kb))*x3(t)*(TCr-x4(t)))/(Kiq*Kp))-((VforCK*x2(t)*x4(t))/(Kia*Kb)))/(1+(x2(t)/Kia)+(x4(t)/Kib)+(x3(t)/Kiq)+((x2(t)*x4(t))/(Kia*Kb))+(((TCr-x4(t))*x3(t))/(Kiq*Kp))))-(kOnATP*x3(t)*x20(t)-kOffATP*x21(t)),
        x4'(t) = ((((((VforCK*Kiq*Kp)/(KeqCK*Kia*Kb))*x3(t)*(TCr-x4(t)))/(Kiq*Kp))-((VforCK*x2(t)*x4(t))/(Kia*Kb)))/(1+(x2(t)/Kia)+(x4(t)/Kib)+(x3(t)/Kiq)+((x2(t)*x4(t))/(Kia*Kb))+(((TCr-x4(t))*x3(t))/(Kiq*Kp)))),
        x5'(t) = -(kOnAMP*x1(t)*x5(t)-kOffAMP*x7(t))-(kOnADP*x2(t)*x5(t)-kOffADP*x8(t))-(kOnATP*x3(t)*x5(t)-kOffATP*x9(t))+(kDephosPP*x20(t)),
        x6'(t) = -(kOnAMP*x1(t)*x6(t)-kOffAMP*x10(t))-(kOnADP*x2(t)*x6(t)-kOffADP*x11(t))-(kOnATP*x3(t)*x6(t)-kOffATP*x12(t))-(kOnPP*x19(t)*x6(t)-kOffPP*x20(t)),
        x7'(t) = (kOnAMP*x1(t)*x5(t)-kOffAMP*x7(t))-(kOnCaMKK*x13(t)*x7(t)-kOffCaMKK*x14(t))-(kOnLKB1*x16(t)*x7(t)-kOffLKB1*x17(t)),
        x8'(t) = (kOnADP*x2(t)*x5(t)-kOffADP*x8(t))-(kOnCaMKK*x13(t)*x8(t)-kOffCaMKK*x15(t))-(kOnLKB1*x16(t)*x8(t)- kOffLKB1*x18(t)),
        x9'(t) = (kOnATP*x3(t)*x5(t)-kOffATP*x9(t))+(kDephosPP*x21(t)),
        x10'(t) = (kOnAMP*x1(t)*x6(t)-kOffAMP*x10(t))+(kPhosCaMKK*x14(t))+(kPhosLKB1*x17(t))-(kOnAMPK*x22(t)*x10(t)-kOffAMPK*x24(t))+(kPhosAMPK*x24(t)),
        x11'(t) = (kOnADP*x2(t)*x6(t)-kOffADP*x11(t))+(kPhosCaMKK*x15(t))+(kPhosLKB1*x18(t)),
        x12'(t) = (kOnATP*x3(t)*x6(t)-kOffATP*x12(t))-(kOnPP*x19(t)*x12(t)-kOffPP*x21(t)),
        x13'(t) = -(kOnCaMKK*x13(t)*x7(t)-kOffCaMKK*x14(t))+(kPhosCaMKK*x14(t))-(kOnCaMKK*x13(t)*x8(t)-kOffCaMKK*x15(t))+(kPhosCaMKK*x15(t)),
        x14'(t) = (kOnCaMKK*x13(t)*x7(t)-kOffCaMKK*x14(t))-(kPhosCaMKK*x14(t)),
        x15'(t) = (kOnCaMKK*x13(t)*x8(t)-kOffCaMKK*x15(t))-(kPhosCaMKK*x15(t)),
        x16'(t) = -(kOnLKB1*x16(t)*x7(t)-kOffLKB1*x17(t))+(kPhosLKB1*x17(t))-(kOnLKB1*x16(t)*x8(t)- kOffLKB1*x18(t))+(kPhosLKB1*x18(t)),
        x17'(t) = (kOnLKB1*x16(t)*x7(t)-kOffLKB1*x17(t))-(kPhosLKB1*x17(t)),
        x18'(t) = (kOnLKB1*x16(t)*x8(t)- kOffLKB1*x18(t))-(kPhosLKB1*x18(t)),
        x19'(t) = -(kOnPP*x19(t)*x6(t)-kOffPP*x20(t))+(kDephosPP*x20(t))-(kOnPP*x19(t)*x12(t)-kOffPP*x21(t))+(kDephosPP*x21(t)),
        x20'(t) = (kOnPP*x19(t)*x6(t)-kOffPP*x20(t))-(kDephosPP*x20(t))-(kOnATP*x3(t)*x20(t)-kOffATP*x21(t)),
        x21'(t) = (kOnPP*x19(t)*x12(t)-kOffPP*x21(t))-(kDephosPP*x21(t))+(kOnATP*x3(t)*x20(t)-kOffATP*x21(t)),
        x22'(t) = -(kOnAMPK*x22(t)*x10(t)-kOffAMPK*x24(t))+(kDephosxPP1*x26(t)),
        x23'(t) = (kPhosAMPK*x24(t))-(kOnPP1*x25(t)*x23(t)-kOffPP1*x26(t)),
        x24'(t) = (kOnAMPK*x22(t)*x10(t)-kOffAMPK*x24(t))-(kPhosAMPK*x24(t)),
        x25'(t) = -(kOnPP1*x25(t)*x23(t)-kOffPP1*x26(t))+(kDephosPP1*x26(t)),
        x26'(t) = (kOnPP1*x25(t)*x23(t)-kOffPP1*x26(t))-(kDephosPP1*x26(t)),
        y1(t) = (x26(t) + x23(t))/(x24(t) + x22(t))
)

# Assess local identifiability with all parameters free, including metabolism parameters
local_id_all_free = assess_local_identifiability(MA_amp_adp_dep, funcs_to_check = [kOnAMP,kOffAMP,kOnADP,kOffADP,kOnATP,kOffATP,kOnCaMKK,kOffCaMKK,kPhosCaMKK,kOnLKB1,kOffLKB1,kPhosLKB1,kOnPP,kOffPP,kDephosPP,kOnAMPK,kOffAMPK,kPhosAMPK,kOnPP1,kOffPP1,kDephosPP1])

# now fix the metabolism parameters and reassess
# Note: n in the Oxphos eqn is fixed at 2 for all calculations bc the computer algebra system
# can't handle the symbolic calculations with exponents as a variable
metabolism_parameters = Dict(kGly => 0.5,kHydro => 0.15,VforAK => 14.66,KeqAK => 2.221,kmm => 0.32,kmd => 0.35,kmt => 0.27,VmaxOxPhos => 0.5,Kadp => 5.8e-2,VforCK => 1e2,Kb => 1.11,Kia => 0.135,Kib => 3.9,Kiq => 3.5,Kp => 3.8,KeqCK => 1.77e2,TCr => 39.0)   

# set_parameters is a function from StructuralIdentifiability that fixes the 
# specified values of the parameters in the model
MA_amp_adp_dep_fixed_metab = set_parameter_values(MA_amp_adp_dep, metabolism_parameters)

local_id_fixed_metab = assess_local_identifiability(MA_amp_adp_dep_fixed_metab, funcs_to_check = [kOnAMP,kOffAMP,kOnADP,kOffADP,kOnATP,kOffATP,kOnCaMKK,kOffCaMKK,kPhosCaMKK,kOnLKB1,kOffLKB1,kPhosLKB1,kOnPP,kOffPP,kDephosPP,kOnAMPK,kOffAMPK,kPhosAMPK,kOnPP1,kOffPP1,kDephosPP1])

# write everything to a file 
fname = "../../../results/identifiability/local_ID_MA_amp_adp_dep.txt"
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
