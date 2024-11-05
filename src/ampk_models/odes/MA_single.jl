using StructuralIdentifiability

MA_single = @ODEmodel(
        x1'(t) = -(kOnAMP*x1(t)*x5(t)-kOffAMP*x7(t))-(kOnAMP*x1(t)*x6(t)-kOffAMP*x10(t))-((((((((VforAK*(kmd*kmd))/(KeqAK*kmt*kmm))*(x2(t)*x2(t)))/(kmd*kmd))*x3(t)*x1(t))/(kmt*kmm))-num_rev)/(1+(x3(t)/kmt)+(x1(t)/kmm)+((x3(t)*x1(t))/(kmt*kmm))+((2*x2(t))/kmd)+((x2(t)*x2(t))/(kmd*kmd)))),
        x2'(t) = -(kOnADP*x2(t)*x5(t)-kOffADP*x8(t))-(kOnADP*x2(t)*x6(t)-kOffADP*x11(t))-(kGly*x2(t))+2*((((((((VforAK*(kmd*kmd))/(KeqAK*kmt*kmm))*(x2(t)*x2(t)))/(kmd*kmd))*x3(t)*x1(t))/(kmt*kmm))-num_rev)/(1+(x3(t)/kmt)+(x1(t)/kmm)+((x3(t)*x1(t))/(kmt*kmm))+((2*x2(t))/kmd)+((x2(t)*x2(t))/(kmd*kmd))))+(kHydro*x3(t))-((VmaxOxPhos*(((x2(t)*x2(t))/(Kadp*Kapd))))/(1+(((x2(t)*x2(t))/(Kadp*Kapd)))))+((((((VforCK*Kiq*Kp)/(KeqCK*Kia*Kb))*x3(t)*(TCr-x4(t)))/(Kiq*Kp))-((VforCK*x2(t)*x4(t))/(Kia*Kb)))/(1+(x2(t)/Kia)+(x4(t)/Kib)+(x3(t)/Kiq)+((x2(t)*x4(t))/(Kia*Kb))+(((TCr-x4(t))*x3(t))/(Kiq*Kp)))),
        x3'(t) = -(kOnATP*x3(t)*x5(t)-kOffATP*x9(t))-(kOnATP*x3(t)*x6(t)-kOffATP*x12(t))+(kGly*x2(t))-((((((((VforAK*(kmd*kmd))/(KeqAK*kmt*kmm))*(x2(t)*x2(t)))/(kmd*kmd))*x3(t)*x1(t))/(kmt*kmm))-num_rev)/(1+(x3(t)/kmt)+(x1(t)/kmm)+((x3(t)*x1(t))/(kmt*kmm))+((2*x2(t))/kmd)+((x2(t)*x2(t))/(kmd*kmd))))-(kHydro*x3(t))+((VmaxOxPhos*(((x2(t)*x2(t))/(Kadp*Kapd))))/(1+(((x2(t)*x2(t))/(Kadp*Kapd)))))-((((((VforCK*Kiq*Kp)/(KeqCK*Kia*Kb))*x3(t)*(TCr-x4(t)))/(Kiq*Kp))-((VforCK*x2(t)*x4(t))/(Kia*Kb)))/(1+(x2(t)/Kia)+(x4(t)/Kib)+(x3(t)/Kiq)+((x2(t)*x4(t))/(Kia*Kb))+(((TCr-x4(t))*x3(t))/(Kiq*Kp)))),
        x4'(t) = ((((((VforCK*Kiq*Kp)/(KeqCK*Kia*Kb))*x3(t)*(TCr-x4(t)))/(Kiq*Kp))-((VforCK*x2(t)*x4(t))/(Kia*Kb)))/(1+(x2(t)/Kia)+(x4(t)/Kib)+(x3(t)/Kiq)+((x2(t)*x4(t))/(Kia*Kb))+(((TCr-x4(t))*x3(t))/(Kiq*Kp)))),
        x5'(t) = -(kOnAMP*x1(t)*x5(t)-kOffAMP*x7(t))-(kOnADP*x2(t)*x5(t)-kOffADP*x8(t))-(kOnATP*x3(t)*x5(t)-kOffATP*x9(t))-(kOnCaMKK*x13(t)*x5(t)-kOffCaMKK*x14(t))+(kDephosPP*x22(t)),
        x6'(t) = -(kOnAMP*x1(t)*x6(t)-kOffAMP*x10(t))-(kOnADP*x2(t)*x6(t)-kOffADP*x11(t))-(kOnATP*x3(t)*x6(t)-kOffATP*x12(t))+(kPhosCaMKK*x14(t))-(kOnPP*x21(t)*x6(t)-kOffPP*x22(t)),
        x7'(t) = (kOnAMP*x1(t)*x5(t)-kOffAMP*x7(t))-(kOnCaMKK*x13(t)*x7(t)-kOffCaMKK*x15(t))-(kOnLKB1*x18(t)*x7(t)-kOffLKB1*x19(t)),
        x8'(t) = (kOnADP*x2(t)*x5(t)-kOffADP*x8(t))-(kOnCaMKK*x13(t)*x8(t)-kOffCaMKK*x16(t))-(kOnLKB1*x18(t)*x8(t)- kOffLKB1*x20(t)),
        x9'(t) = (kOnATP*x3(t)*x5(t)-kOffATP*x9(t))-(kOnCaMKK*x13(t)*x9(t)-kOffCaMKK*x17(t))+(kDephosPP*x23(t)),
        x10'(t) = (kOnAMP*x1(t)*x6(t)-kOffAMP*x10(t))+(kPhosCaMKK*x15(t))+(kPhosLKB1*x19(t))-(kOnAMPK*x24(t)*x10(t)-kOffAMPK*x26(t))+(kPhosAMPK*x26(t)),
        x11'(t) = (kOnADP*x2(t)*x6(t)-kOffADP*x11(t))+(kPhosCaMKK*x16(t))+(kPhosLKB1*x20(t)),
        x12'(t) = (kOnATP*x3(t)*x6(t)-kOffATP*x12(t))+(kPhosCaMKK*x17(t))-(kOnPP*x21(t)*x12(t)-kOffPP*x23(t)),
        x13'(t) = -(kOnCaMKK*x13(t)*x5(t)-kOffCaMKK*x14(t))+(kPhosCaMKK*x14(t))-(kOnCaMKK*x13(t)*x7(t)-kOffCaMKK*x15(t))+(kPhosCaMKK*x15(t))-(kOnCaMKK*x13(t)*x8(t)-kOffCaMKK*x16(t))+(kPhosCaMKK*x16(t))-(kOnCaMKK*x13(t)*x9(t)-kOffCaMKK*x17(t))+(kPhosCaMKK*x17(t)),
        x14'(t) = (kOnCaMKK*x13(t)*x5(t)-kOffCaMKK*x14(t))-(kPhosCaMKK*x14(t)),
        x15'(t) = (kOnCaMKK*x13(t)*x7(t)-kOffCaMKK*x15(t))-(kPhosCaMKK*x15(t)),
        x16'(t) = (kOnCaMKK*x13(t)*x8(t)-kOffCaMKK*x16(t))-(kPhosCaMKK*x16(t)),
        x17'(t) = (kOnCaMKK*x13(t)*x9(t)-kOffCaMKK*x17(t))-(kPhosCaMKK*x17(t)),
        x18'(t) = -(kOnLKB1*x18(t)*x7(t)-kOffLKB1*x19(t))+(kPhosLKB1*x19(t))-(kOnLKB1*x18(t)*x8(t)- kOffLKB1*x20(t))+(kPhosLKB1*x20(t)),
        x19'(t) = (kOnLKB1*x18(t)*x7(t)-kOffLKB1*x19(t))-(kPhosLKB1*x19(t)),
        x20'(t) = (kOnLKB1*x18(t)*x8(t)- kOffLKB1*x20(t))-(kPhosLKB1*x20(t)),
        x21'(t) = -(kOnPP*x21(t)*x6(t)-kOffPP*x22(t))+(kDephosPP*x22(t))-(kOnPP*x21(t)*x12(t)-kOffPP*x23(t))+(kDephosPP*x23(t)),
        x22'(t) = (kOnPP*x21(t)*x6(t)-kOffPP*x22(t))-(kDephosPP*x22(t)),
        x23'(t) = (kOnPP*x21(t)*x12(t)-kOffPP*x23(t))-(kDephosPP*x23(t)),
        x24'(t) = -(kOnAMPK*x24(t)*x10(t)-kOffAMPK*x26(t))+(kDephosx27(t)*x28(t)),
        x25'(t) = (kPhosAMPK*x26(t))-(kOnPP1*x27(t)*x25(t)-kOffPP1*x28(t)),
        x26'(t) = (kOnAMPK*x24(t)*x10(t)-kOffAMPK*x26(t))-(kPhosAMPK*x26(t)),
        x27'(t) = -(kOnPP1*x27(t)*x25(t)-kOffPP1*x28(t))+(kDephosPP1*x28(t)),
        x28'(t) = (kOnPP1*x27(t)*x25(t)-kOffPP1*x28(t))-(kDephosPP1*x28(t)),
        y1(t) = (x28(t) + x25(t))/(x26(t) + x24(t))
)

local_id = assess_local_identifiability(MA_single, funcs_to_check = [kOnAMP,kOffAMP,kOnADP,kOffADP,kOnATP,kOffATP,kOnCaMKK,kOffCaMKK,kPhosCaMKK,kOnLKB1,kOffLKB1,kPhosLKB1,kOnPP,kOffPP,kDephosPP,kOnAMPK,kOffAMPK,kPhosAMPK,kOnPP1,kOffPP1,kDephosPP1])

# save to file for this model
# if the file exists, delete it
fname = "./local_ID_MA_single.txt"
if isfile(fname)
    rm(fname)
end
# create file to save results
file = open(fname, "w")
println(file, "Locally Identifiable parameters:")
for (key, value) in local_id
    if value == 1
        print(file, "$key, ")
    end
end
println(file, "")
println(file, "Nonidentifiable parameters:")
for (key, value) in local_id
    if value == 0
        print(file, "$key, ")
    end
end
close(file)
