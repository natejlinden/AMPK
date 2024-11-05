using StructuralIdentifiability

# MM_single = @ODEmodel(
#     x1'(t) = -(kOnAMP*x1(t)*x5(t) - kOffAMP*x7(t))-(kOnAMP*x1(t)*x6(t) - kOffAMP*x10(t))-((((((((VforAK*(kmd*kmd))/(KeqAK*kmt*kmm))*(x2(t)*x2(t)))/(kmd*kmd))*x3(t)*x1(t))/(kmt*kmm))-num_rev)/(1+(x3(t)/kmt)+(x1(t)/kmm)+((x3(t)*x1(t))/(kmt*kmm))+((2*x2(t))/kmd)+((x2(t)*x2(t))/(kmd*kmd)))),
#     x2'(t) = -(kOnADP*x2(t)*x5(t) - kOffADP*x8(t))-(kOnADP*x2(t)*x6(t) - kOffADP*x11(t))-(kGly*x2(t))+2*((((((((VforAK*(kmd*kmd))/(KeqAK*kmt*kmm))*(x2(t)*x2(t)))/(kmd*kmd))*x3(t)*x1(t))/(kmt*kmm))-num_rev)/(1+(x3(t)/kmt)+(x1(t)/kmm)+((x3(t)*x1(t))/(kmt*kmm))+((2*x2(t))/kmd)+((x2(t)*x2(t))/(kmd*kmd))))+(kHydro*x3(t))-((VmaxOxPhos*(((x2(t)*x2(t))/(Kadp*Kapd))))/(1+(((x2(t)*x2(t))/(Kadp*Kapd)))))+((((((VforCK*Kiq*Kp)/(KeqCK*Kia*Kb))*x3(t)*(TCr-x4(t)))/(Kiq*Kp))-((VforCK*x2(t)*x4(t))/(Kia*Kb)))/(1+(x2(t)/Kia)+(x4(t)/Kib)+(x3(t)/Kiq)+((x2(t)*x4(t))/(Kia*Kb))+(((TCr-x4(t))*x3(t))/(Kiq*Kp)))),
#     x3'(t) = -(kOnATP*x3(t)*x5(t) - kOffATP*x9(t))-(kOnATP*x3(t)*x6(t) - kOffATP*x12(t))+(kGly*x2(t))-((((((((VforAK*(kmd*kmd))/(KeqAK*kmt*kmm))*(x2(t)*x2(t)))/(kmd*kmd))*x3(t)*x1(t))/(kmt*kmm))-num_rev)/(1+(x3(t)/kmt)+(x1(t)/kmm)+((x3(t)*x1(t))/(kmt*kmm))+((2*x2(t))/kmd)+((x2(t)*x2(t))/(kmd*kmd))))-(kHydro*x3(t))+((VmaxOxPhos*(((x2(t)*x2(t))/(Kadp*Kapd))))/(1+(((x2(t)*x2(t))/(Kadp*Kapd)))))-((((((VforCK*Kiq*Kp)/(KeqCK*Kia*Kb))*x3(t)*(TCr-x4(t)))/(Kiq*Kp))-((VforCK*x2(t)*x4(t))/(Kia*Kb)))/(1+(x2(t)/Kia)+(x4(t)/Kib)+(x3(t)/Kiq)+((x2(t)*x4(t))/(Kia*Kb))+(((TCr-x4(t))*x3(t))/(Kiq*Kp)))),
#     x4'(t) = ((((((VforCK*Kiq*Kp)/(KeqCK*Kia*Kb))*x3(t)*(TCr-x4(t)))/(Kiq*Kp))-((VforCK*x2(t)*x4(t))/(Kia*Kb)))/(1+(x2(t)/Kia)+(x4(t)/Kib)+(x3(t)/Kiq)+((x2(t)*x4(t))/(Kia*Kb))+(((TCr-x4(t))*x3(t))/(Kiq*Kp)))),
#     x5'(t) = -(kOnAMP*x1(t)*x5(t) - kOffAMP*x7(t))-(kOnADP*x2(t)*x5(t) - kOffADP*x8(t))-(kOnATP*x3(t)*x5(t) - kOffATP*x9(t))-((kCaMKK*CaMKKtot*x5(t))/(KmCaMKK + x5(t)))+((kPP*PPtot*x6(t))/(KmPP + x6(t))),
#     x6'(t) = -(kOnAMP*x1(t)*x6(t) - kOffAMP*x10(t))-(kOnADP*x2(t)*x6(t) - kOffADP*x11(t))-(kOnATP*x3(t)*x6(t) - kOffATP*x12(t))+((kCaMKK*CaMKKtot*x5(t))/(KmCaMKK + x5(t)))-((kPP*PPtot*x6(t))/(KmPP + x6(t))),
#     x7'(t) = (kOnAMP*x1(t)*x5(t) - kOffAMP*x7(t))-((kCaMKK*CaMKKtot*x7(t))/(KmCaMKK + x7(t)))-((kLKB1*LKB1tot*x7(t))/(KmLKB1 + x7(t))),
#     x8'(t) = (kOnADP*x2(t)*x5(t) - kOffADP*x8(t))-((kCaMKK*CaMKKtot*x8(t))/(KmCaMKK + x8(t)))-((kLKB1*LKB1tot*x8(t))/(KmLKB1 + x8(t))),
#     x9'(t) = (kOnATP*x3(t)*x5(t) - kOffATP*x9(t))-((kCaMKK*CaMKKtot*x9(t))/(KmCaMKK + x9(t)))+((kPP*PPtot*x12(t))/(KmPP + x12(t))),
#     x10'(t) = (kOnAMP*x1(t)*x6(t) - kOffAMP*x10(t))+((kCaMKK*CaMKKtot*x7(t))/(KmCaMKK + x7(t)))+((kLKB1*LKB1tot*x7(t))/(KmLKB1 + x7(t))),
#     x11'(t) = (kOnADP*x2(t)*x6(t) - kOffADP*x11(t))+((kCaMKK*CaMKKtot*x8(t))/(KmCaMKK + x8(t)))+((kLKB1*LKB1tot*x8(t))/(KmLKB1 + x8(t))),
#     x12'(t) = (kOnATP*x3(t)*x6(t) - kOffATP*x12(t))+((kCaMKK*CaMKKtot*x9(t))/(KmCaMKK + x9(t)))-((kPP*PPtot*x12(t))/(KmPP + x12(t))),
#     x13'(t) = -((kAMPK*x10(t)*x13(t))/(KmAMPK + x13(t)))+((kPP1*PP1tot*x14(t))/(KmPP1 + x14(t))),
#     x14'(t) = ((kAMPK*x10(t)*x13(t))/(KmAMPK + x13(t)))-((kPP1*PP1tot*x14(t))/(KmPP1 + x14(t))), 
#     y1(t) = x14(t)/x13(t)
# )

MM_single = @ODEmodel(
    x1'(t) = -(kOnAMP*x1(t)*x5(t) - kOffAMP*x7(t))-(kOnAMP*x1(t)*x6(t) - kOffAMP*x10(t))-((((((((VforAK*(kmd*kmd))/(KeqAK*kmt*kmm))*(x2(t)*x2(t)))/(kmd*kmd))*x3(t)*x1(t))/(kmt*kmm))-num_rev)/(1+(x3(t)/kmt)+(x1(t)/kmm)+((x3(t)*x1(t))/(kmt*kmm))+((2*x2(t))/kmd)+((x2(t)*x2(t))/(kmd*kmd)))),
    x2'(t) = -(kOnADP*x2(t)*x5(t) - kOffADP*x8(t))-(kOnADP*x2(t)*x6(t) - kOffADP*x11(t))-(kGly*x2(t))+2*((((((((VforAK*(kmd*kmd))/(KeqAK*kmt*kmm))*(x2(t)*x2(t)))/(kmd*kmd))*x3(t)*x1(t))/(kmt*kmm))-num_rev)/(1+(x3(t)/kmt)+(x1(t)/kmm)+((x3(t)*x1(t))/(kmt*kmm))+((2*x2(t))/kmd)+((x2(t)*x2(t))/(kmd*kmd))))+(kHydro*x3(t))-((VmaxOxPhos*(((x2(t)*x2(t))/(Kadp*Kapd))))/(1+(((x2(t)*x2(t))/(Kadp*Kapd)))))+((((((VforCK*Kiq*Kp)/(KeqCK*Kia*Kb))*x3(t)*(TCr-x4(t)))/(Kiq*Kp))-((VforCK*x2(t)*x4(t))/(Kia*Kb)))/(1+(x2(t)/Kia)+(x4(t)/Kib)+(x3(t)/Kiq)+((x2(t)*x4(t))/(Kia*Kb))+(((TCr-x4(t))*x3(t))/(Kiq*Kp)))),
    x3'(t) = -(kOnATP*x3(t)*x5(t) - kOffATP*x9(t))-(kOnATP*x3(t)*x6(t) - kOffATP*x12(t))+(kGly*x2(t))-((((((((VforAK*(kmd*kmd))/(KeqAK*kmt*kmm))*(x2(t)*x2(t)))/(kmd*kmd))*x3(t)*x1(t))/(kmt*kmm))-num_rev)/(1+(x3(t)/kmt)+(x1(t)/kmm)+((x3(t)*x1(t))/(kmt*kmm))+((2*x2(t))/kmd)+((x2(t)*x2(t))/(kmd*kmd))))-(kHydro*x3(t))+((VmaxOxPhos*(((x2(t)*x2(t))/(Kadp*Kapd))))/(1+(((x2(t)*x2(t))/(Kadp*Kapd)))))-((((((VforCK*Kiq*Kp)/(KeqCK*Kia*Kb))*x3(t)*(TCr-x4(t)))/(Kiq*Kp))-((VforCK*x2(t)*x4(t))/(Kia*Kb)))/(1+(x2(t)/Kia)+(x4(t)/Kib)+(x3(t)/Kiq)+((x2(t)*x4(t))/(Kia*Kb))+(((TCr-x4(t))*x3(t))/(Kiq*Kp)))),
    x4'(t) = ((((((VforCK*Kiq*Kp)/(KeqCK*Kia*Kb))*x3(t)*(TCr-x4(t)))/(Kiq*Kp))-((VforCK*x2(t)*x4(t))/(Kia*Kb)))/(1+(x2(t)/Kia)+(x4(t)/Kib)+(x3(t)/Kiq)+((x2(t)*x4(t))/(Kia*Kb))+(((TCr-x4(t))*x3(t))/(Kiq*Kp)))),
    x5'(t) = -(kOnAMP*x1(t)*x5(t) - kOffAMP*x7(t))-(kOnADP*x2(t)*x5(t) - kOffADP*x8(t))-(kOnATP*x3(t)*x5(t) - kOffATP*x9(t))-((kCaMKK*x5(t))/(KmCaMKK + x5(t)))+((kPP*x6(t))/(KmPP + x6(t))),
    x6'(t) = -(kOnAMP*x1(t)*x6(t) - kOffAMP*x10(t))-(kOnADP*x2(t)*x6(t) - kOffADP*x11(t))-(kOnATP*x3(t)*x6(t) - kOffATP*x12(t))+((kCaMKK*x5(t))/(KmCaMKK + x5(t)))-((kPP*x6(t))/(KmPP + x6(t))),
    x7'(t) = (kOnAMP*x1(t)*x5(t) - kOffAMP*x7(t))-((kCaMKK*x7(t))/(KmCaMKK + x7(t)))-((kLKB1*x7(t))/(KmLKB1 + x7(t))),
    x8'(t) = (kOnADP*x2(t)*x5(t) - kOffADP*x8(t))-((kCaMKK*x8(t))/(KmCaMKK + x8(t)))-((kLKB1*x8(t))/(KmLKB1 + x8(t))),
    x9'(t) = (kOnATP*x3(t)*x5(t) - kOffATP*x9(t))-((kCaMKK*x9(t))/(KmCaMKK + x9(t)))+((kPP*x12(t))/(KmPP + x12(t))),
    x10'(t) = (kOnAMP*x1(t)*x6(t) - kOffAMP*x10(t))+((kCaMKK*x7(t))/(KmCaMKK + x7(t)))+((kLKB1*x7(t))/(KmLKB1 + x7(t))),
    x11'(t) = (kOnADP*x2(t)*x6(t) - kOffADP*x11(t))+((kCaMKK*x8(t))/(KmCaMKK + x8(t)))+((kLKB1*x8(t))/(KmLKB1 + x8(t))),
    x12'(t) = (kOnATP*x3(t)*x6(t) - kOffATP*x12(t))+((kCaMKK*x9(t))/(KmCaMKK + x9(t)))-((kPP*x12(t))/(KmPP + x12(t))),
    x13'(t) = -((kAMPK*x10(t)*x13(t))/(KmAMPK + x13(t)))+((kPP1*x14(t))/(KmPP1 + x14(t))),
    x14'(t) = ((kAMPK*x10(t)*x13(t))/(KmAMPK + x13(t)))-((kPP1*x14(t))/(KmPP1 + x14(t))), 
    y1(t) = x14(t)/x13(t)
)

local_id = assess_local_identifiability(MM_single, funcs_to_check = [kOnAMP,kOffAMP,kOnADP,kOffADP,kOnATP,kOffATP,kCaMKK,KmCaMKK,kLKB1,KmLKB1,kPP,KmPP,kAMPK,KmAMPK,kPP1,KmPP1])

# save to file for this model
# if the file exists, delete it
fname = "./local_ID_MM_single.txt"
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



