ode = @ODEmodel(
    x1'(t) =  -(kOnAMP*x1(t)*x4(t)-kOffAMP*x6(t))-(kOnAMP*x1(t)*x5(t)-kOffAMP*x9(t))-(((((VforAK*x3(t)*x1(t))/(kmt*kmm))-((((VforAK*(kmd^2))/(KeqAK*kmt*kmm))*(x2(t)^2))/(kmd^2)))/(1+(x3(t)/kmt)+(x1(t)/kmm)+((x3(t)*x1(t))/(kmt*kmm))+((2*x2(t))/kmd)+((x2(t)^2)/(kmd^2))))),
    x2'(t) =  -(kOnADP*x2(t)*x4(t)-kOffADP*x7(t))-(kOnADP*x2(t)*x5(t)-kOffADP*x10(t))-(2*kGly*x2(t)*x2(t))+2*(((((VforAK*x3(t)*x1(t))/(kmt*kmm))-((((VforAK*(kmd^2))/(KeqAK*kmt*kmm))*(x2(t)^2))/(kmd^2)))/(1+(x3(t)/kmt)+(x1(t)/kmm)+((x3(t)*x1(t))/(kmt*kmm))+((2*x2(t))/kmd)+((x2(t)^2)/(kmd^2)))))+(kHydro*x3(t)), #-((VmaxOxPhos*((x2(t)/Kadp)**n))/(1+((x2(t)/Kadp)**n))),
    x3'(t) =  -(kOnATP*x3(t)*x4(t)-kOffATP*x8(t))-(kOnATP*x3(t)*x5(t)-kOffATP*x11(t))+(2*kGly*x2(t)*x2(t))-(((((VforAK*x3(t)*x1(t))/(kmt*kmm))-((((VforAK*(kmd^2))/(KeqAK*kmt*kmm))*(x2(t)^2))/(kmd^2)))/(1+(x3(t)/kmt)+(x1(t)/kmm)+((x3(t)*x1(t))/(kmt*kmm))+((2*x2(t))/kmd)+((x2(t)^2)/(kmd^2)))))-(kHydro*x3(t)), #+((VmaxOxPhos*((x2(t)/Kadp)**n))/(1+((x2(t)/Kadp)**n))),
    x4'(t) =  -(kOnAMP*x1(t)*x4(t)-kOffAMP*x6(t))-(kOnADP*x2(t)*x4(t)-kOffADP*x7(t))-(kOnATP*x3(t)*x4(t)-kOffATP*x8(t))-((VmaxCaMKK*x4(t))/(KmCaMKK+x4(t)))+((VmaxPP*x5(t))/(KmPP+x5(t))),
    x5'(t) =  -(kOnAMP*x1(t)*x5(t)-kOffAMP*x9(t))-(kOnADP*x2(t)*x5(t)-kOffADP*x10(t))-(kOnATP*x3(t)*x5(t)-kOffATP*x11(t))+((VmaxCaMKK*x4(t))/(KmCaMKK+x4(t)))-((VmaxPP*x5(t))/(KmPP+x5(t))),
    x6'(t) =  (kOnAMP*x1(t)*x4(t)-kOffAMP*x6(t))-((VmaxCaMKK*x6(t))/(KmCaMKK+x6(t)))-((VmaxLKB1*x6(t))/(KmLKB1+x6(t))),
    x7'(t) =  (kOnADP*x2(t)*x4(t)-kOffADP*x7(t))-((VmaxCaMKK*x7(t))/(KmCaMKK+x7(t)))-((VmaxLKB1*x7(t))/(KmLKB1+x7(t))),
    x8'(t) =  (kOnATP*x3(t)*x4(t)-kOffATP*x8(t))-((VmaxCaMKK*x8(t))/(KmCaMKK+x8(t)))+((VmaxPP*x11(t))/(KmPP+x11(t))),
    x9'(t) =  (kOnAMP*x1(t)*x5(t)-kOffAMP*x9(t))+((VmaxCaMKK*x6(t))/(KmCaMKK+x6(t)))+((VmaxLKB1*x6(t))/(KmLKB1+x6(t))),
    x10'(t) =  (kOnADP*x2(t)*x5(t)-kOffADP*x10(t))+((VmaxCaMKK*x7(t))/(KmCaMKK+x7(t)))+((VmaxLKB1*x7(t))/(KmLKB1+x7(t))),
    x11'(t) =  (kOnATP*x3(t)*x5(t)-kOffATP*x11(t))+((VmaxCaMKK*x8(t))/(KmCaMKK+x8(t)))-((VmaxPP*x11(t))/(KmPP+x11(t))),
    x12'(t) =  -((kAMPK*x9(t)*x12(t))/(KmAMPK+x12(t)))+((VmaxPP1*x13(t))/(KmPP1+x13(t))),
    x13'(t) =  ((kAMPK*x9(t)*x12(t))/(KmAMPK+x12(t)))-((VmaxPP1*x13(t))/(KmPP1+x13(t))),
    y1(t) = x13(t) / (x12(t) + x13(t))
)
