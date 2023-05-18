ode = @ODEmodel(
    x1'(t) =  -(kOnAMP*x1(t)*x4(t)-kOffAMP*x6(t))-(kOnAMP*x1(t)*x5(t)-kOffAMP*x9(t))-((kForAK*x3(t)*x1(t))-(kRevAK*x2(t)*x2(t))),
    x2'(t) =  -(kOnADP*x2(t)*x4(t)-kOffADP*x7(t))-(kOnADP*x2(t)*x5(t)-kOffADP*x10(t))-(2*kGly*x2(t)*x2(t))+2*((kForAK*x3(t)*x1(t))-(kRevAK*x2(t)*x2(t)))+(kHydro*x3(t)), # -((VmaxOxPhos*((x2(t)/Kadp)**n))/(1+((x2(t)/Kadp)**n))),
    x3'(t) =  -(kOnATP*x3(t)*x4(t)-kOffATP*x8(t))-(kOnATP*x3(t)*x5(t)-kOffATP*x11(t))+(2*kGly*x2(t)*x2(t))-((kForAK*x3(t)*x1(t))-(kRevAK*x2(t)*x2(t)))-(kHydro*x3(t)), #, +((VmaxOxPhos*((x2(t)/Kadp)**n))/(1+((x2(t)/Kadp)**n))),
    # free AMPK
    x4'(t) =  -(kOnAMP*x1(t)*x4(t)-kOffAMP*x6(t))-(kOnADP*x2(t)*x4(t)-kOffADP*x7(t))-(kOnATP*x3(t)*x4(t)-kOffATP*x8(t))-((kCaMKK*CaMKKtot*x4(t))/(KmCaMKK+x4(t)))+((kPP*PPtot*x5(t))/(KmPP+x5(t))),
    x5'(t) =  -(kOnAMP*x1(t)*x5(t)-kOffAMP*x9(t))-(kOnADP*x2(t)*x5(t)-kOffADP*x10(t))-(kOnATP*x3(t)*x5(t)-kOffATP*x11(t))+((kCaMKK*CaMKKtot*x4(t))/(KmCaMKK+x4(t)))-((kPP*PPtot*x5(t))/(KmPP+x5(t))),
    # single AXP-AMPK complexes
    x6'(t) =  (kOnAMP*x1(t)*x4(t)-kOffAMP*x6(t))-((kCaMKK*CaMKKtot*x6(t))/(KmCaMKK+x6(t)))-((kLKB1*LKB1tot*x6(t))/(KmLKB1+x6(t))),
    x7'(t) =  (kOnADP*x2(t)*x4(t)-kOffADP*x7(t))-((kCaMKK*CaMKKtot*x7(t))/(KmCaMKK+x7(t)))-((kLKB1*LKB1tot*x7(t))/(KmLKB1+x7(t))),
    x8'(t) =  (kOnATP*x3(t)*x4(t)-kOffATP*x8(t))-((kCaMKK*CaMKKtot*x8(t))/(KmCaMKK+x8(t)))+((kPP*PPtot*x11(t))/(KmPP+x11(t))),
    # single AXP-pAMPK complexes
    x9'(t) =  (kOnAMP*x1(t)*x5(t)-kOffAMP*x9(t))+((kCaMKK*CaMKKtot*x6(t))/(KmCaMKK+x6(t)))+((kLKB1*LKB1tot*x6(t))/(KmLKB1+x6(t))),
    x10'(t) =  (kOnADP*x2(t)*x5(t)-kOffADP*x10(t))+((kCaMKK*CaMKKtot*x7(t))/(KmCaMKK+x7(t)))+((kLKB1*LKB1tot*x7(t))/(KmLKB1+x7(t))),
    x11'(t) =  (kOnATP*x3(t)*x5(t)-kOffATP*x11(t))+((kCaMKK*CaMKKtot*x8(t))/(KmCaMKK+x8(t)))-((kPP*PPtot*x11(t))/(KmPP+x11(t))),
    # AMPKAR
    x12'(t) =  -((kAMPK*x9(t)*x12(t))/(KmAMPK+x12(t)))+((kPP1*PP1tot*x13(t))/(KmPP1+x13(t))),
    x13'(t) =  ((kAMPK*x9(t)*x12(t))/(KmAMPK+x12(t)))-((kPP1*PP1tot*x13(t))/(KmPP1+x13(t))),
    y1(t) = x13(t) / (x12(t) + x13(t))
)
