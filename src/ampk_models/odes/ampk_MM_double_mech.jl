ode = @ODEmodel(
    x0'(t) =  -(kOnAMP*x0(t)*x3(t)-kOffAMP*x5(t))-(kOnAMP*x0(t)*x4(t)-kOffAMP*x8(t))-(kOnAMP*x0(t)*x5(t)-kOffAMP*x11(t))-(kOnAMP*x0(t)*x6(t)-kOffAMP*x12(t))-(kOnAMP*x0(t)*x7(t)-kOffAMP*x13(t))-(kOnAMP*x0(t)*x8(t)-kOffAMP*x17(t))-(kOnAMP*x0(t)*x9(t)-kOffAMP*x18(t))-(kOnAMP*x0(t)*x10(t)-kOffAMP*x19(t))-Jak,
    x1'(t) =  -(kOnADP*x1(t)*x3(t)-kOffADP*x6(t))-(kOnADP*x1(t)*x4(t)-kOffADP*x9(t))-(kOnADP*x1(t)*x5(t)-kOffADP*x12(t))-(kOnADP*x1(t)*x6(t)-kOffADP*x14(t))-(kOnADP*x1(t)*x7(t)-kOffADP*x15(t))-(kOnADP*x1(t)*x8(t)-kOffADP*x18(t))-(kOnADP*x1(t)*x9(t)-kOffADP*x20(t))-(kOnADP*x1(t)*x10(t)-kOffADP*x21(t))-Jgly+2*Jak+Jhydro-Joxphos,
    x2'(t) =  -(kOnATP*x2(t)*x3(t)-kOffATP*x7(t))-(kOnATP*x2(t)*x4(t)-kOffATP*x10(t))-(kOnATP*x2(t)*x5(t)-kOffATP*x13(t))-(kOnATP*x2(t)*x6(t)-kOffATP*x15(t))-(kOnATP*x2(t)*x7(t)-kOffATP*x16(t))-(kOnATP*x2(t)*x8(t)-kOffATP*x19(t))-(kOnATP*x2(t)*x9(t)-kOffATP*x21(t))-(kOnATP*x2(t)*x10(t)-kOffATP*x22(t))+Jgly-Jak-Jhydro+Joxphos,
    # free AMPK
    x3'(t) =  -(kOnAMP*x0(t)*x3(t)-kOffAMP*x5(t))-(kOnADP*x1(t)*x3(t)-kOffADP*x6(t))-(kOnATP*x2(t)*x3(t)-kOffATP*x7(t))-((kCaMKK*CaMKKtot*x3(t))/(KmCaMKK+x3(t)))+((kPP*PPtot*x4(t))/(KmPP+x4(t))),
    x4'(t) =  -(kOnAMP*x0(t)*x4(t)-kOffAMP*x8(t))-(kOnADP*x1(t)*x4(t)-kOffADP*x9(t))-(kOnATP*x2(t)*x4(t)-kOffATP*x10(t))+((kCaMKK*CaMKKtot*x3(t))/(KmCaMKK+x3(t)))-((kPP*PPtot*x4(t))/(KmPP+x4(t))),
    # single AXP-AMPK complexes
    x5'(t) =  (kOnAMP*x0(t)*x3(t)-kOffAMP*x5(t))-(kOnAMP*x0(t)*x5(t)-kOffAMP*x11(t))-(kOnADP*x1(t)*x5(t)-kOffADP*x12(t))-(kOnATP*x2(t)*x5(t)-kOffATP*x13(t))-((kCaMKK*CaMKKtot*x5(t))/(KmCaMKK+x5(t)))-((kLKB1*LKB1tot*x5(t))/(KmLKB1+x5(t))),
    x6'(t) =  (kOnADP*x1(t)*x3(t)-kOffADP*x6(t))-(kOnAMP*x0(t)*x6(t)-kOffAMP*x12(t))-(kOnADP*x1(t)*x6(t)-kOffADP*x14(t))-(kOnATP*x2(t)*x6(t)-kOffATP*x15(t))-((kCaMKK*CaMKKtot*x6(t))/(KmCaMKK+x6(t)))-((kLKB1*LKB1tot*x6(t))/(KmLKB1+x6(t))),
    x7'(t) =  (kOnATP*x2(t)*x3(t)-kOffATP*x7(t))-(kOnAMP*x0(t)*x7(t)-kOffAMP*x13(t))-(kOnADP*x1(t)*x7(t)-kOffADP*x15(t))-(kOnATP*x2(t)*x7(t)-kOffATP*x16(t))-((kCaMKK*CaMKKtot*x7(t))/(KmCaMKK+x7(t)))+((kPP*PPtot*x10(t))/(KmPP+x10(t))),
    # single AXP-pAMPK complexes
    x8'(t) =  (kOnAMP*x0(t)*x4(t)-kOffAMP*x8(t))-(kOnAMP*x0(t)*x8(t)-kOffAMP*x17(t))-(kOnADP*x1(t)*x8(t)-kOffADP*x18(t))-(kOnATP*x2(t)*x8(t)-kOffATP*x19(t))+((kCaMKK*CaMKKtot*x5(t))/(KmCaMKK+x5(t)))+((kLKB1*LKB1tot*x5(t))/(KmLKB1+x5(t))),
    x9'(t) =  (kOnADP*x1(t)*x4(t)-kOffADP*x9(t))-(kOnAMP*x0(t)*x9(t)-kOffAMP*x18(t))-(kOnADP*x1(t)*x9(t)-kOffADP*x20(t))-(kOnATP*x2(t)*x9(t)-kOffATP*x21(t))+((kCaMKK*CaMKKtot*x6(t))/(KmCaMKK+x6(t)))+((kLKB1*LKB1tot*x6(t))/(KmLKB1+x6(t))),
    x10'(t) =  (kOnATP*x2(t)*x4(t)-kOffATP*x10(t))-(kOnAMP*x0(t)*x10(t)-kOffAMP*x19(t))-(kOnADP*x1(t)*x10(t)-kOffADP*x21(t))-(kOnATP*x2(t)*x10(t)-kOffATP*x22(t))+((kCaMKK*CaMKKtot*x7(t))/(KmCaMKK+x7(t)))-((kPP*PPtot*x10(t))/(KmPP+x10(t))),
    # double AXP-AMPK complexes
    x11'(t) =  (kOnAMP*x0(t)*x5(t)-kOffAMP*x11(t))-((kCaMKK*CaMKKtot*x11(t))/(KmCaMKK+x11(t)))-((kLKB1*LKB1tot*x11(t))/(KmLKB1+x11(t))),
    x12'(t) =  (kOnAMP*x0(t)*x6(t)-kOffAMP*x12(t))+(kOnADP*x1(t)*x5(t)-kOffADP*x12(t))-((kCaMKK*CaMKKtot*x12(t))/(KmCaMKK+x12(t)))-((kLKB1*LKB1tot*x12(t))/(KmLKB1+x12(t))),
    x13'(t) =  (kOnAMP*x0(t)*x7(t)-kOffAMP*x13(t))+(kOnATP*x2(t)*x5(t)-kOffATP*x13(t))-((kCaMKK*CaMKKtot*x13(t))/(KmCaMKK+x13(t)))+((kPP*PPtot*x19(t))/(KmPP+x19(t))),
    x14'(t) =  (kOnADP*x1(t)*x6(t)-kOffADP*x14(t))-((kCaMKK*CaMKKtot*x14(t))/(KmCaMKK+x14(t)))-((kLKB1*LKB1tot*x14(t))/(KmLKB1+x14(t))),
    x15'(t) =  (kOnADP*x1(t)*x7(t)-kOffADP*x15(t))+(kOnATP*x2(t)*x6(t)-kOffATP*x15(t))-((kCaMKK*CaMKKtot*x15(t))/(KmCaMKK+x15(t)))+((kPP*PPtot*x21(t))/(KmPP+x21(t))),
    x16'(t) =  (kOnATP*x2(t)*x7(t)-kOffATP*x16(t))-((kCaMKK*CaMKKtot*x16(t))/(KmCaMKK+x16(t)))+((kPP*PPtot*x22(t))/(KmPP+x22(t))),
    # double AXP-pAMPK complexes
    x17'(t) =  (kOnAMP*x0(t)*x8(t)-kOffAMP*x17(t))+((kCaMKK*CaMKKtot*x11(t))/(KmCaMKK+x11(t)))+((kLKB1*LKB1tot*x11(t))/(KmLKB1+x11(t))), 
    x18'(t) =  (kOnAMP*x0(t)*x9(t)-kOffAMP*x18(t))+(kOnADP*x1(t)*x8(t)-kOffADP*x18(t))+((kCaMKK*CaMKKtot*x12(t))/(KmCaMKK+x12(t)))+((kLKB1*LKB1tot*x12(t))/(KmLKB1+x12(t))), 
    x19'(t) =  (kOnAMP*x0(t)*x10(t)-kOffAMP*x19(t))+(kOnATP*x2(t)*x8(t)-kOffATP*x19(t))+((kCaMKK*CaMKKtot*x13(t))/(KmCaMKK+x13(t)))-((kPP*PPtot*x19(t))/(KmPP+x19(t))),
    x20'(t) =  (kOnADP*x1(t)*x9(t)-kOffADP*x20(t))+((kCaMKK*CaMKKtot*x14(t))/(KmCaMKK+x14(t)))+((kLKB1*LKB1tot*x14(t))/(KmLKB1+x14(t))),
    x21'(t) =  (kOnADP*x1(t)*x10(t)-kOffADP*x21(t))+(kOnATP*x2(t)*x9(t)-kOffATP*x21(t))+((kCaMKK*CaMKKtot*x15(t))/(KmCaMKK+x15(t)))-((kPP*PPtot*x21(t))/(KmPP+x21(t))),
    x22'(t) =  (kOnATP*x2(t)*x10(t)-kOffATP*x22(t))+((kCaMKK*CaMKKtot*x16(t))/(KmCaMKK+x16(t)))-((kPP*PPtot*x22(t))/(KmPP+x22(t))),
    # AMPKAR
    x23'(t) =  -((kAMPK*x8(t)*x23(t))/(KmAMPK+x23(t)))-((kAMPK*x17(t)*x23(t))/(KmAMPK+x23(t)))-((kAMPK*x18(t)*x23(t))/(KmAMPK+x23(t)))+((kPP1*PP1tot*x24(t))/(KmPP1+x24(t))),
    x24'(t) =  ((kAMPK*x8(t)*x23(t))/(KmAMPK+x23(t)))+((kAMPK*x17(t)*x23(t))/(KmAMPK+x23(t)))+((kAMPK*x18(t)*x23(t))/(KmAMPK+x23(t)))-((kPP1*PP1tot*x24(t))/(KmPP1+x24(t))),
    y1(t) = x24(t) / (x23(t) + x24(t))
)