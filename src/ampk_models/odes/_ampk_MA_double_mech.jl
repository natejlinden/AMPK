ode = @ODEmodel(
        x0'(t) = -kOnAMP*x0(t)*x3(t)-kOffAMP*x5(t)-kOnAMP*x0(t)*x4(t)-kOffAMP*x8(t)-kOnATP*x2(t)*x4(t)-kOffATP*x10(t)-kOnATP*x2(t)*x4(t)-kOffATP*x10(t)-kOnAMP*x0(t)*x7(t)-kOffAMP*x13(t)-kOnAMP*x0(t)*x8(t)-kOffAMP*x17(t)-kOnAMP*x0(t)*x9(t)-kOffAMP*x18(t)-kOnAMP*x0(t)*x10(t)-kOffAMP*x19(t)-kForAK*x2(t)*x0(t)-kRevAK*x1(t)*x1(t),
        x1'(t) = -kOnADP*x1(t)*x3(t)-kOffADP*x6(t)-kOnAMP*x0(t)*x4(t)-kOffAMP*x8(t)-kOnADP*x1(t)*x5(t)-kOffADP*x12(t)-kOnADP*x1(t)*x6(t)-kOffADP*x14(t)-kOnADP*x1(t)*x7(t)-kOffADP*x15(t)-kOnADP*x1(t)*x8(t) - kOffADP*x18(t)-kOnADP*x1(t)*x9(t)-kOffADP*x20(t)-kOnADP*x1(t)*x10(t)-kOffADP*x21(t)-2*kGly*x1(t)*x1(t)+2*kForAK*x2(t)*x0(t)-kRevAK*x1(t)*x1(t)+kHydro*x2(t),
        x2'(t) = -kOnATP*x2(t)*x3(t)-kOffATP*x7(t)-kOnATP*x2(t)*x4(t)-kOffATP*x10(t)-kOnATP*x2(t)*x5(t)-kOffATP*x13(t)-kOnATP*x2(t)*x6(t)-kOffATP*x15(t)-kOnATP*x2(t)*x7(t)-kOffATP*x16(t)-kOnATP*x2(t)*x8(t)-kOffATP*x19(t)-kOnATP*x2(t)*x9(t)-kOffATP*x21(t)-kOnATP*x2(t)*x10(t)-kOffATP*x22(t)+2*kGly*x1(t)*x1(t)-kForAK*x2(t)*x0(t)-kRevAK*x1(t)*x1(t)-kHydro*x2(t),
        # free AMPK
        x3'(t) = -kOnAMP*x0(t)*x3(t)-kOffAMP*x5(t)-kOnADP*x1(t)*x3(t)-kOffADP*x6(t)-kOnATP*x2(t)*x3(t)-kOffATP*x7(t)-kOnCaMKK*x49(t)*x3(t)-kOffCaMKK*x23(t)+kDephosPP*x38(t),
        x4'(t) = -kOnAMP*x0(t)*x4(t)-kOffAMP*x8(t)-kOnAMP*x0(t)*x4(t)-kOffAMP*x8(t)-kOnATP*x2(t)*x4(t)-kOffATP*x10(t)+kPhosCaMKK*x23(t)-kOnPP*x51(t)*x4(t)-kOffPP*x38(t),
        # single AXP-AMPK complexes
        x5'(t) = kOnAMP*x0(t)*x3(t)-kOffAMP*x5(t)-kOnATP*x2(t)*x4(t)-kOffATP*x10(t)-kOnADP*x1(t)*x5(t)-kOffADP*x12(t)-kOnATP*x2(t)*x5(t)-kOffATP*x13(t)-J27-kOnLKB1*x50(t)*x5(t)-kOffLKB1*x33(t),
        x6'(t) = kOnADP*x1(t)*x3(t)-kOffADP*x6(t)-kOnATP*x2(t)*x4(t)-kOffATP*x10(t)-kOnADP*x1(t)*x6(t)-kOffADP*x14(t)-kOnATP*x2(t)*x6(t)-kOffATP*x15(t)-kOnCaMKK*x49(t)*x6(t)-kOffCaMKK*x25(t)-kOnLKB1*x50(t)*x6(t)-kOffLKB1*x34(t),
        x7'(t) = kOnATP*x2(t)*x3(t)-kOffATP*x7(t)-kOnAMP*x0(t)*x7(t)-kOffAMP*x13(t)-kOnADP*x1(t)*x7(t)-kOffADP*x15(t)-kOnATP*x2(t)*x7(t)-kOffATP*x16(t)-kOnCaMKK*x49(t)*x7(t)-kOffCaMKK*x26(t)+kDephosPP*x39(t),
        # single AXP-pAMPK complexes
        x8'(t) = kOnAMP*x0(t)*x4(t)-kOffAMP*x8(t)-kOnAMP*x0(t)*x8(t)-kOffAMP*x17(t)-kOnADP*x1(t)*x8(t) - kOffADP*x18(t)-kOnATP*x2(t)*x8(t)-kOffATP*x19(t)+kPhosCaMKK*x24(t)+kPhosLKB1*x33(t)-kOnAMPK*x43(t)*x8(t)-kOffAMPK*x45(t)+kPhosAMPK*x45(t),
        x9'(t) = kOnAMP*x0(t)*x4(t)-kOffAMP*x8(t)-kOnAMP*x0(t)*x9(t)-kOffAMP*x18(t)-kOnADP*x1(t)*x9(t)-kOffADP*x20(t)-kOnATP*x2(t)*x9(t)-kOffATP*x21(t)+kPhosCaMKK*x25(t)+kPhosLKB1*x34(t),
        x10'(t) = kOnATP*x2(t)*x4(t)-kOffATP*x10(t)-kOnAMP*x0(t)*x10(t)-kOffAMP*x19(t)-kOnADP*x1(t)*x10(t)-kOffADP*x21(t)-kOnATP*x2(t)*x10(t)-kOffATP*x22(t)+kPhosCaMKK*x26(t)-kOnPP*x51(t)*x10(t)-kOffPP*x39(t),
        # double AXP-AMPK complexes
        x11'(t) = kOnATP*x2(t)*x4(t)-kOffATP*x10(t)-kOnCaMKK*x49(t)*x11(t)-kOffCaMKK*x27(t)-kOnLKB1*x50(t)*x11(t)-kOffLKB1*x35(t),
        x12'(t) = kOnATP*x2(t)*x4(t)-kOffATP*x10(t)+kOnADP*x1(t)*x5(t)-kOffADP*x12(t)-kOnCaMKK*x49(t)*x12(t)-kOffCaMKK*x28(t)-kOnLKB1*x50(t)*x12(t)-kOffLKB1*x36(t),
        x13'(t) = kOnAMP*x0(t)*x7(t)-kOffAMP*x13(t)+kOnATP*x2(t)*x5(t)-kOffATP*x13(t)-kOnCaMKK*x49(t)*x13(t)-kOffCaMKK*x29(t)+kDephosPP*x40(t),
        x14'(t) = kOnADP*x1(t)*x6(t)-kOffADP*x14(t)-kOnCaMKK*x49(t)*x14(t)-kOffCaMKK*x30(t)-kOnLKB1*x50(t)*x14(t)-kOffLKB1*x37(t),
        x15'(t) = kOnADP*x1(t)*x7(t)-kOffADP*x15(t)+kOnATP*x2(t)*x6(t)-kOffATP*x15(t)-kOnCaMKK*x49(t)*x15(t)-kOffCaMKK*x31(t)+kDephosPP*x41(t),
        x16'(t) = kOnATP*x2(t)*x7(t)-kOffATP*x16(t)-kOnCaMKK*x49(t)*x16(t)-kOffCaMKK*x32(t)+kDephosPP*x42(t),
        # double AXP-pAMPK complexes
        x17'(t) = kOnAMP*x0(t)*x8(t)-kOffAMP*x17(t)+kPhosCaMKK*x27(t)+kPhosLKB1*x35(t)-kOnAMPK*x43(t)*x17(t)-kOffAMPK*x46(t)+kPhosAMPK*x46(t),
        x18'(t) = kOnAMP*x0(t)*x9(t)-kOffAMP*x18(t)+kOnADP*x1(t)*x8(t) - kOffADP*x18(t)+kPhosCaMKK*x28(t)+kPhosLKB1*x36(t)-kOnAMPK*x43(t)*x18(t)-kOffAMPK*x47(t)+kPhosAMPK*x47(t),
        x19'(t) = kOnAMP*x0(t)*x10(t)-kOffAMP*x19(t)+kOnATP*x2(t)*x8(t)-kOffATP*x19(t)+kPhosCaMKK*x29(t)-kOnPP*x51(t)*x19(t)-kOffPP*x40(t),
        x20'(t) = kOnADP*x1(t)*x9(t)-kOffADP*x20(t)+kPhosCaMKK*x30(t)+kPhosLKB1*x37(t),
        x21'(t) = kOnADP*x1(t)*x10(t)-kOffADP*x21(t)+kOnATP*x2(t)*x9(t)-kOffATP*x21(t)+kPhosCaMKK*x31(t)-kOnPP*x51(t)*x21(t)-kOffPP*x41(t),
        x22'(t) = kOnATP*x2(t)*x10(t)-kOffATP*x22(t)+kPhosCaMKK*x32(t)-kOnPP*x51(t)*x22(t)-kOffPP*x42(t),
        # CaMKK complexes
        x49'(t) = -kOnCaMKK*x49(t)*x3(t)-kOffCaMKK*x23(t)+kPhosCaMKK*x23(t)-J27+kPhosCaMKK*x24(t)-kOnCaMKK*x49(t)*x6(t)-kOffCaMKK*x25(t)+kPhosCaMKK*x25(t)-kOnCaMKK*x49(t)*x7(t)-kOffCaMKK*x26(t)+kPhosCaMKK*x26(t)-kOnCaMKK*x49(t)*x11(t)-kOffCaMKK*x27(t)+kPhosCaMKK*x27(t)-kOnCaMKK*x49(t)*x12(t)-kOffCaMKK*x28(t)+kPhosCaMKK*x28(t)-kOnCaMKK*x49(t)*x13(t)-kOffCaMKK*x29(t)+kPhosCaMKK*x29(t)-kOnCaMKK*x49(t)*x14(t)-kOffCaMKK*x30(t)+kPhosCaMKK*x30(t)-kOnCaMKK*x49(t)*x15(t)-kOffCaMKK*x31(t)+kPhosCaMKK*x31(t)-kOnCaMKK*x49(t)*x16(t)-kOffCaMKK*x32(t)+kPhosCaMKK*x32(t),
        x23'(t) = kOnCaMKK*x49(t)*x3(t)-kOffCaMKK*x23(t)-kPhosCaMKK*x23(t),
        x24'(t) = J27-kPhosCaMKK*x24(t),
        x25'(t) = kOnCaMKK*x49(t)*x6(t)-kOffCaMKK*x25(t)-kPhosCaMKK*x25(t),
        x26'(t) = kOnCaMKK*x49(t)*x7(t)-kOffCaMKK*x26(t)-kPhosCaMKK*x26(t),
        x27'(t) = kOnCaMKK*x49(t)*x11(t)-kOffCaMKK*x27(t)-kPhosCaMKK*x27(t),
        x28'(t) = kOnCaMKK*x49(t)*x12(t)-kOffCaMKK*x28(t)-kPhosCaMKK*x28(t),
        x29'(t) = kOnCaMKK*x49(t)*x13(t)-kOffCaMKK*x29(t)-kPhosCaMKK*x29(t),
        x30'(t) = kOnCaMKK*x49(t)*x14(t)-kOffCaMKK*x30(t)-kPhosCaMKK*x30(t),
        x31'(t) = kOnCaMKK*x49(t)*x15(t)-kOffCaMKK*x31(t)-kPhosCaMKK*x31(t),
        x32'(t) = kOnCaMKK*x49(t)*x16(t)-kOffCaMKK*x32(t)-kPhosCaMKK*x32(t),
        # LKB1 complexes
        x50'(t) = -kOnLKB1*x50(t)*x5(t)-kOffLKB1*x33(t)+kPhosLKB1*x33(t)-kOnLKB1*x50(t)*x6(t)-kOffLKB1*x34(t)+kPhosLKB1*x34(t)-kOnLKB1*x50(t)*x11(t)-kOffLKB1*x35(t)+kPhosLKB1*x35(t)-kOnLKB1*x50(t)*x12(t)-kOffLKB1*x36(t)+kPhosLKB1*x36(t)-kOnLKB1*x50(t)*x14(t)-kOffLKB1*x37(t)+kPhosLKB1*x37(t),
        x33'(t) = kOnLKB1*x50(t)*x5(t)-kOffLKB1*x33(t)-kPhosLKB1*x33(t),
        x34'(t) = kOnLKB1*x50(t)*x6(t)-kOffLKB1*x34(t)-kPhosLKB1*x34(t),
        x35'(t) = kOnLKB1*x50(t)*x11(t)-kOffLKB1*x35(t)-kPhosLKB1*x35(t),
        x36'(t) = kOnLKB1*x50(t)*x12(t)-kOffLKB1*x36(t)-kPhosLKB1*x36(t),
        x37'(t) = kOnLKB1*x50(t)*x14(t)-kOffLKB1*x37(t)-kPhosLKB1*x37(t),
        # AMPK phosphatase complexes
        x51'(t) = -kOnPP*x51(t)*x4(t)-kOffPP*x38(t)+kDephosPP*x38(t)-kOnPP*x51(t)*x10(t)-kOffPP*x39(t)+kDephosPP*x39(t)-kOnPP*x51(t)*x19(t)-kOffPP*x40(t)+kDephosPP*x40(t)-kOnPP*x51(t)*x21(t)-kOffPP*x41(t)+kDephosPP*x41(t)-kOnPP*x51(t)*x22(t)-kOffPP*x42(t)+kDephosPP*x42(t),
        x38'(t) = kOnPP*x51(t)*x4(t)-kOffPP*x38(t)-kDephosPP*x38(t),
        x39'(t) = kOnPP*x51(t)*x10(t)-kOffPP*x39(t)-kDephosPP*x39(t),
        x40'(t) = kOnPP*x51(t)*x19(t)-kOffPP*x40(t)-kDephosPP*x40(t),
        x41'(t) = kOnPP*x51(t)*x21(t)-kOffPP*x41(t)-kDephosPP*x41(t),
        x42'(t) = kOnPP*x51(t)*x22(t)-kOffPP*x42(t)-kDephosPP*x42(t),
        # free AMPKAR
        x43'(t) = -kOnAMPK*x43(t)*x8(t)-kOffAMPK*x45(t)-kOnAMPK*x43(t)*x17(t)-kOffAMPK*x46(t)-kOnAMPK*x43(t)*x18(t)-kOffAMPK*x47(t)+kDephosPP1*x48(t),
        x44'(t) = kPhosAMPK*x45(t)+kPhosAMPK*x46(t)+kPhosAMPK*x47(t)-kOnPP1*x52(t)*x44(t)-kOffPP1*x48(t),
        # AMPKAR-pAMPK complexes
        x45'(t) = kOnAMPK*x43(t)*x8(t)-kOffAMPK*x45(t)-kPhosAMPK*x45(t),
        x46'(t) = kOnAMPK*x43(t)*x17(t)-kOffAMPK*x46(t)-kPhosAMPK*x46(t),
        x47'(t) = kOnAMPK*x43(t)*x18(t)-kOffAMPK*x47(t)-kPhosAMPK*x47(t),
        # AMPKAR phosphatase complexes
        x52'(t) = -kOnPP1*x52(t)*x44(t)-kOffPP1*x48(t)+kDephosPP1*x48(t),
        x48'(t) = kOnPP1*x52(t)*x44(t)-kOffPP1*x48(t)-kDephosPP1*x48(t),
        y1(t) = x44(t)/x34(t)
)

