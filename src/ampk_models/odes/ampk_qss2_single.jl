ode = @ODEmodel( 
    x1'(t) =  -(kForAK*x3(t)*x1(t)) + ((kRevAK*x2(t)*x2(t))),
    x2'(t) =  -2*(2*kGly*x2(t)*x2(t)) + 2*(kForAK*x3(t)*x1(t)) - 2*((kRevAK*x2(t)*x2(t))) + (kHydro*x3(t)), #- ((VmaxOxPhos * ((x2(t)/Kadp)^n))/(1 + ((x2(t)/Kadp)^n))),
    x3'(t) =  2*(2*kGly*x2(t)*x2(t)) - (kForAK*x3(t)*x1(t)) + ((kRevAK*x2(t)*x2(t))) - ( kHydro*x3(t)), # + ((VmaxOxPhos * ((x2(t)/Kadp)^n))/(1 + ((x2(t)/Kadp)^n))),
    x4'(t) = -((kCaMKK*CaMKKtot*x4(t))/(KmCaMKK + x4(t))) - ((((kLKB1*LKB1tot*x4(t))/KmLKB1) + ((beta2*kLKB1*LKB1tot*x4(t)*x1(t))/(alpha2*KmLKB1*KdAMP)))/(1 + (x4(t)/KmLKB1) + (x1(t)/KdAMP) + ((x4(t)*x1(t))/(alpha2*KmLKB1*KdAMP)))) + ((kPP*PPtot*S)/(KmPP + S)),
    x5'(t) = ((kCaMKK*CaMKKtot*x4(t))/(KmCaMKK + x4(t))) + ((((kLKB1*LKB1tot*x4(t))/KmLKB1) + ((beta2*kLKB1*LKB1tot*x4(t)*x1(t))/(alpha2*KmLKB1*KdAMP)))/(1 + (x4(t)/KmLKB1) + (x1(t)/KdAMP) + ((x4(t)*x1(t))/(alpha2*KmLKB1*KdAMP)))) - ((kPP*PPtot*S)/(KmPP + S)),
    x6'(t) = -((((kAMPK*x5(t)*x6(t))/KmAMPK) + ((beta*kAMPK*x5(t)*x6(t)*x1(t))/(alpha*KmAMPK*KdAMP)))/(1 + (x6(t)/KmAMPK) + (x1(t)/KdAMP) + ((x6(t)*x1(t))/(alpha*KmAMPK*KdAMP)))) + ((kPP1*PP1tot*x7(t))/(KmPP1 + x7(t))),
    x7'(t) = ((((kAMPK*x5(t)*x6(t))/KmAMPK) + ((beta*kAMPK*x5(t)*x6(t)*x1(t))/(alpha*KmAMPK*KdAMP)))/(1 + (x6(t)/KmAMPK) + (x1(t)/KdAMP) + ((x6(t)*x1(t))/(alpha*KmAMPK*KdAMP)))) - ((kPP1*PP1tot*x7(t))/(KmPP1 + x7(t))),
    y1(t) = x7(t) / (x6(t) + x7(t))
)