ode = @ODEmodel( 
    x1'(t) =  -(((((VforAK*x3(t)*x1(t))/(kmt*kmm))-((((VforAK*(kmd^2))/(KeqAK*kmt*kmm))*(x2(t)^2))/(kmd^2)))/(1+(x3(t)/kmt)+(x1(t)/kmm)+((x3(t)*x1(t))/(kmt*kmm))+((2*x2(t))/kmd)+((x2(t)^2)/(kmd^2))))),
    x2'(t) =  -(2*kGly*x2(t)*x2(t)) + 2*(((((VforAK*x3(t)*x1(t))/(kmt*kmm))-((((VforAK*(kmd^2))/(KeqAK*kmt*kmm))*(x2(t)^2))/(kmd^2)))/(1+(x3(t)/kmt)+(x1(t)/kmm)+((x3(t)*x1(t))/(kmt*kmm))+((2*x2(t))/kmd)+((x2(t)^2)/(kmd^2))))) + (kHydro*x3(t)), # - ((VmaxOxPhos*((x2(t)/Kadp)^n))/(1+((x2(t)/Kadp)^n))),
    x3'(t) =  (2*kGly*x2(t)*x2(t)) -(((((VforAK*x3(t)*x1(t))/(kmt*kmm))-((((VforAK*(kmd^2))/(KeqAK*kmt*kmm))*(x2(t)^2))/(kmd^2)))/(1+(x3(t)/kmt)+(x1(t)/kmm)+((x3(t)*x1(t))/(kmt*kmm))+((2*x2(t))/kmd)+((x2(t)^2)/(kmd^2))))) - (kHydro*x3(t)),# + ((VmaxOxPhos*((x2(t)/Kadp)^n))/(1+((x2(t)/Kadp)^n))),
    x4'(t) =  -((kCaMKK*CaMKKtot*x1(t)K)/(KmCaMKK+x1(t)K)) - ((kLKB1*LKB1tot*x1(t)K)/(KmLKB1+x1(t)K)) + ((kPP*PPtot*(0.5*(((((x1(t)-x5(t)+KdAMP)^2)+4*KdAMP*x5(t))^0.5)-(x1(t)-x5(t)+KdAMP))))/(KmPP+(0.5*(((((x1(t)-x5(t)+KdAMP)^2)+4*KdAMP*x5(t))^0.5)-(x1(t)-x5(t)+KdAMP))))),
    x5'(t) =  ((kCaMKK*CaMKKtot*x1(t)K)/(KmCaMKK+x1(t)K)) + ((kLKB1*LKB1tot*x1(t)K)/(KmLKB1+x1(t)K)) - ((kPP*PPtot*(0.5*(((((x1(t)-x5(t)+KdAMP)^2)+4*KdAMP*x5(t))^0.5)-(x1(t)-x5(t)+KdAMP))))/(KmPP+(0.5*(((((x1(t)-x5(t)+KdAMP)^2)+4*KdAMP*x5(t))^0.5)-(x1(t)-x5(t)+KdAMP))))),
    x6'(t) =  -((((kAMPK*x5(t)*x6(t))/KmAMPK)+((beta*kAMPK*x5(t)*x6(t)*x1(t))/(alpha*KmAMPK*KdAMP)))/(1+(x6(t)/KmAMPK)+(x1(t)/KdAMP)+((x6(t)*x1(t))/(alpha*KmAMPK*KdAMP)))) + ((kPP1*PP1tot*x7(t))/(KmPP1+x7(t))),
    x7'(t) =  ((((kAMPK*x5(t)*x6(t))/KmAMPK)+((beta*kAMPK*x5(t)*x6(t)*x1(t))/(alpha*KmAMPK*KdAMP)))/(1+(x6(t)/KmAMPK)+(x1(t)/KdAMP)+((x6(t)*x1(t))/(alpha*KmAMPK*KdAMP)))) - ((kPP1*PP1tot*x7(t))/(KmPP1+x7(t))),
    y1(t) =  x7(t) / (x6(t) + x7(t))
);
