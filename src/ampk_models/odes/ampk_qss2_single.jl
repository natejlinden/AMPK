ode = @ODEmodel( 
    'AMP': -JAK,
    'ADP': -Jgly + 2*JAK + Jhydro - Joxphos,
    'ATP': Jgly -JAK - Jhydro + Joxphos,
    'AMPK': -J1 - J2 + J3,
    'pAMPK': J1 + J2 - J3,
    'AMPKAR': -J4 + J5,
    'pAMPKAR': J4 - J5
)