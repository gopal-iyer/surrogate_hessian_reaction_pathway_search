import numpy as np
from math import exp, sqrt, pi

def kinetics(G1, dG1, G2, dG2, qty):
    DG = G2 - G1
    dDG = sqrt(dG1**2. + dG2**2.)
    kB = 8.617333262e-5
    T = 298.15
    kBT = kB*T
    h = 2 * pi * 6.582119569e-16
    
    print('Delta G = ', DG, " +/- ", dDG)
    if qty == 'Keq':
        Keq = exp(-DG/kBT)
        dKeq = (1/kBT) * dDG * Keq
        print("K_eq = ", Keq, " +/- ", dKeq)
    
    elif qty == 'kf':
        kf = (kBT/h) * exp(-DG/kBT)
        dkf = (1/kBT) * dDG * kf
        print(f"k_f = {kf:.7e} +/- {dkf:.7e}")