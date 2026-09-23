# Constants for the project

#Integration Constants
NUM_Z_POINTS = int(10e5)
Z_MIN = 0.0
Z_MAX = 10.0

#Radiation density
OMEGA_R_0 = 2.47e-5 #Photons: omega_gamma = Omega_gamma h^2 (COBE, T_CMB = 2.7255 K)
N_UR = 2.0308 #Massless neutrinos: N_eff = 3.044 with one massive neutrino (as BAO.r_drag_class)
OMEGA_R_H2 = OMEGA_R_0 * (1 + (7/8) * (4/11)**(4/3) * N_UR) #omega_r = Omega_r h^2 (~3.61e-5)
#If False, the background H(z) of all the models neglects the radiation (runs before 2026-09-22)
RADIATION = True

def Omega_r(H_0):
    '''
    Radiation density parameter today (photons + massless neutrinos) for H_0 in km/s/Mpc.
    The massive neutrino is non-relativistic at late times and is included in Omega_m.
    '''
    if not RADIATION or H_0 <= 0:
        return 0.0
    return OMEGA_R_H2 / (H_0/100)**2

#Baryonic matter density of BBN
WB_BBN = 0.02218 #± 0.00055 #Baryon density (eq. 2.8 arXiv:2404.03002)

#Kappa 
KAPPA = 1.0
#KAPPA = 8 * np.pi * G_newton / 3