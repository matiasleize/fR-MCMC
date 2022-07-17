import sympy as sym
from sympy.utilities.lambdify import lambdify
import numpy as np
import math
from scipy.constants import c as c_luz #meters/seconds
c_luz_norm=c_luz/1000;

import sys
import os
from os.path import join as osjoin
from pc_path import definir_path
path_git, path_datos_global = definir_path()
os.chdir(path_git)
sys.path.append('./Software/utils/')
from change_of_parameters import params_fisicos_to_modelo_HS


def condiciones_iniciales(omega_m,b,H0,z0=30):

    c1,c2=params_fisicos_to_modelo_HS(omega_m,b)
    h= H0/100
    Lamb = 3*(1-omega_m)*H0**2
    R_HS = (c_luz_norm/8315)**2 * (omega_m*h**2)/0.13
    #R_HS = H0**2 * 6*(1-omega_m)*c2/c1 #Dan lo mismo, este factor no depende de b

    R = sym.Symbol('R')
    #Calculo F y sus derivadas
    #Ambas F dan las mismas CI para z=0 :)
    F = R - ((c1*R)/((c2*R/R_HS)+1))
    F = R - 2*Lamb * (1 - 1/ (1 + (R/(b*Lamb))) )

    F_R = sym.simplify(sym.diff(F,R))
    F_2R = sym.simplify(sym.diff(F_R,R))

    z = sym.Symbol('z')
    H = H0*(omega_m*(1+z)**3 + (1-omega_m))**(0.5)
    #H_z = H0*((1+z)**3 *3 * omega_m)/(2*(1+omega_m*(-1+(1+z)**3))**(0.5))
    H_z = sym.simplify(sym.diff(H,z))
    H_2z = sym.simplify(sym.diff(H_z,z))

    Ricci = (12*H**2 + 6*H_z*(-H*(1+z)))
    Ricci_t=sym.simplify(sym.diff(Ricci,z)*(-H*(1+z)))

    Ricci_ci=sym.lambdify(z,Ricci)
    Ricci_t_ci=sym.lambdify(z,Ricci_t)
    H_ci=sym.lambdify(z,H)
    H_z_ci=sym.lambdify(z,H_z)
    F_ci=sym.lambdify(R,F)
    F_R_ci=sym.lambdify(R,F_R)
    F_2R_ci=sym.lambdify(R,F_2R)

    R0=Ricci_ci(z0)
    Ricci_t_ci(z0)
    H_ci(z0) #Chequie que de lo esperado x Basilakos
    H_z_ci(z0) #Chequie que de lo esperado x Basilakos
    F_ci(R0) # debe ser simil a R0-2*Lamb
    F_R_ci(R0) # debe ser simil a 1
    F_2R_ci(R0) # debe ser simil a 0

    x0 = Ricci_t_ci(z0)*F_2R_ci(R0) / (H_ci(z0)*F_R_ci(R0))
    y0 = F_ci(R0) / (6*(H_ci(z0)**2)*F_R_ci(R0))
    v0 = R0 / (6*H_ci(z0)**2)
    w0 = 1+x0+y0-v0
    r0 = R0/R_HS

    return[x0,y0,v0,w0,r0]
#%%
if __name__ == '__main__':
    z0 = 0
    omega_m = 0.24
    b = 2
    H0 = 73.48

    cond_iniciales=condiciones_iniciales(omega_m,b,H0,z0)
    print(cond_iniciales)
    w0 = 1+cond_iniciales[0]*10+cond_iniciales[1]-cond_iniciales[2]
    w0
#%%
    c1,c2 = params_fisicos_to_modelo(omega_m,b)
    R_HS = H0**2 * 6*(1-omega_m)*c2/c1
