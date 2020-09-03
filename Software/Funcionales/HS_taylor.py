import sympy as sym
from sympy.utilities.lambdify import lambdify
import numpy as np
import math

import sys
import os
from os.path import join as osjoin
from pc_path import definir_path
path_git, path_datos_global = definir_path()
os.chdir(path_git)
sys.path.append('./Software/Funcionales/')
from funciones_data import leer_data_cronometros
os.chdir(path_git+'/Software/Estadística/Datos/')
z_data, H_data, dH  = leer_data_cronometros('datos_cronometros.txt')

#%%

def Taylor_HS(z,omega_m,b,H0):
    '''Calculo del H(z) sugerido por Basilakos para el modelo de Hu-Sawicki
    N!=a, N=ln(a)=-ln(1+z)'''
    omega_r=0
    N = sym.Symbol('N')
    Hs_tay=N
    Hs_tay = (H0**2*(1/((omega_m-4*math.e**(3*N)*(omega_m+omega_r-1))**8) * (b**2)*(math.e**(5*N))*((omega_m+omega_r-1)**3) *(37*math.e**(N)*omega_m**6-4656*math.e**(4*N)*omega_m**5*(omega_m+omega_r-1)-
    7452*math.e**(7*N)*omega_m**4*(omega_m+omega_r-1)**2-
    8692*math.e**(3*N)*omega_m**4*omega_r*(omega_m+omega_r-1)-
    4032*math.e**(2*N)*omega_m**3*omega_r**2*(omega_m+omega_r-1)+
    25408*math.e**(10*N)*omega_m**3*(omega_m+omega_r-1)**3-
    25728*math.e**(6*N)*omega_m**3*omega_r*(omega_m+omega_r-1)**2-
    17856*math.e**(5*N)*omega_m**2*omega_r**2*(omega_m+omega_r-1)**2-
    22848*math.e**(13*N)*omega_m**2*(omega_m+omega_r-1)**4+
    22016*math.e**(9*N)*omega_m**2*omega_r*(omega_m+omega_r-1)**3-
    9216*math.e**(8*N)*omega_m*omega_r**2*(omega_m+omega_r-1)**3+
    9216*math.e**(16*N)*omega_m*(omega_m+omega_r-1)**5-
    2048*math.e**(12*N)*omega_m*omega_r*(omega_m+omega_r-1)**4+
    1024*math.e**(19*N)*(omega_m+omega_r-1)**6+
    3072*math.e**(15*N)*omega_r*(omega_m+omega_r-1)**5+40*omega_m**5*omega_r)+
    ((2*b*math.e**(2*N)*(omega_m+omega_r-1)**2*(-6*math.e**(N)*omega_m**2+3*math.e**(4*N)*omega_m*(omega_m+omega_r-1)+12*math.e**(7*N)*(omega_m+omega_r-1)**2+4*math.e**(3*N)*omega_r*(omega_m+omega_r-1)-7*omega_m*omega_r))/(4*math.e**(3*N)*(omega_m+omega_r-1)-omega_m)**3)+
    (math.e**(-3*N)-1)*omega_m + (math.e**(-4*N)-1)*omega_r+1))**(0.5)

    func = lambdify(N, Hs_tay,'numpy') # returns a numpy-ready function

    N_dato = -np.log(1+z)
    numpy_array_of_results = func(N_dato)
    return numpy_array_of_results




#%%
if __name__ == '__main__':
    def H_LCDM(z, omega_m, H_0):
        omega_lambda = 1 - omega_m
        H = H_0 * np.sqrt(omega_m * (1 + z)**3 + omega_lambda)
        return H

    #Parámetros
    from matplotlib import pyplot as plt
    %matplotlib qt5
    omega_m = 0.24
    b = 2
    H0 = 73.48

    zs = np.linspace(0,2,10000);
    HL_posta = H_LCDM(zs,omega_m,H0)
    H_taylor = Taylor_HS(zs,omega_m,b,H0)

    plt.close()
    plt.figure()
    plt.grid(True)
    plt.xlabel('z (redshift)')
    plt.ylabel('H(z)')
    #plt.hlines(0, xmin=0 ,xmax = 2)
    plt.plot(zs,HL_posta/H0,label='LCDM')
    plt.plot(zs,H_taylor/H0,'-.',label='Taylor')
    #plt.plot(z_data,H_data,'.',label='Cronometros')
    plt.legend(loc='best')
    plt.show()
