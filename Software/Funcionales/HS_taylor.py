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

    N = sym.Symbol('N')

    HL = H0 * (omega_m * (math.e**(-3*N)) + (1 - omega_m))**(0.5)

    #Calculo las derivadas
    HL_N = sym.diff(HL,N)
    HL_2N = sym.diff(HL_N,N)
    HL_3N = sym.diff(HL_2N,N)
    HL_4N = sym.diff(HL_3N,N)

    #Calculo los términos del taylor
    dE21 =  ((H0**2 * (-1 + omega_m)**2 * (6 * HL**2 + 4 * HL_N**2 + HL * (15 * HL_N + 2 * HL_2N))) / (2 * HL * (2 * HL + HL_N)**3))

    #REVISAR
    t1 = ((-1 + omega_m) * H0**2)

    t2 = (540 * t1 * HL_N**2 + 124 * HL_N**4 + 12*HL_N**3 * HL_2N
    + 3 * t1 * HL_2N * (17 * HL_2N + 4 * HL_3N) + 2 * t1 * HL_N * (129*HL_2N
    + 12*HL_3N - HL_4N))

    t3 = (84 * t1 * HL_N**3 - 53 * HL_N**5
    - 3 * HL_N**4 * HL_2N + 21 * t1 * HL_2N**3 + 3 * t1 * HL_N * HL_2N * (41*HL_2N - 4*HL_3N)
    + t1 * HL_N**2 * (217 * HL_2N - 42 * HL_3N
    + HL_4N))


    dE22 = ((H0**4 * (-1 + omega_m)**3  * (128 * HL**8 - 32 * t1 * HL_N**6
            + 32 * HL**7 * (25*HL_N + 3*HL_2N) - 2 * t1 * HL * HL_N**4 * (139*HL_N + 22*HL_2N)
            + 16 * HL**6 * (9 * t1 + 89*HL_N**2 + 12 * HL_N * HL_2N)
            + HL**2 * HL_N**2 * (-749 * t1 * HL_N**2 + 9 * HL_N**4 - 48 * t1 * HL_2N**2
            - 4 * t1 * HL_N * (74*HL_2N - 3*HL_3N)) + 8 * HL**5 * (144 * t1 * HL_N
            + 146 * HL_N**3 + 18 * HL_N**2 * HL_2N + t1 * (15*HL_2N - 6*HL_3N - 4*HL_4N))
            + 4 * HL**4 * t2 - 2 * HL**3 * t3
            )) / (4*HL**4 * (2*HL + HL_N)**8))

    EL2 = (omega_m * (math.e**(-3*N)) + (1 - omega_m))

    HS_tay = H0 * (EL2 - b * dE21 - (b**2) * dE22)**(0.5)
    func = lambdify(N, HS_tay,'numpy') # returns a numpy-ready function

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
