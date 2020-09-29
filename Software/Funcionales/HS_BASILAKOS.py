import sympy as sym
from sympy.utilities.lambdify import lambdify
import numpy as np
import math
from scipy.constants import c as c_luz #metros/segundos
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
c_luz_norm=c_luz/1000

import sys
import os
from os.path import join as osjoin
from pc_path import definir_path
path_git, path_datos_global = definir_path()
os.chdir(path_git)
sys.path.append('./Software/Funcionales/')
from funciones_int import integrador
from funciones_cambio_parametros import params_fisicos_to_modelo
from funciones_taylor import Taylor_HS

#%%
bs=np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 2])
#bs=np.linspace(0.01,2,100)
error_b=np.zeros(len(bs))

omega_m=0.24
H0=73.48
h= H0/100
n=1
Lamb = 3*(1-omega_m)*H0**2
#R_HS = (c_luz_norm/8315)**2 * (omega_m*h**2)/0.13

z0=30


for i, b in enumerate(bs):

    c1,c2=params_fisicos_to_modelo(omega_m,b)
    #print(c1,c2)
    R_HS = H0**2 * 6*(1-omega_m)*c2/c1 #Dan lo mismo, este factor no depende de b
    print(i)

    R = sym.Symbol('R')
    #Calculo F y sus derivadas
    #Ambas F dan las mismas CI para z=0 :)
    #F = R - 2*Lamb * (1 - 1/ (1 + (R/(b*Lamb))) )
    F = R - ((c1*R)/((c2*R/R_HS)+1))
    F_R = sym.simplify(sym.diff(F,R))
    F_2R = sym.simplify(sym.diff(F_R,R))

    z = sym.Symbol('z')
    H = H0*(omega_m*(1+z)**3 + (1-omega_m))**(0.5)
    H_z = sym.diff(H,z)
    H_2z = sym.diff(H_z,z)

    Ricci = 6*(2*H**2-H_z*H*(1+z))
    Ricci_t=sym.diff(Ricci,z)*(-H*(1+z))

    Ricci_ci=sym.lambdify(z,Ricci)
    Ricci_t_ci=sym.lambdify(z,Ricci_t)
    H_ci=sym.lambdify(z,H)
    H_z_ci=sym.lambdify(z,H_z)
    F_ci=sym.lambdify(R,F)
    F_R_ci=sym.lambdify(R,F_R)
    F_2R_ci=sym.lambdify(R,F_2R)

    R0=Ricci_ci(z0)

    x0 = Ricci_t_ci(z0)*F_2R_ci(R0) / (H_ci(z0)*F_R_ci(R0))
    y0 = F_ci(R0) / (6*H_ci(z0)**2*F_R_ci(R0))
    v0 = R0 / (6*H_ci(z0)**2)
    w0 = 1+x0+y0-v0
    r0 = R0/R_HS

    cantidad_zs=int(2*10**6)
    max_step=0.001
    params_modelo=[c1,c2,1]

    params_fisicos = [omega_m,b,H0]
    z, H = integrador(params_fisicos, n, cantidad_zs=cantidad_zs,
                    max_step=max_step, z_inicial=z0, z_final=0)
    H_int = interp1d(z,H)
    # Comparo ambas soluciones
    zs = np.linspace(0,30,100000)
    H_ODE = H_int(zs)
    H_taylor = Taylor_HS(zs,omega_m,b,H0)

    error_b[i] = 100*np.abs(np.mean(1-(H_taylor/H_ODE)))

#%%
%matplotlib qt5
plt.close()
plt.figure()
plt.yscale('log')
plt.grid(True)
plt.xlabel('b')
plt.ylabel('Error Porcentual')
plt.plot(bs,error_b,label='Error en b')
plt.legend(loc='best')
plt.show()
