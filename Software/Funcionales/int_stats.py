import time
import math
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.constants import c as c_luz #metros/segundos
c_luz_km = c_luz/1000

import sys
import os
from os.path import join as osjoin
from pc_path import definir_path
path_git, path_datos_global = definir_path()
os.chdir(path_git)
sys.path.append('./Software/Funcionales/')

from funciones_condiciones_iniciales import condiciones_iniciales, z_condicion_inicial
from funciones_cambio_parametros import params_fisicos_to_modelo_HS, params_fisicos_to_modelo_ST
from funciones_int import integrador

#%%
if __name__ == '__main__':
    from matplotlib import pyplot as plt

    b = 0.6
    omega_m = 0.24
    H0 = 73.48
    params_fisicos = [omega_m,b,H0]

    z_inicial = 10
    z_final = 0

    cantidad_zs = int(10**5)
    max_step = 10**(-4)

    zs, H_ode = integrador(params_fisicos, n=1, cantidad_zs=cantidad_zs,
                max_step=max_step, z_inicial=z_inicial, z_final=z_final,
                verbose=True,
                model='EXP')
    plt.plot(zs,H_ode)

    #%%
    zs_LCDM = np.linspace(z_final,z_inicial,cantidad_zs)
    Hs_LCDM = H0 * np.sqrt(omega_m * (1+zs_LCDM)**3 + (1-omega_m))

    from matplotlib import pyplot as plt
    %matplotlib qt5
    plt.plot(zs_ode,H_ode)
    plt.plot(zs_LCDM,Hs_LCDM)
    #%%
    #out= np.zeros((len(zs),2))
    #out[:,0] = zs
    #out[:,1] = H_ode/H0
#        np.savetxt('HS_b={}.txt'.format(b), out,
#                    delimiter='\t', newline='\n')
#        print('Completado:{} %'.format(100 * i / len(bs)))
    #lt.plot(zs,H_ode)
    from scipy.integrate import cumtrapz as cumtrapz
    INT = cumtrapz(H_ode**(-1), zs, initial=0)
    DA = (c_luz_km/(1 + zs)) * INT
    plt.plot(zs[1:],DA[1:]) #El 6 está de mas!
#%%google
    #%matplotlib qt5
    #plt.close()
    plt.figure()
    plt.grid(True)
    plt.title('Parámetro de Hubble por integración numérica')
    plt.xlabel('z(redshift)')
    plt.ylabel('E(z)')
    plt.plot(zs, H_ode/H0, label='$\Omega_{m}=0.24, b=0.5$')
    plt.legend(loc='best')
    out= np.zeros((len(zs),2))
    out[:,0] = zs
    out[:,1] = H_ode/H0
    np.savetxt('HS_b=0.02.txt', out,
                delimiter='\t', newline='\n')
    H_ode/H0

    #%% Testeamos el cumtrapz comparado con simpson para la integral de 1/H
    from scipy.integrate import simps,trapz,cumtrapz

    def integrals(ys, xs):
        x_range = []
        y_range = []
        results = []
        for i in range(len(xs)):
            x_range.append(xs[i])
            y_range.append(ys[i])
            integral = simps(y_range, x_range) #Da un error relativo de 10**(-7)
            #integral = trapz(y_range, x_range) #Da un error relativo de 10**(-15)
            results.append(integral)
        return np.array(results)

    integral = cumtrapz(H_ode**(-1),zs, initial=0)
    integral_1 = integrals(H_ode**(-1),zs)

    #%matplotlib qt5
#    plt.figure(1)
    plt.plot(zs,H_ode**(-1))
    plt.plot(zs,integral)
    #plt.plot(zs,integral_1)
    plt.figure(2)
    plt.plot(zs,1-integral_1/integral) #Da un warning xq da 0/0 en z=0

    #Para HS n=2 tarda 9 minutos y 30 seg con b = 0.7
    #Para HS n=2 tarda 9 minutos y 37 seg con b = 0.1
    #Para HS n=2 tarda 9 minutos y 39 seg con b = 0.08

    #Para ST n=1 tarda 8 minutos y 30 seg con b = 2
    #Para ST n=1 tarda 8 minutos y 22 seg con b = 1
    #Para ST n=1 tarda 8 minutos y 11 seg con b = 0.1
    #Para ST n=1 tarda 8 minutos y 17 seg con b = 0.08
