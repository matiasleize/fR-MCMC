import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.constants import c as c_luz #metros/segundos
c_luz_norm = c_luz/1000;

import sys
import os
from os.path import join as osjoin
from pc_path import definir_path
path_git, path_datos_global = definir_path()
os.chdir(path_git)
sys.path.append('./Software/Funcionales/')

from funciones_condiciones_iniciales import condiciones_iniciales
from funciones_cambio_parametros import params_fisicos_to_modelo
#%%

def dX_dz(z, variables,*params_modelo,model='HS'):
    '''Defino el sistema de ecuaciones a resolver. El argumento params_modelo
    es una lista donde los primeros n-1 elementos son los parametros del sistema,
    mientras que el útimo argumento especifica el modelo en cuestión,
    matematicamente dado por la función gamma.'''

    x = variables[0]
    y = variables[1]
    v = variables[2]
    w = variables[3]
    r = variables[4]

    #[p1,p2,n,model]=imput

    if model == 'ST':
        [lamb,Rs,N] = params_modelo
        if N==1:
            gamma = lambda r,lamb: (r**2+1) * ((r**2+1)**2-2*r*lamb)/(2*r*lamb*(3*r**2-1))
            G = gamma(r,lamb)
        else:
            print('Guarda que estas poniendo n!=1')
            pass
    if model == 'HS':
        [B,D,N] = params_modelo
        if N==1:
            gamma = lambda r,b,d: ((1+d*r) * (-b*r + r*(1+d*r)**2)) / (b*2*d*r**2)
            G = gamma(r,B,D)
        else:
            gamma = lambda r,b,d,n: ((1+d*r**n) * (-b*n*r**n + r*(1+d*r**n)**2)) / (b*n*(1 - n + d * (1+n) * r**n)*r**n)
            G = gamma(r,B,D,N)
            print('Guarda que estas poniendo n!=1')
    else:
        pass
    s0 = (-w + x**2 + (1+v)*x - 2*v + 4*y) / (z+1)
    s1 = (- (v*x*G - x*y + 4*y - 2*y*v)) / (z+1)
    s2 = (-v * (x*G + 4 - 2*v)) / (z+1)
    s3 = (w * (-1 + x + 2*v)) / (z+1)
    s4 = (-(x * r * G)) / (1+z)
    return [s0,s1,s2,s3,s4]


def integrador(params_fisicos, n=1, cantidad_zs=int(10**5), max_step=0.003,
                z_inicial=30, z_final=0, sistema_ec=dX_dz, verbose=False,
                model='HS'):

    '''Esta función integra el sistema de ecuaciones diferenciales entre
    z_inicial y z_final, dadas las condiciones iniciales y los parámetros
    del modelo.
        Para el integrador, dependiendo que datos se usan hay que ajustar
    el cantidad_zs y el max_step.
    Para cronometros: max_step=0.1
    Para supernovas:  max_step=0.05
    '''
    t1 = time.time()

    [omega_m,b,H0] = params_fisicos

    #Calculo las condiciones cond_iniciales, eta
    # y los parametros de la ecuación
    cond_iniciales = condiciones_iniciales(*params_fisicos)
    eta = (c_luz_norm/(8315*100)) * np.sqrt(omega_m/0.13)
    c1, c2 = params_fisicos_to_modelo(omega_m,b)

    #Integramos el sistema
    zs_int = np.linspace(z_inicial,z_final,cantidad_zs)
    sol = solve_ivp(sistema_ec, (z_inicial,z_final),
        cond_iniciales, t_eval=zs_int, args=[c1,c2,n], max_step=max_step)

    if (len(sol.t)!=cantidad_zs):
        print('Está integrando mal!')
    if np.all(zs_int==sol.t)==False:
        print('Hay algo raro!')

    #Calculamos el Hubble
    zs = sol.t[::-1]
    v=sol.y[2][::-1]
    r=sol.y[4][::-1]
    Hs = H0 * (eta * np.sqrt(r/(6*v)))

    t2 = time.time()

    if verbose == True:
        print('Duración {} minutos y {} segundos'.format(int((t2-t1)/60),
              int((t2-t1) - 60*int((t2-t1)/60))))

    return zs, Hs

#%%
if __name__ == '__main__':
    from matplotlib import pyplot as plt

    omega_m = .24 #Con valores mas altos de omega deja de andar para b=0.01! Hacer
    #mapa de paparams para Hs n=2
    b = 2
    H0 = 73.48
    params_fisicos = [omega_m,b,H0]

    z_inicial = 30
    z_final = 0
    cantidad_zs = int(10**4)
    max_step = 0.003

    zs, H_ode = integrador(params_fisicos, n=1, cantidad_zs=cantidad_zs,
                max_step=max_step, z_inicial=30, z_final=0, sistema_ec=dX_dz,
                verbose=True)

    #%matplotlib qt5
    #plt.close()
    plt.figure()
    plt.grid(True)
    plt.title('Parámetro de Hubble por integración numérica')
    plt.xlabel('z(redshift)')
    plt.ylabel('H(z)')
    plt.plot(zs,H_ode,label='ODE')
    plt.legend(loc='best')


#%% Testeamos el cumtrapz comparado con simpson para la integral de 1/H
    from scipy.integrate import simps,trapz

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
