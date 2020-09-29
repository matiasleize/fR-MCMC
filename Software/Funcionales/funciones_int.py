"""
Created on Sun Feb  2 13:28:48 2020

@author: matias
"""

import time
import numpy as np
from matplotlib import pyplot as plt
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

def dX_dz(z, variables,*params_modelo, model='HS'):
    '''Defino el sistema de ecuaciones a resolver. El argumento params_modelo
    es una lista donde los primeros n-1 elementos son los parametros del sistema,
    mientras que el útimo argumento especifica el modelo en cuestión,
    matematicamente dado por la función gamma.'''

    x = variables[0]
    y = variables[1]
    v = variables[2]
    w = variables[3]
    r = variables[4]

    if model == 'ST':
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


def integrador(params_fisicos, n=1, cantidad_zs=int(10**6), max_step=0.01,
                z_inicial=30, z_final=0, sistema_ec=dX_dz, verbose=False):

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

def plot_sol(solucion):

    '''Dada una gamma (que especifica el modelo) y una solución de las
    variables dinámicas, grafica estas por separado en una figura de 4x4.'''

    f, axes = plt.subplots(2,3)
    ax = axes.flatten()

    color = ['b','r','g','y','k']
    y_label = ['x','y','v','w','r']
    [ax[i].plot(solucion.t,solucion.y[i],color[i]) for i in range(5)]
    [ax[i].set_ylabel(y_label[i],fontsize='medium') for i in range(5)];
    [ax[i].set_xlabel('z (redshift)',fontsize='medium') for i in range(5)];
    [ax[i].invert_xaxis() for i in range(5)]; #Doy vuelta los ejes
    plt.show()

#%%
if __name__ == '__main__':

    from matplotlib import pyplot as plt
    import numpy as np
    from scipy.integrate import cumtrapz as cumtrapz

    import sys
    import os
    from os.path import join as osjoin
    from pc_path import definir_path
    path_git, path_datos_global = definir_path()
    os.chdir(path_git)
    sys.path.append('./Software/Funcionales/')
    from funciones_taylor import Taylor_HS

    sistema_ec=dX_dz
    z_inicial = 30
    z_final = 0
    cantidad_zs = int(10**6)
    max_step = 0.01

    omega_m = 0.24
    b = 2
    H0 = 73.48
    params_fisicos = [omega_m,b,H0]

    cond_iniciales= condiciones_iniciales(*params_fisicos)
    eta = (c_luz_norm/(8315*100)) * np.sqrt(omega_m/0.13)
    c1,c2 = params_fisicos_to_modelo(omega_m,b)
    params_modelo=[c1,c2,1]

    #%% Grafico lo que sale de la ODE
    zs = np.linspace(z_inicial,z_final,cantidad_zs)
    t1 = time.time()
    sol = solve_ivp(sistema_ec, [z_inicial,z_final],
          cond_iniciales,t_eval=zs,args=params_modelo, max_step=max_step)
    t2 = time.time()
    print('Duración {} minutos y {} segundos'.format(int((t2-t1)/60),
          int((t2-t1) - 60*int((t2-t1)/60))))

    plt.figure()
    plot_sol(sol)
    print(np.all(zs==sol.t))
    #%%
    def H_LCDM(z, omega_m, H_0):
        omega_lambda = 1 - omega_m
        H = H_0 * np.sqrt(omega_m * (1 + z)**3 + omega_lambda)
        return H

    r=sol.y[4][::-1]
    v=sol.y[2][::-1]
    z_int=sol.t[::-1]

    E_LCDM = H_LCDM(z_int,omega_m,1)
    E_ODE = eta*np.sqrt(r/(6*v))
    #E_taylor = Taylor_HS(z_int,omega_m,b,H0)/H0


    #%% Ploteo y comparo las soluciones
    #%matplotlib qt5
    plt.close()
    plt.figure()
    plt.grid(True)
    plt.xlabel('z (redshift)')
    plt.ylabel('H(z)')
    plt.plot(z_int,E_ODE,label='ODE')
    #plt.plot(z_int,E_taylor,'-.',label='Taylor')
    plt.plot(z_int,E_LCDM,'-.',label='LCDM')
    plt.legend(loc='best')
    plt.xlim([0,3])
    plt.ylim([0,6])

    #%% Calculo el error porcentual entre la ODE y el taylor a b fijo.
    plt.figure()
    plt.grid(True)
    plt.xlabel('z (redshift)')
    plt.ylabel('Error porcentual')
    plt.plot(z_int,100*(1-(E_taylor/E_ODE)),label='Error porcentual')
    plt.legend(loc='best')
    plt.show()
    np.mean(100*(1-(E_taylor/E_ODE)))

    #%% Da bien con un error de 10(-12)
    plt.plot(sol.t,sol.y[3]+sol.y[2]-sol.y[0]-sol.y[1])

    #%% Ricci(z) para HS
    r_inv=sol.y[4][::-1]
    zs=np.linspace(0,10,len(r_inv))
    plt.grid(True)
    plt.xlabel('z (redshift)')
    plt.ylabel(r'$R_{HS}/R$')
    plt.plot(zs,r_inv)
    plt.show()

    #%% Esto es lo viejo que no anda!
    plt.figure()
    z_int=sol.t[::-1]
    v_int=sol.y[2][::-1]
    int_v =  cumtrapz(v_int/(1+z_int),z_int,initial=0)
    E = (1+z_int)**2 * np.exp(-int_v)
    plt.figure()
    plt.plot(z_int,E_LCDM,'.')
    plt.plot(z_int,E)
    #%% Error en la integración de v de 10(-3) (Tambien Viejo)
    dx=z_int[1]-z_int[0]
    dH = np.diff(E)/dx
    plt.plot(100*(1-((2-v_int)*E/(1+z_int))[1:]/dH))
