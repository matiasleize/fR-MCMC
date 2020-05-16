"""
Created on Sun Feb  2 13:28:48 2020

@author: matias
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate import simps as simps
from scipy.integrate import cumtrapz as cumtrapz
import time

from scipy.interpolate import interp1d
from scipy.constants import c as c_luz #metros/segundos

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

    #nombre_modelo = params_modelo[-1]
    #if nombre_modelo == 'Star':
#        gamma = lambda r,b,c,d,n : b*r
#        [B,C,D,N,_] = params_modelo
#        G = gamma(r,B,C,D,N)
#    elif nombre_modelo == 'HS':
        #if params_modelo[3]==1:
        #    gamma = lambda y,v: y*v/(2*(y-v)**2)
        #    G = gamma(y,v)
        #else:
            #[b,d,c,n] = parametros_modelo
    if model == 'HS':
        gamma = lambda r,b,d,n: ((1+d*r**n) * (-b*n*r**n + r*(1+d*r**n)**2)) / (b*n*r**n * (1-n+d*(1+n)*r**n))
        [B,D,N] = params_modelo
    else:
        pass
    #G = gamma(r,B,D,N)
    G = 0
    s0 = (-w + x**2 + (1+v)*x - 2*v + 4*y) / (z+1)
    s1 = - (v*x*G - x*y + 4*y - 2*y*v) / (z+1)
    s2 = -v * (x*G + 4 - 2*v) / (z+1)
    s3 = w * (-1 + x + 2*v) / (z+1)
    s4 = -x*r*G/(1+z)
    return [s0,s1,s2,s3,s4]



def integrador(cond_iniciales, params_modelo, cantidad_zs, max_step,
                sistema_ec=dX_dz, z_inicial=0, z_final=3, verbose=False):
                #cantidad_zs_ideal = 7000, # max_step_ideal = 0.005

    '''Esta función integra el sistema de ecuaciones diferenciales entre
    z_inicial y z_final, dadas las condiciones iniciales y los parámetros
    del modelo.
        Para el integrador, dependiendo que datos se usan hay que ajustar
    el cantidad_zs y el max_step.
    Para cronometros: max_step=0.1
    Para supernovas:  max_step=0.05
    '''
    t1 = time.time()
    # Integramos el vector v y calculamos el Hubble
    zs = np.linspace(z_inicial,z_final,cantidad_zs)
    sol = solve_ivp(sistema_ec, (z_inicial,z_final),
    cond_iniciales,t_eval=zs,args=params_modelo, max_step=max_step)

    if (len(sol.t)!=cantidad_zs):
        print('Está integrando mal!')

    #Chequear este paso!
    #hubbles = np.ones(len(sol.t)) #Para que el primer valor de este array sea 1!
    #for i in range (1, len(zs)):
    #    int_v =  simps((sol.y[2][:i])/(1+sol.t[:i]),sol.t[:i])
    #    hubbles[i] = (1+sol.t[i])**2 * np.e**(-int_v)

    int_v =  cumtrapz(sol.y[2]/(1+sol.t), sol.t, initial=0)
    hubbles = (1+sol.t)**2 * np.exp(-int_v)
    t2 = time.time()
    if verbose == True:
        print('Duración {} minutos y {} segundos'.format(int((t2-t1)/60),
        int((t2-t1) - 60*int((t2-t1)/60))))
    return sol.t, hubbles

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

    sistema_ec=dX_dz
    z_inicial=0
    z_final=3
    cantidad_zs = 100
    max_step=0.1
    verbose=True


    x_0 = -0.339
    y_0 = 1.246
    v_0 = 1.64
    w_0 = 1 + x_0 + y_0 - v_0
    r_0 = 41
    ci = [x_0, y_0, v_0, w_0, r_0] #Condiciones iniciales
    cond_iniciales=ci

    #c1_true = 1
    #c2_true = 1/19
    #n=1
    params_modelo=[1,1/19,1]

#%% Forma nueva de integrar
    #%matplotlib qt5
    from matplotlib import pyplot as plt

    plt.figure()
    zs = np.linspace(z_inicial,z_final,cantidad_zs)
    t1 = time.time()
    sol = solve_ivp(sistema_ec, [z_inicial,z_final],
          cond_iniciales,t_eval=zs,args=params_modelo, max_step=max_step)
    t2 = time.time()
    print('Duración {} minutos y {} segundos'.format(int((t2-t1)/60),
          int((t2-t1) - 60*int((t2-t1)/60))))
    plot_sol(sol)

    plt.figure()
    #E=np.ones(len(sol. t))
    #for i in range (1, len(sol.t)):
    #    int_v =  simps((sol.y[2][:i])/(1+sol.t[:i]),sol.t[:i])
    #    E[i] = (1+sol.t[i])**2 * np.e**(-int_v)
    int_v =  cumtrapz((sol.y[2])/(1+sol.t),sol.t,initial=0)
    E = (1+sol.t)**2 * np.exp(-int_v)
    plt.plot(sol.t,E)
#%% Forma vieja de integrar
    plt.figure()
    t1 = time.time()
    zs = np.linspace(z_inicial,z_final,cantidad_zs)
    hubbles_1 = np.zeros(len(zs))
    for i in range(len(zs)):
        zf = zs[i] # ''z final'' para cada paso de integracion
        sol_1 = solve_ivp(sistema_ec,[z_inicial,zf], cond_iniciales,args=params_modelo, max_step=max_step)
        int_v = simps((sol_1.y[2])/(1+sol_1.t),sol_1.t) # integro desde 0 a z
        hubbles_1[i] = (1+zf)**2 * np.exp(-int_v)
    t2 = time.time()
    print('Duración {} minutos y {} segundos'.format(int((t2-t1)/60),
          int((t2-t1) - 60*int((t2-t1)/60))))
    plt.plot(zs,hubbles_1)
