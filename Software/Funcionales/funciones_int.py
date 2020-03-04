"""
Created on Sun Feb  2 13:28:48 2020

@author: matias
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate import simps as simps
import time

from scipy.interpolate import interp1d
from scipy.constants import c as c_luz #metros/segundos

def dX_dz(z, variables,*params_modelo):
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
    gamma = lambda r,b,c,d,n: ((1+d*r**n) * (-b*n*r**n + r*(1+d*r**n)**2)) / (b*n*r**n * (1-n+d*(1+n)*r**n))
    [B,C,D,N] = params_modelo
    G = gamma(r,B,C,D,N)

    s0 = (-w + x**2 + (1+v)*x - 2*v + 4*y) / (z+1)
    s1 = - (v*x*G - x*y + 4*y - 2*y*v) / (z+1)
    s2 = -v * (x*G + 4 - 2*v) / (z+1)
    s3 = w * (-1 + x + 2*v) / (z+1)
    s4 = -x*r*G/(1+z)
    return [s0,s1,s2,s3,s4]



def integrador(cond_iniciales, params_modelo ,sistema_ec=dX_dz,
                z_inicial=0, z_final=3, cantidad_zs=100, max_step=0.1,
                ): #cantidad_zs_ideal = 7000, # max_step_ideal = 0.005
    '''Esta función integra el sistema de ecuaciones diferenciales entre
    z_inicial y z_final, dadas las condiciones iniciales y los parámetros
    del modelo.'''

    # Integramos el vector v y calculamos el Hubble
    zs = np.linspace(z_inicial,z_final,cantidad_zs)
    hubbles = np.zeros(len(zs))

    t1 = time.time()
    for i in range(len(zs)):
        zf = zs[i] # ''z final'' para cada paso de integracion
        sol = solve_ivp(sistema_ec, [z_inicial,zf],
         cond_iniciales,args=params_modelo, max_step=max_step)
        int_v = simps((sol.y[2])/(1+sol.t),sol.t) # integro desde 0 a z
        hubbles[i]=(1+zf)**2 * np.e**(-int_v)
    t2 = time.time()

    print('Duración {} minutos y {} segundos'.format(int((t2-t1)/60),
          int((t2-t1) - 60*int((t2-t1)/60))))
    return zs, hubbles

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


def magn_aparente_teorica(z,E,zhel,zcmb,H_0=73.48):
    '''A partir de un array de redshift y un array de la magnitud E = H_0/H
    que salen de la integración numérica, se calcula el mu teórico que deviene
    del modelo. muth = 25 + 5 * log_{10}(d_L),
    donde d_L =  (c/H_0) (1+z) int(dz'/E(z'))'''

    d_c=np.zeros(len(E)) #Distancia comovil
    for i in range (1, len(E)):
        d_c[i] = 0.001*(c_luz/H_0) * simps(1/E[:i],z[:i])

    dc_int = interp1d(z,d_c) #Interpolamos

    d_L = (1+zhel) * dc_int(zcmb) #Obs, Caro multiplica por Zhel, con Zobs da un poquin mejor

    ##Magnitud aparente teorica
    muth = 25 + 5 * np.log10(d_L)
    return muth
