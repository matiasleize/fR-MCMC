"""
Created on Sun Feb  2 13:28:48 2020

@author: matias
"""
from matplotlib import pyplot as plt

from scipy.integrate import solve_ivp
import numpy as np
from scipy.integrate import simps as simps
import time

from scipy.interpolate import interp1d
from scipy.constants import c
H_0 =  73.48



def integrador(sistema_ec, cond_iniciales, parametros_modelo, z_inicial=0, z_final=3):    
    '''Esta función integra el sistema de ecuaciones diferenciales entre 
    z_inicial y z_final, dadas las condiciones iniciales y los parámetros
    del modelo.'''
    [b,d,c,r_0,n] = parametros_modelo
    # Resolvemos
    sol = solve_ivp(sistema_ec, [z_inicial,z_final], cond_iniciales, max_step=0.01)
    
    # Integramos el vector v y calculamos el Hubble
    zs = np.linspace(0,3,7000) #7000
    hubbles = np.zeros(len(zs))
    
    lala = np.zeros(len(zs))
    
    t1 = time.time()
    for i in range(len(zs)):
        zi = zs[0]
        zf = zs[i]
        sol = solve_ivp(sistema_ec, [zi,zf], cond_iniciales, max_step=0.05)      # 0.005
    
        int_v = simps((sol.y[2])/(1+sol.t),sol.t) 
        lala[i] = int_v
        hubbles[i]=(1+zf)**2 * np.e**(-int_v) # integro desde 0 a z, ya arreglado
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
    

def magn_aparente_teorica(z,E,zhel,zcmb):
    '''A partir de un array de redshift y un array de la magnitud E = H_0/H
    que salen de la integración numérica, se calcula el mu teórico que deviene
    del modelo. muth = 25 + 5 * log_{10}(d_L), 
    donde d_L =  (c/H_0) (1+z) int(dz'/E(z'))'''
    
    d_c=np.zeros(len(E)) #Distancia comovil
    for i in range (1, len(E)):
        d_c[i] = 0.001*(c/H_0) * simps(1/E[:i],z[:i])
        
    dc_int = interp1d(z,d_c) #Interpolamos 
    
    d_L = (1+zhel) * dc_int(zcmb) #Obs, Caro multiplica por Zhel, con Zobs da un poquin mejor
    
    ##magnitud aparente teorica
    muth = 25 + 5 * np.log10(d_L) #base cosmico
    return muth