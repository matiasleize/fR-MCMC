#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 13:28:48 2020

@author: matias
"""
from matplotlib import pyplot as plt

from scipy.integrate import solve_ivp
import numpy as np
from scipy.integrate import simps as simps
from numpy.linalg import inv
import time

from scipy.interpolate import interp1d
from scipy.constants import c
H_0 =  73.48



def integrador(sistema_ec, cond_iniciales, parametros_modelo, z_inicial=0, z_final=3):    
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
    
    '''Dado un gamma y una solución de las variables dinamicas, grafica estas
    por separado en una figura de 4x4.'''

    f, axes = plt.subplots(2,3)
    ax = axes.flatten()
    
    color = ['b','r','g','y','k']
    y_label = ['x','y','v','w','r']
    [ax[i].plot(solucion.t,solucion.y[i],color[i]) for i in range(5)]
    [ax[i].set_ylabel(y_label[i],fontsize='medium') for i in range(5)];
    [ax[i].set_xlabel('z (redshift)',fontsize='medium') for i in range(5)];
    [ax[i].invert_xaxis() for i in range(5)]; #Doy vuelta los ejes
    plt.show()
    

def leer_data(archivo):
    # leo la tabla de datos:
    zcmb,zhel,dz,mb,dmb=np.loadtxt(archivo, usecols=(1,2,3,4,5),unpack=True)
    
    #creamos la matriz diagonal con los errores de mB. ojo! esto depende de alfa y beta:
    Dstat=np.diag(dmb**2.)
    # hay que leer la matriz de los errores sistematicos que es de NxN
    sn=len(zcmb)
    Csys=np.loadtxt('lcparam_full_long_sys.txt',unpack=True)
    Csys=Csys.reshape(sn,sn)
    #armamos la matriz de cov final y la invertimos:
    Ccov=Csys+Dstat
    Cinv=inv(Ccov)

    return zcmb,zhel, Cinv, mb



def magn_aparente_teorica(z,E,zhel,zcmb):
    '''Parte teorica a partir de mi modelo de H(z), para cada sn voy a hacer
    la integral correspondiente: Calculamos la distancia luminosa 
    d_L =  (c/H_0) int(dz'/E(z'))'''
    
    d_c=np.zeros(len(E)) #Distancia comovil
    for i in range (1, len(E)):
        d_c[i] = 0.001*(c/H_0) * simps(1/E[:i],z[:i])
        
    dc_int = interp1d(z,d_c) #Interpolamos 
    
    d_L = (1+zhel) * dc_int(zcmb) #Obs, Caro multiplica por Zhel, con Zobs da un poquin mejor
    
    ##magnitud aparente teorica
    muth = 25 + 5 * np.log10(d_L) #base cosmico
    return muth



def chi_2(muth,magn_aparente_obs,M_abs,C_invertida):
    sn = len(muth)
    muobs =  magn_aparente_obs - M_abs
    deltamu = muobs - muth
    transp = np.transpose(deltamu)
    aux = np.dot(C_invertida,deltamu)
    chi2 = np.dot(transp,aux)/sn
    return chi2
