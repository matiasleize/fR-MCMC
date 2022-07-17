"""
Created on Sun Feb  2 13:28:48 2020

@author: matias
"""
import time
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import cumtrapz as cumtrapz
from scipy.integrate import solve_ivp
from scipy.constants import c as c_luz #meters/seconds
c_luz_norm = c_luz/1000;

import sys
import os
from os.path import join as osjoin
from pc_path import definir_path
path_git, path_datos_global = definir_path()
os.chdir(path_git)
sys.path.append('./Software/utils/')
from condiciones_iniciales import condiciones_iniciales
from cambio_parametros import params_fisicos_to_modelo
from int import dX_dz, integrador
#%%
def plot_sol(solucion):
    '''Grafica la solución de la ODE de 5x5, por separado
    en distintos subplots.'''

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
sistema_ec = dX_dz
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
print('Duration: {} minutes and {} seconds'.format(int((t2-t1)/60),
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

#%% Ploteo y comparo las soluciones de HS y LCDM
#%matplotlib qt5
plt.close()
plt.figure()
plt.grid(True)
plt.xlabel('z (redshift)')
plt.ylabel('H(z)')
plt.plot(z_int,E_ODE,label='ODE')
plt.plot(z_int,E_LCDM,'-.',label='LCDM')
plt.legend(loc='best')
plt.xlim([0,3])
plt.ylim([0,6])

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
