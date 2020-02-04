#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 23:03:17 2019

@author: matias
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate import simps as simps
import time

#%%
#Gamma de Lucila
gamma = lambda r,b,c,d,n: ((1+d*r**n) * (-b*n*r**n + r*(1+d*r**n)**2)) / (b*n*r**n * (1-n+d*(1+n)*r**n))  

#Gamma de De la Cruz (Esta mal!)
#gamma = lambda r,b,c,d,n: -((1+d*r**n) * (-b*n*c*r**n + r*(1+d*r**n)**2)) / (b*n*c*r**n * (n-1 + (1+n)*d*r**n))  


#Segun el paper b =200
#b = 200
#d = 200/19
#c = 6*(1-0.24)*(d/b)
#r_0 = 41 * c * 74.2**2
#n = 1

#Segun el paper c=0.24
b = 1
d = 1/19 #(valor del paper)
c = 0.24
r_0 = 41
n = 1


##Tesis de Caro
##omega_m = 0.3089
#b = 1.25*10**(-3)
#d = 6.56*10**(-5)
#c = 0.24 #c = 6*(1-omega_m)*d/b
#r_0 = 41
#n = 1




def dX_dz(z, variables): 

    x = variables[0]
    y = variables[1]
    v = variables[2]
    w = variables[3]
    r = variables[4]
    
    G = gamma(r,b,c,d,n)
    
    s0 = (-w + x**2 + (1+v)*x - 2*v + 4*y) / (z+1)
    s1 = - (v*x*G - x*y + 4*y - 2*y*v) / (z+1)
    s2 = -v * (x*G + 4 - 2*v) / (z+1)
    s3 = w * (-1 + x+ 2*v) / (z+1)
    s4 = -x*r*G/(1+z)
        
    return [s0,s1,s2,s3,s4]

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
    

#%%
##Coindiciones iniciales e intervalo
x_0 = -0.339
y_0 = 1.246
v_0 = 1.64

w_0 = 1+x_0+y_0-v_0 

ci = [x_0, y_0, v_0, w_0, r_0] #Condiciones iniciales
zi = 0
zf = 3 # Es un valor razonable con las SN 1A
#%% Resolvemos y graficamos
#plt.close('all')

sol = solve_ivp(dX_dz, [zi,zf], ci, max_step=0.01)
#plot_sol(sol)   


# Guardamos z y v(z)
np.savez('/home/matias/Documents/tesis_licenciatura/Software/Integracion numérica/Python/Sistema_5ec/v(z)'
         , zs=sol.t, v=sol.y[2])

#%% Integramos el vector v y calculamos el Hubble
zs = np.linspace(0,3,7000)
hubbles = np.zeros(len(zs))

lala = np.zeros(len(zs))
plt.close()

t1 = time.time()
for i in range(len(zs)):
    zi = zs[0]
    zf = zs[i]
    sol = solve_ivp(dX_dz, [zi,zf], ci, max_step=0.05)      # 0.005

    int_v = simps((sol.y[2])/(1+sol.t),sol.t) 
    lala[i] = int_v
    hubbles[i]=(1+zf)**2 * np.e**(-int_v) # integro desde 0 a z, ya arreglado
t2 = time.time()

print('Duración {} minutos y {} segundos'.format(int((t2-t1)/60),
      int((t2-t1) - 60*int((t2-t1)/60))))

plt.plot(zs,hubbles,label=r'$\Gamma=\Gamma(r)$')
plt.title('Parámetro de Hubble')
plt.xlabel('z(redshift)')
plt.ylabel(r'$H(z)/H_{0}$')

plt.legend(loc='best')
plt.grid(True)

# Guardamos z y H()
np.savez('/home/matias/Documents/tesis_licenciatura/Software/Integracion numérica/Python/Sistema_5ec/H(z)'
         , zs=zs, hubbles=hubbles)

#%% Chequeo
cte = sol.y[3]+sol.y[2]-sol.y[1]-sol.y[0]

plt.plot(sol.t,cte)