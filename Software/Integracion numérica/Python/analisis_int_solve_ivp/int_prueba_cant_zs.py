"""
Created on Sun Feb  2 13:28:48 2020

@author: matias
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate import simps as simps
from scipy.integrate import cumtrapz as cumtrapz
from scipy.interpolate import interp1d
import time

from scipy.interpolate import interp1d
from scipy.constants import c as c_luz #metros/segundos
c_luz_norm = c_luz/1000

def params_fisicos_to_modelo(omega_m, b, n=1):
    '''Toma los parametros fisicos (omega_m, el parametro de distorsión b)
    y devuelve los parametros del modelo c1 y c2'''
    c_luz_norm = c_luz/1000 #km/seg
    alpha = 1 / (8315)**2
    beta = 1 / 0.13
    aux = ((100/c_luz_norm)**2 * 6 * (1 - omega_m))  / (alpha * omega_m * beta)
    c_2 =  (2/(aux * b))
    c_1 =  2/b
    return c_1, c_2


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

    [B,D,N] = params_modelo
    gamma = lambda r,b,d: ((1+d*r) * (-b*r + r*(1+d*r)**2)) / (b*2*d*r**2)
    G = gamma(r,B,D)

#    gamma = lambda r,b,d,n: ((1+d*r**n) * (-b*n*r**n + r*(1+d*r**n)**2)) / (b*n*(1 - n + d * (1+n) * r**n)*r**n)
#    G = gamma(r,B,D,N)

    s0 = (-w + x**2 + (1+v)*x - 2*v + 4*y) / (z+1)
    s1 = (- (v*x*G - x*y + 4*y - 2*y*v)) / (z+1)
    s2 = (-v * (x*G + 4 - 2*v)) / (z+1)
    s3 = (w * (-1 + x + 2*v)) / (z+1)
    s4 = (-x * r * G) / (1+z)
    return [s0,s1,s2,s3,s4]
#%%
sistema_ec=dX_dz
z_inicial = 0
z_final = 3

max_step = np.inf
verbose = True

x_0 = -0.339
y_0 = 1.246
v_0 = 1.64
w_0 = 1 + x_0 + y_0 - v_0
r_0 = 41
ci = [x_0, y_0, v_0, w_0, r_0] #Condiciones iniciales
cond_iniciales = ci

#%%
omega_m = 0.3
b = 1
H0 = 73.48
#Parametros problematicos
c1,c2 = params_fisicos_to_modelo(omega_m,b)
print(c1,c2)
params_modelo=[c1, c2, 1]
#params_modelo = [1,1/19,1]
zs_patron  = np.linspace(z_inicial,z_final,10000)
cantidad_zs = np.linspace(10,10000,5000)
cant_zs = np.array([int(i) for i in cantidad_zs])
print(cant_zs)
Hs = []
for i,cant in enumerate(cant_zs):
    # Integramos el vector v y calculamos el Hubble
    zs = np.linspace(z_inicial,z_final,cant)
    print(zs)
    sol = solve_ivp(sistema_ec, (z_inicial,z_final),cond_iniciales, t_eval=zs,
                    args=params_modelo)#,max_step=0.001)
    int_v =  cumtrapz(sol.y[2]/(1+sol.t),sol.t,initial=0)
    E = (1+sol.t)**2 * np.exp(-int_v)
    f = interp1d(zs,H0*E)
    Hs.append(f(zs_patron))

final = []
for j in range(1,len(Hs)):
    aux = np.mean((Hs[j]-Hs[j-1])/Hs[j]);
    final.append(aux);
#%%
%matplotlib qt5
plt.close()
plt.figure()
plt.grid(True)
plt.xlabel('cantidad_zs')
plt.ylabel('%Hs')
plt.plot(cant_zs[1:],np.array(final)*100,'.');
#plt.legend(loc='best')
plt.show()
