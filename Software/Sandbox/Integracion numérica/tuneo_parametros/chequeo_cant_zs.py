import numpy as np
import math
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz as cumtrapz
from scipy.constants import c as c_luz #metros/segundos
c_luz_km = c_luz/1000

import sys
import os
from os.path import join as osjoin
from pc_path import definir_path
path_git, path_datos_global = definir_path()
os.chdir(path_git)
sys.path.append('./Software/Funcionales/')
from funciones_int import integrador

#%%
omega_m = 0.24
b = 1
H0 = 73.48
params_fisicos = [omega_m,b,H0]

cantidad_zs = np.logspace(2,6,50,dtype=int)
cantidad_zs
hs = []
ds = []
z_ref = np.linspace(0,3,50) #Evaluoy analizo de 0 a 3 (la zona de interes!)

for cant in cantidad_zs:
    zs, H_ode = integrador(params_fisicos, n=1, cantidad_zs=cant,
                max_step=0.003)
    d_c =  c_luz_km * cumtrapz(H_ode**(-1), zs, initial=0)
    g=interp1d(zs,H_ode)
    f=interp1d(zs,d_c)
    hs.append(g(z_ref))
    ds.append(f(z_ref))

final_ds = np.zeros(len(ds))
final_hs = np.zeros(len(hs))
for j in range(1,len(ds)):
    aux_hs = np.mean(hs[j]-hs[j-1]);
    aux_ds = np.mean(ds[j]-ds[j-1]);
    #aux = np.mean((1-ds[j]/ds[j-1]));
    final_hs[j]=aux_hs;
    final_ds[j]=aux_ds;

#%%
%matplotlib qt5
plt.close()
plt.figure()
plt.grid(True)
plt.xlabel('max_steps')
plt.ylabel('$\Delta$hs')
plt.plot(cantidad_zs[::-1],np.array(final_hs)[::-1],'.-');
#plt.legend(loc='best')
plt.show()

tol = 10**(-4)
final_hs[np.where(np.abs(final_hs)<=tol)[0]]
cantidad_zs[np.where(np.abs(final_hs)<=tol)[0]]

#Pareceria que en mas_steps y valores de los parametros razonables
#no hay diferencia en el valor de H(z) para distintos cant_zs

#%%
%matplotlib qt5
plt.close()
plt.figure()
plt.grid(True)
plt.xlabel('max_steps')
plt.ylabel('$\Delta$ds')
plt.plot(cantidad_zs[::-1],np.array(final_ds)[::-1],'.-');
#plt.legend(loc='best')
plt.show()

tol = 10**(-4)
#final_ds[np.where(np.abs(final_ds)<=tol)[0]]
cantidad_zs[np.where(np.abs(final_ds)<=tol)[0]]
