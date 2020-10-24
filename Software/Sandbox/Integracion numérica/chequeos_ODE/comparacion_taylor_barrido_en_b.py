import numpy as np
from matplotlib import pyplot as plt

import sys
import os
from os.path import join as osjoin
from pc_path import definir_path
path_git, path_datos_global = definir_path()
os.chdir(path_git)
sys.path.append('./Software/Funcionales/')
from funciones_int import integrador
from funciones_taylor import Taylor_HS, Taylor_ST

#%%
bs=np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 2])
bs=np.array([0.08,0.1, 0.2, 0.3, 0.4, 0.5, 1, 2])
#bs=np.linspace(0.01,2,100)
error_b=np.zeros(len(bs))
omega_m=0.24
H0=73.48


#%% Hu-Sawicki n=2
for i, b in enumerate(bs):
    print(i)
    params_fisicos = [omega_m,b,H0]

    zs, H_ode = integrador(params_fisicos, cantidad_zs=int(10**5), max_step=10**(-5),model='HS',n=2)
    H_taylor = Taylor_ST(zs,omega_m,b,H0)

    error_b[i] = 100*np.abs(np.mean(1-(H_taylor/H_ode)))
#Tarda 1h y 20 aprox
#%%
%matplotlib qt5
plt.close()
plt.figure()
plt.title('Error Porcentual para Hu-Sawicki n=2')
plt.yscale('log')
plt.grid(True)
plt.xlabel('b')
plt.ylabel('Error Porcentual')
plt.plot(bs,error_b,label='Error en b')
plt.legend(loc='best')
plt.show()
error_b

#%%
#b=0.5
#params_fisicos = [omega_m,b,H0]
#zs, H_ode = integrador(params_fisicos, cantidad_zs=int(10**3), max_step=60**(-5),model='ST',n=1)
#H_taylor = Taylor_ST(zs,omega_m,b,H0)

#plt.plot(zs,H_ode/H0)
#plt.plot(zs,H_taylor/H0)
#plt.plot(zs,H_ode/H_taylor)
#%%

#%% Starobinsky n=1
for i, b in enumerate(bs):
    print(i)
    params_fisicos = [omega_m,b,H0]

    zs, H_ode = integrador(params_fisicos, cantidad_zs=int(10**5), max_step=10**(-5),model='ST')
    H_taylor = Taylor_ST(zs,omega_m,b,H0)

    error_b[i] = 100*np.abs(np.mean(1-(H_taylor/H_ode)))
#Tarda 1h y 15 aprox
#%%
%matplotlib qt5
plt.close()
plt.figure()
plt.title('Error Porcentual para Starobinsky n=1')
plt.yscale('log')
plt.grid(True)
plt.xlabel('b')
plt.ylabel('Error Porcentual')
plt.plot(bs,error_b,label='Error en b')
plt.legend(loc='best')
plt.show()
error_b

#%% Hu-Sawicki n=1
for i, b in enumerate(bs):
    print(i)
    params_fisicos = [omega_m,b,H0]

    zs, H_ode = integrador(params_fisicos, cantidad_zs=int(10**5), max_step=0.0005)
    H_taylor = Taylor_HS(zs,omega_m,b,H0)

    error_b[i] = 100*np.abs(np.mean(1-(H_taylor/H_ode)))
#Conclusion: Para HS n=1, max_step mayor a 0.001 lleva a resultados erroneos!.
#%%
%matplotlib qt5
plt.close()
plt.figure()
plt.title('Error Porcentual para Hu-Sawicki n=1')
plt.yscale('log')
plt.grid(True)
plt.xlabel('b')
plt.ylabel('Error Porcentual')
plt.plot(bs,error_b,label='Error en b')
plt.legend(loc='best')
plt.show()
error_b
