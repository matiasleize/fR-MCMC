import numpy as np
from matplotlib import pyplot as plt

import sys
import os
from os.path import join as osjoin
from pc_path import definir_path
path_git, path_datos_global = definir_path()
os.chdir(path_git)
sys.path.append('./Software/utils/')
from int import integrator
from taylor import Taylor_HS,Taylor_ST

#%%
bs=np.linspace(0.01,0.5,500)
#bs=np.array([0.01, 0.02, 0.05, 0.1, 0.12, 0.15, 0.2, 0.3, 0.4, 0.5, 1, 2]) #HS

error_b = np.zeros(len(bs))
omega_m = 0.3
H0 = 73.48


#%% Hu-Sawicki n=2
for i, b in enumerate(bs):
    print(i)
    params_fisicos = [omega_m,b,H0]

    zs, H_ode = integrator(params_fisicos, cantidad_zs=int(10**5), max_step=10**(-5), model='HS')
    H_taylor = Taylor_HS(zs,omega_m,b,H0)

    error_b[i] = 100*np.abs(np.mean(1-(H_taylor/H_ode)))

#%%
%matplotlib qt5
plt.close()
plt.figure()
plt.grid(True)
plt.title('Error Porcentual para Hu-Sawicki n=1',fontsize=15)

plt.xlabel('b',fontsize=13)
plt.ylabel('Error Porcentual',fontsize=13)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.yscale('log')

plt.plot(bs,error_b,label='Error en b')
plt.legend(loc='best',prop={'size': 12})
plt.show()
#plt.savefig('/home/matias/Error_porcentual_HS.png')
