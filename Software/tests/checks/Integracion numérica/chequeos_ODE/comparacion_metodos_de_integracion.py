import numpy as np
from matplotlib import pyplot as plt

import sys
import os
from os.path import join as osjoin
from pc_path import definir_path
path_git, path_datos_global = definir_path()
os.chdir(path_git)
sys.path.append('./Software/utils/')
from int import integrador
from taylor import Taylor_HS

b = 0.15
omega_m = 0.3
H0 = 73.48
params_fisicos=[omega_m,b,H0]
zs, H_RK45 = integrador(params_fisicos, cantidad_zs=int(10**5),
                        max_step=10**(-5), model='HS',method='RK45')
_, H_LSODA = integrador(params_fisicos, cantidad_zs=int(10**5),
                        max_step=10**(-5), model='HS',method='LSODA')
error = 100*np.abs(1-(H_LSODA/H_RK45))

#%%
%matplotlib qt5
plt.close()
plt.figure()
plt.grid(True)
plt.title('Error Porcentual para Hu-Sawicki',fontsize=15)

plt.xlabel('z',fontsize=13)
plt.ylabel('Error Porcentual',fontsize=13)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.yscale('log')

plt.plot(zs,error,label='Error por el metodo')
plt.legend(loc='best',prop={'size': 12})
plt.show()
plt.savefig('/home/matias/Error_porcentual_metodo_HS.png')
