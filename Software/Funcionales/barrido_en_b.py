"""
Created on Sun Feb  2 13:28:48 2020

@author: matias
"""
import numpy as np
from scipy.interpolate import interp1d
from scipy.constants import c as c_luz #metros/segundos
c_luz_km = c_luz/1000

import sys
import os
from os.path import join as osjoin
from pc_path import definir_path
path_git, path_datos_global = definir_path()
os.chdir(path_git)
sys.path.append('./Software/Funcionales/')
from funciones_int import Hubble_teorico
from funciones_supernovas import magn_aparente_teorica, chi2_supernovas
from funciones_BAO import r_drag, Hs_to_Ds, Ds_to_obs_final
from funciones_AGN import zs_2_logDlH0
from funciones_alternativos import params_to_chi2
from matplotlib import pyplot as plt

os.chdir(path_git)
sys.path.append('./Software/Funcionales/')
from funciones_data import leer_data_pantheon, leer_data_cronometros, leer_data_BAO, leer_data_AGN

# Supernovas
os.chdir(path_git+'/Software/Estadística/Datos/Datos_pantheon/')
ds_SN = leer_data_pantheon('lcparam_full_long_zhel.txt')

# Cronómetros
os.chdir(path_git+'/Software/Estadística/Datos/')
ds_CC = leer_data_cronometros('datos_cronometros.txt')

# BAO
os.chdir(path_git+'/Software/Estadística/Datos/BAO/')
ds_BAO = []
archivos_BAO = ['datos_BAO_da.txt','datos_BAO_dh.txt','datos_BAO_dm.txt',
                'datos_BAO_dv.txt','datos_BAO_H.txt']
for i in range(5):
    aux = leer_data_BAO(archivos_BAO[i])
    ds_BAO.append(aux)

# AGN
os.chdir(path_git+'/Software/Estadística/Datos/Datos_AGN')
ds_AGN = leer_data_AGN('table3.dat')

#%%
bs = np.linspace(0.1,4,10)
Hs = np.arange(67,73)
chies = np.zeros((len(bs),len(Hs)))
omega_m=0.24
int=0
for (i,b) in enumerate(bs):
    print(i,b)
    for (j,H0) in enumerate(Hs):
        chies[i,j] = params_to_chi2([-19.41, omega_m, b, H0], _, index=4,
                    dataset_SN = ds_SN,
                    dataset_CC = ds_CC,
                    #dataset_BAO = ds_BAO,
                    #dataset_AGN = ds_AGN,
                    #H0_Riess = True,
                    model = 'HS',
                    integrador=int
                    )

#1077.8293845284927/(1048+20+len(ds_CC[0])-4)
#%%
%matplotlib qt5
plt.figure()
plt.title('HS: CC+SN, omega_m=0.24, M=-19.41')
plt.grid(True)
for k in range(0,6):
#for k in range(7,len(chies[0,:])):
#for k in range(len(chies[0,:])):
    plt.plot(bs,chies[:,k],label='H0={}'.format(Hs[k]))
plt.ylabel(r'$\chi^2$')
plt.xlabel('b')
plt.legend()
plt.savefig('/home/matias/integrador/Barrido_b_H0_bajos.png')
#%%
bs = np.linspace(0.1,4,20)
chies = np.zeros((len(bs),3))
for j in range(3):
    for (i,b) in enumerate(bs):
        print(j)
        print(i,b)
        chies[i,j] = params_to_chi2([-19.41, 0.352, b, 62], _, index=4,
                    dataset_SN = ds_SN,
                    dataset_CC = ds_CC,
                    #dataset_BAO = ds_BAO,
                    #dataset_AGN = ds_AGN,
                    #H0_Riess = True,
                    model = 'HS',
                    integrador=j
                    )
#%%
metodo_0=chies[:,0]
metodo_1=chies[:,1]
metodo_2=chies[:,2]

error_01 =np.abs((metodo_0-metodo_1)/metodo_0)*100
error_12 =np.abs((metodo_1-metodo_2)/metodo_1)*100

plt.figure()
plt.title('HS: CC+SN, omega_m=0.352, M=-19.41')
plt.grid(True)
plt.plot(bs,error_01,label='Error_01')
#plt.plot(bs,error_12,label='Error_12')
plt.ylabel('Error porcentual')
plt.xlabel('b')
plt.legend()
plt.savefig('/home/matias/integrador/Error_en_b.png')
