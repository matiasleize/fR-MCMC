"""
Created on Wed Feb  5 13:04:17 2020

@author: matias
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize


import sys
import os
from os.path import join as osjoin
from pc_path import definir_path
path_git, path_datos_global = definir_path()
os.chdir(path_git)
sys.path.append('./Software/utils/')
from funciones_data import leer_data_pantheon
from funciones_LambdaCDM_1 import params_to_chi2
#%%

min_z = 0
max_z = 3

os.chdir(path_git+'/Software/Estadística/Datos/Datos_pantheon/')
zcmb,zhel, Cinv, mb = leer_data_pantheon('lcparam_full_long_zhel.txt'
                        ,min_z=min_z,max_z=max_z)
sn = len(zcmb)
Cinv.shape


#%%
#Parametros a ajustar
name = 'data_9'
H0_true =  73.5 #Unidades de (km/seg)/Mpc
M_true = -19.22

# Calculo el chi 2
omegas = np.linspace(0,1,1000)
chies = np.zeros((len(omegas)))
for i,omega_m in enumerate(omegas):
    theta = omega_m
    chies[i] = params_to_chi2(theta,[M_true,H0_true], zcmb, zhel,
                Cinv, mb) #Chis normalizados

os.chdir(path_datos_global+'/Entrega/Supernovas/Valores_chies')
file2write=open(name,'w+')
for i in range(len(chies)):
    file2write.write( '{} \t {}\n'.format(omegas[i], chies[i]))
file2write.close()



#%%
mins = np.where(chies==np.min(chies))
print(chies[mins[0][0]], omegas[mins[0][0]])

chis=chies
#%% Para ver cuanto es un sigma, me fijo cuales valores de chi estan a +- 1.07
delta_chi = chies-min(chies)
index_min = np.where(chies==min(chies))[0][0]
index_sigma_Om = np.where(delta_chi<1.07)[0][0] #1 corresponde a 1 sigma
index_sigma_om = np.where(delta_chi<1.07)[0][-1]

omega_posta = omegas[index_min]

# OJOTA, chequear que este bien esto que estoy reportando
print('Omega es {} +{} -{}'.format(omega_posta,abs(omega_posta-omegas[index_sigma_Om]),abs(omega_posta-omegas[index_sigma_om])))

# La de Mati: 0.9964890707326334 0.38538538538538536
# La de Caro: 0.9964890707326336 0.38538538538538536

#%% Calculo el posterior
log_likelihood = -0.5 * chies
log_prior=np.zeros((len(log_likelihood)))
for i in range(len(omegas)):
    if (0 < omegas[i] < 1):
        log_prior[i] = 0
    else:
        log_prior[i] = -np.inf
log_posterior = log_likelihood + log_prior
distribution = np.exp(log_posterior)
np.mean(distribution)
np.quantile(distribution, 0.2)
#%%
%matplotlib qt5
plt.close()
plt.figure()
plt.plot(omegas,np.exp(log_posterior),'.');
plt.show()
