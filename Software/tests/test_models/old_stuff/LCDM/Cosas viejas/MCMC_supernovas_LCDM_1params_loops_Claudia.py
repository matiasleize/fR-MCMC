"""
Created on Wed Feb  5 13:04:17 2020

@author: matias
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize
import  matplotlib.pyplot as mp
from scipy import optimize
import scipy
import math

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
H0_true =  73.5 #Unidades de (km/seg)/Mpc
M_true = -19.22

# Calculo el chi 2
omegas = np.linspace(0,1,100)
chies = np.zeros((len(omegas)))
for i,omega_m in enumerate(omegas):
    theta = omega_m
    chies[i] = params_to_chi2(theta,[M_true,H0_true], zcmb, zhel,
                Cinv, mb) #Chis normalizados

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

#%%
# OJOTA, chequear que este bien esto que estoy reportando
print('Omega es {} +{} -{}'.format(omega_posta,abs(omega_posta-omegas[index_sigma_Om]),abs(omega_posta-omegas[index_sigma_om])))

#%% ACA EMPIEZA LO DE CLAUDIA
#graficos
plt.close()
plt.figure()
plt.plot(omegas,chies,'r.')
#mp.savefig('Mabsvschi27348.pdf')
#%%
Likelihood = np.exp(-0.5*(chies-chies.min()))
#print(Likelihood)
Maximo_Like = Likelihood.max()   #max de la likelihood (normalizada a 1)

#Pruebo distintos niveles (cortes en altura para calcular los errores):
Level = Maximo_Like - 0.318
#Level_1 = Maximo_Like - 0.1
#Level_2 = Maximo_Like - 0.2
#Level_3 = Maximo_Like - 0.3
#Level_4 = Maximo_Like - 0.4
#Level_412 = Maximo_Like - 0.412


#Pasos:
# elijo un nivel en el eje y (Likelihood)
# busco los valores de x que correspondan a ese likelihood.
# Integro entre esos dos valores
# Hago el cociente con la integral total
# Cuando llego al 68% del valor total, ya encontré el intervalo a 1sigma.

integral_total= scipy.integrate.trapz(Likelihood,omegas)
y = Level*np.ones(100)    #linea horizontal con un nivel de Likelihood

#Calculo los puntos del eje x donde la Likelihood corta la horizontal:
idx = np.argwhere(np.diff(np.sign(y - Likelihood))).flatten()

#Los indices son:
indice_1= idx[0]
indice_2= idx[1]

print(indice_1, indice_2)

#Calculo las integrales acumulativas hasta los puntos x_1 y x_2:
integral_1 = scipy.integrate.cumtrapz(Likelihood,omegas)[indice_1]
integral_2 = scipy.integrate.cumtrapz(Likelihood,omegas)[indice_2]

#Restando las integrales acumulativas, calculo la integral en el rango:
integral_rango = integral_2 - integral_1

#Hago el cociente con la integral total:
cociente= integral_rango/integral_total

#Me fijo para que Level obtengo que el area sea el 68% del total:
print(cociente*100.)

#Me fijo cuales son los valores de Mabs:
print(omegas[indice_1])
print(omegas[indice_2])

#Me fijo cual es el indice del maximo Likelihood:
# Get the indices of maximum element in numpy array
idx_max = np.where(Likelihood == np.amax(Likelihood))

print(omegas[idx_max])

#Calculo los errores inferiores y superiores:
error_1 = omegas[idx_max] - omegas[indice_1] #error inferior
error_2 = omegas[indice_2] - omegas[idx_max] #error superior

print(error_1)
print(error_2)
print(omegas[indice_2] , omegas[idx_max] , omegas[indice_1])

#plt.close()
plt.figure()
plt.plot(omegas,Likelihood,'r.')
#plt.savefig('Lik#e_Mabs.pdf')
#%%

#efectivamente da una parabola (el grafico lo hace mas abajo con el ajuste ya hecho)
#ajustamos un polinomio de grado dos a la parabola y buscamos el minimo
fit=np.polyfit(omegas,chies,2) #esto me devuelve un array con a,b,c
pol=np.poly1d(fit) #esto me lo convierte para poder evaluar el pol
Mmin = optimize.fmin(pol, -19)

#Mmin[0] es el minimo en Mabs. y pol(Mmin[0]) el valor del chi2 en el minimo
#ahora hay que encontrar el error. hay que sumarle 1.00 al chi2 de acuerdo a la tabla de la pg 815 del numerical recipies.
chi2sigma=pol(Mmin[0])+1.00
#ahora hay que encontrar el Mabs asociado a este valor de chi2. depejo la cuadratica considerando c=c-chi2sigma
Mabssigma=(-fit[1]+math.sqrt((fit[1]**2.-4.0*fit[0]*(fit[2]-chi2sigma))))/(2.0*fit[0])
print(Mmin[0],abs(Mabssigma-Mmin[0]),pol(Mmin[0])) #escribo el minimo, la desviacion a 1 sigma  y el chi2 minimo

#ploteo el el chi2 y el fit
#Claudia: xp=np.linspace(-19.1,-19.7,100)
#xp=np.linspace(-19.28,-19.22,100)
#xp=np.linspace(-19.1,-19.7,100)
xp=omegas
#plt.close()
plt.figure()
plt.plot(omegas,chies,'r.',xp,pol(xp),'g.')
#plt.savefig('MabsvschiLCDM.pdf')
