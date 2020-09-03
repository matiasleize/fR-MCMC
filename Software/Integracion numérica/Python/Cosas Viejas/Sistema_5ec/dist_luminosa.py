#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 17:48:22 2019

@author: matias
"""    

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import simps as simps,quad
from scipy.interpolate import interp1d


npzfile = np.load('/home/matias/Documents/Tesis/Software/Integracion numérica/Python/Sistema_5ec/H(z).npz')
zs = npzfile['zs']
hubbles = npzfile['hubbles']

#Calculamos la distancia luminosa d_L =  (c/H_0) int(dz'/E(z'))

from scipy.constants import c
H_0 =  74.2


d_c=np.zeros(len(hubbles)) #Distancia comovil

for i in range (1,len(hubbles)):
    d_c[i] = 0.001*(c/H_0) * simps(1/hubbles[:i],zs[:i])

d_L = (1+zs) * d_c


#Guardamos la distancia luminosa
#np.savez('/home/matias/Documents/Tesis/Software/Integracion numérica/Python/Sistema_5ec/d_L(z)', zs=zs, d_L=d_L)

#%%Otra forma
#f = interp1d(zs,1/hubbles) 
#
#for i in range (1,len(hubbles)):
#    d_c = (c/H_0) * (quad(f, 0, zs[i])[0])
#d_L2 = (1+zs) * d_c


#%%
plt.plot(zs,d_L, label= '$H_{0}=74.2$' )
#plt.plot(zs,d_L2, label= '$H_{0}=74.2$' )
#plt.plot(zs,mu, label='gamma = {}'.format(gamma))

plt.title('Distancia luminosa')
plt.xlabel('z(redshift)')
plt.ylabel(r'$d_L$')
plt.legend(loc='best')
plt.grid(True)
#%%plt.close()
#mahnitud aparent
muth = 25+5*np.log10(d_L) #base cosmico
plt.plot(zs,mu)
zcmb,zhel,dz,mb,dmb=np.loadtxt('lcparam_full_long_zhel.txt', usecols=(1,2,3,4,5),unpack=True)
#plt.plot(zcmb,mb+19.2)
#%%
#f = interp1d(zs,mu) 
#resta = f(zcmb)-mb-19.2
#plt.plot(zcmb,resta,'.')
#%%
Mabs=-19.3
muobs=mb-Mabs
deltamu=muobs-muth
#mp.plot(zcmb, muobs , 'ro')
#mp.savefig('prueba.jpg')
#calculamos el chi2
transp=np.transpose(deltamu)
chi2=np.dot(np.dot(transp,Cinv),deltamu)
print(chi2,chi2/sn)
#graficos
mp.errorbar(zcmb,muobs,yerr=dmb, fmt='.',label='observado')
mp.plot(zcmb,muth,'r.',label='teorico')
mp.savefig('comparacion7348.pdf')
mp.clf()
mp.plot(zcmb,(muth-muobs)/muobs,'r.')
mp.savefig('diferencias7348.pdf')
