#en este programa vamos a calcular el chi2 MOG-sn considerando todos los parametros fijos. Solo hay que prestarle atencion al valor de H0.
import numpy as np
from numpy.linalg import inv
import  matplotlib.pyplot as mp
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import simps
from scipy.constants import c
H_0 =  74.2 #La de wikipedia de Planck (Vieja tradicional)
#H_0 =  67.4 #La que usaba Lucila
#H_0 =  54.4 #Nueva contante desde Planck, para usarla tenemos que usar la D_L de universo cerrado!
# z no tiene que empezar justo en 0!, empezarlo de 0.00001

#%%
#leo la tabla de la funcion que tengo que integrar que viene del mathematica (donde tambien use H0)
npzfile = np.load('/home/matias/Documents/Tesis/Software/Integracion numérica/Python/Sistema_5ec/H(z).npz')
z = npzfile['zs']
E = npzfile['hubbles']

H_int = interp1d(z,E) 
#%%
# leo la tabla de datos:
zcmb,zhel,dz,mb,dmb=np.loadtxt('lcparam_full_long_zhel.txt', usecols=(1,2,3,4,5),unpack=True)
#%% Esto esta bien
#creamos la matriz diagonal con los errores de mB. ojo! esto depende de alfa y beta:
Dstat=np.diag(dmb**2.)
# hay que leer la matriz de los errores sistematicos que es de NxN
sn=len(zcmb)
Csys=np.loadtxt('lcparam_full_long_sys.txt',unpack=True)
Csys=Csys.reshape(sn,sn)
#armamos la matriz de cov final y la invertimos:
Ccov=Csys+Dstat
Cinv=inv(Ccov)

#longitud de la tabla de sn:
muth_c=np.zeros(sn)
muth_m=np.zeros(sn)
#%%
#Parte teorica a partir de mi modelo de H(z)
#para cada sn voy a hacer la integral correspondiente:
#Calculamos la distancia luminosa d_L =  (c/H_0) int(dz'/E(z'))

d_c=np.zeros(len(E)) #Distancia comovil
for i in range (1, len(E)):
    d_c[i] = 0.001*(c/H_0) * simps(1/E[:i],z[:i])

dc_int = interp1d(z,d_c) #Interpolamos  
#%% Especificamos la dist luminosa en los z experimentales 
d_L_full = d_c * (1+z)
d_L = (1+zhel) * dc_int(zcmb) #Obs, Caro multiplica por Zhel, con Zobs da un poquin mejor

plt.figure()
plt.plot(z,d_L_full,'o')
plt.plot(zcmb,d_L,'.',label='$z_{cmb}$')

plt.title('Distancia luminosa')
plt.xlabel('z(redshift)')
plt.ylabel(r'$d_L$')
plt.legend(loc='best')
plt.grid(True)
#%%plt.close()
#mahnitud aparente teorica
muth = 25+5*np.log10(d_L) #base cosmico

#Magnitud aparente experimental
Mabs=-19.3
muobs=mb-Mabs

plt.figure()
plt.plot(zcmb,muth,'o',label= r'$\mu_{th}$')
plt.plot(zcmb,muobs,'.',label= r'$\mu_{exp}$')
plt.title('$\mu$ teórico y expermiental')
plt.xlabel('z(redshift)')
plt.ylabel(r'$d_L$')
plt.legend(loc='best')
plt.grid(True)
#%%
deltamu=muobs-muth

plt.figure()
plt.plot(zcmb,deltamu,'.',label= r'$\Delta\mu$')
plt.title('Diferencias en la magnitud aparente')
plt.xlabel('z(redshift)')
plt.ylabel(r'$\Delta\mu$')
plt.legend(loc='best')
plt.grid(True)

#Se observan que las mayores diferencias se dan para z chicos!

#%% calculamos el chi2
transp=np.transpose(deltamu)
aux = np.dot(Cinv,deltamu)
chi2=np.dot(transp,aux)
print(chi2,chi2/sn)
#%%
#graficos
mp.errorbar(zcmb,muobs,yerr=dmb, fmt='.',label='observado')
mp.plot(zcmb,muth,'r.',label='teorico')
mp.savefig('comparacion7348.pdf')
mp.clf()
mp.plot(zcmb,(muth-muobs)/muobs,'r.') #Error relativo
mp.savefig('diferencias7348.pdf')

#%% Comentarios auxiliares
#OJOTA: Para ordenar deberia reordenar tambien la matriz de covarianza! (lo picante es Csys)
#p = zcmb.argsort()
#zcmb = zcmb[p]
#zhel = zhel[p]
#dz = dz[p]
#mb = mb[p]
#dmb = dmb[p]
#%%
plt.close()
sn_eliminadas = 100
mask_1 = np.ones(len(deltamu), dtype = bool)

p = zcmb.argsort()
for k in p[:sn_eliminadas]: 
    mask_1[k] = False 

mask_2 = np.ones([len(deltamu), len(deltamu)], dtype = bool)
for i in p[:sn_eliminadas]: 
    for j in p[:sn_eliminadas]: 
        mask_2[i,j] = False
        
# Filtramos las primeras supernovas
deltamu_1 = deltamu[mask_1]
zcmb_1 = zcmb[mask_1]

Cinv_11 = Cinv[:,mask_1]
Cinv_1 = Cinv_11[mask_1,:]
#%%
plt.figure()
plt.plot(zcmb_1,deltamu_1,'.',label= r'$\Delta\mu$')
transp_1=np.transpose(deltamu_1)
aux_1 = np.dot(Cinv_1,deltamu_1)
chi2=np.dot(transp_1,aux_1)
print(chi2,chi2/sn)
