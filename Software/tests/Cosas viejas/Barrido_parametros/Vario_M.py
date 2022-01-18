#en este programa vamos a calcular el chi2 MOG-sn considerando todos los parametros fijos. Solo hay que prestarle atencion al valor de H0.
import numpy as np
from numpy.linalg import inv
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import simps
from scipy.constants import c
#H_0 =  74.2 #La de wikipedia de Planck (Vieja tradicional)
H_0 =  73.48
#H_0 =  67.37 #La que usaba Lucila
#H_0 =  54.4 #Nueva contante desde Planck, para usarla tenemos que usar la D_L de universo cerrado!
#%%
#leo la tabla de la funcion que tengo que integrar que viene del mathematica (donde tambien use H0)

#npzfile = np.load('/home/matias/Documents/Tesis/Software/Integracion numérica/Python/Sistema_5ec/H(z).npz')
npzfile = np.load('/home/matias/Documents/Tesis/Software/Integracion numérica/Python/Sistema_5ec/H(z)_lcdm.npz')

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

#%%
#Parte teorica a partir de mi modelo de H(z)
#para cada sn voy a hacer la integral correspondiente:
#Calculamos la distancia luminosa d_L =  (c/H_0) int(dz'/E(z'))

d_c=np.zeros(len(E)) #Distancia comovil
for i in range (1, len(E)):
    d_c[i] = 0.001*(c/H_0) * simps(1/E[:i],z[:i])
    
dc_int = interp1d(z,d_c) #Interpolamos 

d_L = (1+zhel) * dc_int(zcmb) #Obs, Caro multiplica por Zhel, con Zobs da un poquin mejor

##magnitud aparente teorica
muth = 25 + 5 * np.log10(d_L) #base cosmico

#%%
#Magnitud aparente observacional
#Mabs=-19.3

Ms = np.linspace(-19.6,-19,10000)
chis = np.zeros(len(Ms))
for i, Mabs in enumerate(Ms):
    muobs =  mb - Mabs
    deltamu=muobs-muth
    transp=np.transpose(deltamu)
    aux = np.dot(Cinv,deltamu)
    chi2=np.dot(transp,aux)
    chis[i] = chi2/sn
#%% Para ver cuanto es un sigma, me fijo cuales valores de chi estan a +- 1.07
delta_chi = chis-min(chis)

index_min = np.where(chis==min(chis))[0][0]
index_sigma_M = np.where(delta_chi<1)[0][0] #1 corresponde a 1 sigma
index_sigma_m = np.where(delta_chi<1)[0][-1]

M_posta = Ms[index_min]

# OJOTA, chequear que este bien esto que estoy reportando
print('Magnitud es {} +{} -{}'.format(M_posta,abs(M_posta-Ms[index_sigma_M]),abs(M_posta-Ms[index_sigma_m]))) 

#%%
plt.figure()
plt.plot(Ms,delta_chi,'.')
plt.title(r'$\Delta\chi^{2}$ vs M')
plt.xlabel('M')
plt.ylabel(r'$\Delta\chi^{2}$')
plt.grid(True)
print(min(chis))

#%%
muobs_posta=mb-M_posta

plt.close()
plt.figure()
plt.errorbar(zcmb,muobs_posta,yerr=dmb, fmt='.',label='observado')
plt.plot(zcmb,muth,'r.',label='teorico')
plt.title('Distancia relativa en función del redshift')
plt.xlabel('z (reshift)')
plt.ylabel(r'$\mu$')
plt.grid(True)
plt.legend(loc='best')


#%%
plt.close()
aux = dmb/zcmb
dmb_lala = dmb
dmb_2 = aux[zcmb>0.5]
dmb_3 = aux[zcmb>1]

#for dmb_lala in (dmb_1, dmb_2):
plt.hist(dmb_lala,bins=int(np.sqrt(len(dmb_lala))),alpha=0.5,density=True)
plt.grid(True)
plt.legend(loc='best')