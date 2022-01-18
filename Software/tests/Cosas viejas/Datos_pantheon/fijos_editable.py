#en este programa vamos a calcular el chi2 MOG-sn considerando todos los parametros fijos. Solo hay que prestarle atencion al valor de H0.
import math
import numpy as np
from numpy.linalg import inv
import  matplotlib.pyplot as mp
from scipy.integrate import simps
from matplotlib import pyplot as plt
c=3.0e8*3.154e7
#Ojo que el valor de H0 esta puesto a mano!tiene que ser el mismo que el del mathe que genero la tabla que se lee mas abajo
H0=73.48*3.154e7/(3.086e19)
conv=1000000*3.0857e16
omega_m = 0.3
def inte(z,omega_m):
	integrando = (omega_m * (1+z)**3 + (1-omega_m))**(-1)
	return integrando

z = np.linspace(0,1000)
Integrando = inte(z,omega_m)

zcmb,zhel,dz,mb,dmb=np.loadtxt('/home/matias/Documents/Tesis/tesis_licenciatura/Software/Estadística/Cosas viejas/Datos_pantheon/lcparam_full_long_zhel.txt', usecols=(1,2,3,4,5),unpack=True)
sn=len(zcmb)
#longitud de la tabla de sn:
d_l=np.zeros(sn)
#para cada sn voy a hacer la integral correspondiente:
#lo que hago es cortar la lista en el limite superior de la integral que me lo da zcmb.
aux = np.zeros(sn)
for i in range(1,sn):
	j=int(round(zcmb[i]/0.00001))
	aux[i] = j
	Intj=Integrando[:j]
	zj=z[:j]
	d_l[i] = (1+zhel[i])*(c/H0)*simps(Integrando[:i], z[:i])/conv
plt.plot(zcmb,d_l,'.')
len(Integrando[:5])

print(aux)
print(zcmb)
print(sn)
