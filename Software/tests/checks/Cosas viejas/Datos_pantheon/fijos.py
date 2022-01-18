#en este programa vamos a calcular el chi2 MOG-sn considerando todos los parametros fijos. Solo hay que prestarle atencion al valor de H0.
import math
import numpy as np
from numpy.linalg import inv
import  matplotlib.pyplot as mp
c=3.0e8*3.154e7
#Ojo que el valor de H0 esta puesto a mano!tiene que ser el mismo que el del mathe que genero la tabla que se lee mas abajo
H0=73.48*3.154e7/(3.086e19)
conv=1000000*3.0857e16
#leo la tabla de la funcion que tengo que integrar que viene del mathematica (donde tambien use H0)
z, Integrando = np.loadtxt('H7348.dat', unpack=True)
# ya chequee graficamente que el archivo se estuviera leyendo bien
# leo la tabla de datos:
zcmb,zhel,dz,mb,dmb=np.loadtxt('lcparam_full_long_zhel.txt', usecols=(1,2,3,4,5),unpack=True)
# ya chequee graficamente que el archivo se estuviera leyendo bien
#creamos la matriz diagonal con los errores de mB. ojo! esto depende de alfa y beta:
Dstat=np.diag(dmb**2.)
# hay que leer la matriz de los errores sistematicos que es de NxN
sn=len(zcmb)
Csys=np.loadtxt('lcparam_full_long_sys.txt',unpack=True)
Csys=Csys.reshape(sn,sn)
#armamos la matriz de cov final y la invertimos:
Ccov=Csys+Dstat
Cinv=inv(Ccov)
from scipy.integrate import simps
#longitud de la tabla de sn:
muth=np.zeros(sn)
#para cada sn voy a hacer la integral correspondiente:
#lo que hago es cortar la lista en el limite superior de la integral que me lo da zcmb.
for i in range(0,sn):
	j=int(round(zcmb[i]/0.00001))
	Intj=Integrando[:j]
	zj=z[:j]
	muth[i] = 25.0+5.0*math.log10((1+zhel[i])*(c/H0)*simps(Intj, zj)/conv)
#chequeado de forma grafica comparando con el mathe y con algunos puntos de forma analitica.
#ya tengo el calculo del modulo de distancia teorico para cada supernova.
#calculemos el x2 para un M fijo para empezar.
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
#pendientes: Liberar M
#		combinar las tablas .fits con el zhel de esta tabla. 
#		hacer el calculo de chi2 con los tres parametros y buscar los valores que lo minimizan
