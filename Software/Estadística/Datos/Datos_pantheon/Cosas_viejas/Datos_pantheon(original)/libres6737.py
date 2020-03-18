#en este programa vamos a calcular el chi2 MOG-sn considerando todos los parametros libres. Ojo que tenemos menos dof. Hay que prestarle atencion al valor de H0.
import math
import numpy as np
from numpy.linalg import inv
import  matplotlib.pyplot as mp
from scipy import optimize
from scipy.integrate import simps
c=3.0e8*3.154e7
#Ojo que el valor de H0 esta puesto a mano!tiene que ser el mismo que el del mathe que genero la tabla que se lee mas abajo
H0=67.37*3.154e7/(3.086e19)
conv=1000000*3.0857e16
#leo la tabla de la funcion que tengo que integrar que viene del mathematica (donde tambien use H0)
z, Integrando = np.loadtxt('H6737.dat', unpack=True)
# leo las tablas de datos:
zhel,dz,mb0,dmb0=np.loadtxt('lcparam_full_long_zhel.txt', usecols=(2,3,4,5),unpack=True)
zcmb,hmass,dhmass,x1,dx1,cor,dcor,mb,dmb,x0,dx0,covx1c,covx1x0,covcx0=np.loadtxt('ancillary_g10.txt', usecols=(7,13,14,20,21,22,23,24,25,26,27,28,29,30),unpack=True)
#los errores son las desviaciones estandar, es decir, son sigma.
#longitud de la tabla de sn:
sn=len(zcmb)
#calculo de DeltaB. Necesito los parametros nuisance dados por Pantheon para G10 para poder despejar DeltaB
alfa0=0.154
beta0=3.02
gamma0=0.053
mstep0=10.13
tau0=0.001
DeltaM=np.zeros(sn)
DeltaM=gamma0*np.power((1.+np.exp((mstep0-hmass)/tau0)),-1)
# el calculo de deltaM tira algunos errores pero esta bien calculado, da practicamente un escalon.
DeltaB=mb0-mb-alfa0*x1+beta0*cor-DeltaM
# para el error vamos a usar el error en la magnitud final dado por pantheon. Ya tengo todos los valores que necesito


##################### Calculo de la parte teorica ##################
muth=np.zeros(sn)
#para cada sn voy a hacer la integral correspondiente:
#lo que hago es cortar la lista en el limite superior de la integral que me lo da zcmb.
for i in range(0,sn):
	j=int(round(zcmb[i]/0.00001))
	Intj=Integrando[:j]
	zj=z[:j]
	muth[i] = 25.0+5.0*math.log10((1+zhel[i])*(c/H0)*simps(Intj, zj)/conv)
#ya tengo el calculo del modulo de distancia teorico para cada supernova.


##################### errores ############################################
# hay que leer la matriz de los errores sistematicos que es de NxN
Csys=np.loadtxt('lcparam_full_long_sys.txt',unpack=True)
Csys=Csys.reshape(sn,sn)
#creamos la matriz diagonal con los errores de mB final. ojo! en realidad esto depende de alfa y beta, estamos asumiendo que nuestro alfa y beta no van a dar muy diferentes a los de Pantheon y entonces esta bien usar el error final que dan ellos. No podemos hacer el calculo nosotros porque no hay forma de conocer el error en DeltaB:
Dstat=np.diag(dmb0**2.)
#armamos la matriz de cov final y la invertimos:
Ccov=Csys+Dstat
Cinv=inv(Ccov)

#############################################################
################ calculo de la parte observada ##############
##############################################################
#Ahora nos falta el observado que depende de Mabs, alfa, beta, y gamma.
Mabs=np.linspace(-19.44,-19.415,20)
alfa=np.linspace(0.156,0.165,20)
beta=np.linspace(3.00,3.09,20)
gamma=np.linspace(0.04,0.07,20) 


############## calculo de chi min para cada alfa #########
chi2alfa=25000+np.zeros(20) #aca se va a guardar el mejor chi2 para cada alfa. 
for i in range(20): #alfa
    for j in range(20): #beta
        for h in range(20): #gamma
            DeltaMc=gamma[h]*np.heaviside(hmass-mstep0,1/2.)
            for k in range(20): #Mabs
                mbc=mb+alfa[i]*x1-beta[j]*cor+DeltaMc+DeltaB #m aparente calculada
                muobs=mbc-Mabs[k]
                deltamu=muobs-muth
                transp=np.transpose(deltamu)
                chi2temp=np.dot(np.dot(transp,Cinv),deltamu) #esto es el chi2 para cada conjunto i,j,h,k.
                #mp.plot(alfa[i],chi2temp,'r.')
                if chi2temp<chi2alfa[i]:
                    chi2alfa[i]=chi2temp
    print(alfa[i],chi2alfa[i])
#mp.savefig('chisalfa.pdf')
#chequeado que esto esta funcionando bien, hay que mirar que sea una parabola
#ajustamos un polinomio de grado dos a la parabola y buscamos el minimo
fitalfa=np.polyfit(alfa,chi2alfa,2) #esto me devuelve un array con a,b,c
polalfa=np.poly1d(fitalfa) #esto me lo convierte para poder evaluar el pol
alfamin = optimize.fmin(polalfa, 0.157)
#alfamin[0] es el minimo en alfa. y polalfa(alfamin[0]) el valor del chi2 en el minimo
#ahora hay que encontrar el error. hay que sumarle 1.00 al chi2 de acuerdo a la tabla de la pg 815 del numerical recipies.
chi2sigmaalfa=polalfa(alfamin[0])+1.00
#ahora hay que encontrar el alfa asociado a este valor de chi2. depejo la cuadratica considerando c=c-chi2sigmaalfa
alfasigma=(-fitalfa[1]+math.sqrt((fitalfa[1]**2.-4.0*fitalfa[0]*(fitalfa[2]-chi2sigmaalfa))))/(2.0*fitalfa[0])
print(alfamin[0],abs(alfasigma-alfamin[0]),polalfa(alfamin[0])) #escribo el minimo, la desviacion a 1 sigma  y el chi2 minimo
#plot
xpalfa=np.linspace(0.156,0.165,100)
mp.plot(alfa,chi2alfa,'r.',xpalfa,polalfa(xpalfa),'-')
mp.savefig('alfavschi6737.pdf')
mp.clf()


############## calculo de chi min para cada beta #########
chi2beta=25000+np.zeros(20) #aca se va a guardar el mejor chi2 para cada beta. 
for j in range(20): #beta
    for i in range(20): #alfa
        for h in range(20): #gamma
            DeltaMc=gamma[h]*np.heaviside(hmass-mstep0,1/2.)
            for k in range(20): #Mabs
                mbc=mb+alfa[i]*x1-beta[j]*cor+DeltaMc+DeltaB #m aparente calculada
                muobs=mbc-Mabs[k]
                deltamu=muobs-muth
                transp=np.transpose(deltamu)
                chi2temp=np.dot(np.dot(transp,Cinv),deltamu) #esto es el chi2 para cada conjunto i,j,h,k.
                if chi2temp<chi2beta[j]:
                    chi2beta[j]=chi2temp
    print(beta[j],chi2beta[j])
#chequeado que esto esta funcionando bien, hay que mirar que sea una parabola
#ajustamos un polinomio de grado dos a la parabola y buscamos el minimo
fitbeta=np.polyfit(beta,chi2beta,2) #esto me devuelve un array con a,b,c
polbeta=np.poly1d(fitbeta) #esto me lo convierte para poder evaluar el pol
betamin = optimize.fmin(polbeta, 3.00)
#betamin[0] es el minimo en beta. y polbeta(betamin[0]) el valor del chi2 en el minimo
#ahora hay que encontrar el error. hay que sumarle 1.00 al chi2 de acuerdo a la tabla de la pg 815 del numerical recipies.
chi2sigmabeta=polbeta(betamin[0])+1.00
#ahora hay que encontrar el beta asociado a este valor de chi2. depejo la cuadratica considerando c=c-chi2sigma
betasigma=(-fitbeta[1]+math.sqrt((fitbeta[1]**2.-4.0*fitbeta[0]*(fitbeta[2]-chi2sigmabeta))))/(2.0*fitbeta[0])
print(betamin[0],abs(betasigma-betamin[0]),polbeta(betamin[0])) #escribo el minimo, la desviacion a 1 sigma  y el chi2 minimo
#plot
xpbeta=np.linspace(3.00,3.09,100)
mp.plot(beta,chi2beta,'r.',xpbeta,polbeta(xpbeta),'-')
mp.savefig('betavschi6737.pdf')
mp.clf()

############## calculo de chi min para cada gamma #########
chi2gamma=25000+np.zeros(20) #aca se va a guardar el mejor chi2 para cada gamma. 
for h in range(20): #gamma
    DeltaMc=gamma[h]*np.heaviside(hmass-mstep0,1/2.)
    for i in range(20): #alfa
        for j in range(20): #beta
            for k in range(20): #Mabs
                mbc=mb+alfa[i]*x1-beta[j]*cor+DeltaMc+DeltaB #m aparente calculada
                muobs=mbc-Mabs[k]
                deltamu=muobs-muth
                transp=np.transpose(deltamu)
                chi2temp=np.dot(np.dot(transp,Cinv),deltamu) #esto es el chi2 para cada conjunto i,j,h,k.
                if chi2temp<chi2gamma[h]:
                    chi2gamma[h]=chi2temp
    print(gamma[h],chi2gamma[h])
#chequeado que esto esta funcionando bien, hay que mirar que sea una parabola
#ajustamos un polinomio de grado dos a la parabola y buscamos el minimo
fitgamma=np.polyfit(gamma,chi2gamma,2) #esto me devuelve un array con a,b,c
polgamma=np.poly1d(fitgamma) #esto me lo convierte para poder evaluar el pol
gammamin = optimize.fmin(polgamma, 0.10)
#gammamin[0] es el minimo en gamma. y polgamma(gammamin[0]) el valor del chi2 en el minimo
#ahora hay que encontrar el error. hay que sumarle 1.00 al chi2 de acuerdo a la tabla de la pg 815 del numerical recipies.
chi2sigmagamma=polgamma(gammamin[0])+1.00
#ahora hay que encontrar el gamma asociado a este valor de chi2. depejo la cuadratica considerando c=c-chi2sigma
gammasigma=(-fitgamma[1]+math.sqrt((fitgamma[1]**2.-4.0*fitgamma[0]*(fitgamma[2]-chi2sigmagamma))))/(2.0*fitgamma[0])
print(gammamin[0],abs(gammasigma-gammamin[0]),polgamma(gammamin[0])) #escribo el minimo, la desviacion a 1 sigma  y el chi2 minimo
#plot
xpgamma=np.linspace(0.04,0.07,100)
mp.plot(gamma,chi2gamma,'r.',xpgamma,polgamma(xpgamma),'-')
mp.savefig('gammavschi6737.pdf')
mp.clf()

############## calculo de chi min para cada Mabs #########
chi2Mabs=25000+np.zeros(20) #aca se va a guardar el mejor chi2 para cada Mabs. 
for k in range(20): #Mabs
    for i in range(20): #alfa
        for j in range(20): #beta
            for h in range(20): #gamma
                DeltaMc=gamma[h]*np.heaviside(hmass-mstep0,1/2.)
                mbc=mb+alfa[i]*x1-beta[j]*cor+DeltaMc+DeltaB #m aparente calculada
                muobs=mbc-Mabs[k]
                deltamu=muobs-muth
                transp=np.transpose(deltamu)
                chi2temp=np.dot(np.dot(transp,Cinv),deltamu) #esto es el chi2 para cada conjunto i,j,h,k.
                if chi2temp<chi2Mabs[k]:
                    chi2Mabs[k]=chi2temp
    print(Mabs[k],chi2Mabs[k])
#chequeado que esto esta funcionando bien, hay que mirar que sea una parabola
#ajustamos un polinomio de grado dos a la parabola y buscamos el minimo
fitMabs=np.polyfit(Mabs,chi2Mabs,2) #esto me devuelve un array con a,b,c
polMabs=np.poly1d(fitMabs) #esto me lo convierte para poder evaluar el pol
Mabsmin = optimize.fmin(polMabs, 19.0)
#Mabsmin[0] es el minimo en Mabs. y polMabs(Mabsmin[0]) el valor del chi2 en el minimo
#ahora hay que encontrar el error. hay que sumarle 1.00 al chi2 de acuerdo a la tabla de la pg 815 del numerical recipies.
chi2sigmaMabs=polMabs(Mabsmin[0])+1.00
#ahora hay que encontrar el Mabs asociado a este valor de chi2. depejo la cuadratica considerando c=c-chi2sigma
Mabssigma=(-fitMabs[1]+math.sqrt((fitMabs[1]**2.-4.0*fitMabs[0]*(fitMabs[2]-chi2sigmaMabs))))/(2.0*fitMabs[0])
print(Mabsmin[0],abs(Mabssigma-Mabsmin[0]),polMabs(Mabsmin[0])) #escribo el minimo, la desviacion a 1 sigma  y el chi2 minimo
#plot
xpMabs=np.linspace(-19.44,-19.415,100)
mp.plot(Mabs,chi2Mabs,'r.',xpMabs,polMabs(xpMabs),'-')
mp.savefig('Mabsvschi6737.pdf')
mp.clf()
