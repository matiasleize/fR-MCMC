#en este programa vamos a calcular el chi2 MOG-sn considerando todos los parametros fijos. Solo hay que prestarle atencion al valor de H0.
import numpy as np
from numpy.linalg import inv
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import simps
from scipy.constants import c as C
from scipy.integrate import solve_ivp


H_0 =  74.2 #La de wikipedia de Planck (Vieja tradicional)
#H_0 =  67.4 #La que usaba Lucila
#H_0 =  54.4 #Nueva contante desde Planck, para usarla tenemos que usar la D_L de universo cerrado!
# z no tiene que empezar justo en 0!, empezarlo de 0.00001
#%%
#Gamma de Lucila
gamma = lambda r,b,c,d,n: ((1+d*r**n) * (-b*n*r**n + r*(1+d*r**n)**2)) / (b*n*r**n * (1-n+d*(1+n)*r**n))  


#Segun el paper c=0.24
#b = 1
d = 1/19 #(valor del paper)
c = 0.24
r_0 = 41
n = 1


def dX_dz(z, variables): 

    x = variables[0]
    y = variables[1]
    v = variables[2]
    w = variables[3]
    r = variables[4]
    
    G = gamma(r,b,c,d,n)
    
    s0 = (-w + x**2 + (1+v)*x - 2*v + 4*y) / (z+1)
    s1 = - (v*x*G - x*y + 4*y - 2*y*v) / (z+1)
    s2 = -v * (x*G + 4 - 2*v) / (z+1)
    s3 = w * (-1 + x+ 2*v) / (z+1)
    s4 = -x*r*G/(1+z)
        
    return [s0,s1,s2,s3,s4]

def plot_sol(solucion):
    
    '''Dado un gamma y una solución de las variables dinamicas, grafica estas
    por separado en una figura de 4x4.'''

    f, axes = plt.subplots(2,3)
    ax = axes.flatten()
    
    color = ['b','r','g','y','k']
    y_label = ['x','y','v','w','r']
    [ax[i].plot(solucion.t,solucion.y[i],color[i]) for i in range(5)]
    [ax[i].set_ylabel(y_label[i],fontsize='medium') for i in range(5)];
    [ax[i].set_xlabel('z (redshift)',fontsize='medium') for i in range(5)];
    [ax[i].invert_xaxis() for i in range(5)]; #Doy vuelta los ejes
    plt.show()
    

#%%
##Coindiciones iniciales e intervalo
x_0 = -0.339
y_0 = 1.246
v_0 = 1.64

w_0 = 1+x_0+y_0-v_0 

ci = [x_0, y_0, v_0, w_0, r_0] #Condiciones iniciales

# leo la tabla de datos:
zcmb,zhel,dz,mb,dmb=np.loadtxt('lcparam_full_long_zhel.txt', usecols=(1,2,3,4,5),unpack=True)

#Parte observacional (fija)
#creamos la matriz diagonal con los errores de mB. ojo! esto depende de alfa y beta:
Dstat=np.diag(dmb**2.)
# hay que leer la matriz de los errores sistematicos que es de NxN
sn=len(zcmb)
Csys=np.loadtxt('lcparam_full_long_sys.txt',unpack=True)
Csys=Csys.reshape(sn,sn)
#armamos la matriz de cov final y la invertimos:
Ccov=Csys+Dstat
Cinv=inv(Ccov)

#Magnitud aparente experimental
Mabs=-19.3
# calculamos el chi2
muobs=mb-Mabs

#%% Integramos el vector v y calculamos el Hubble

bs = np.linspace(0.85,0.95,5)
chis = np.zeros(len(bs))
for i, b in enumerate(bs):

    z = np.linspace(0.008,3,3000)
    E = np.zeros(len(z)) #Array de H/H_0
    
    for j in range(len(z)):
        zi = z[0]
        zf = z[j]
        sol = solve_ivp(dX_dz, [zi,zf], ci, max_step=0.005)     
    
        int_v = simps((sol.y[2])/(1+sol.t),sol.t) 
        E[j]=(1+zf)**2 * np.e**(-int_v) # integro desde 0 a z, ya arreglado
        
    H_int = interp1d(z,E) 
    
    #Parte teorica a partir de mi modelo de H(z)
    #para cada sn voy a hacer la integral correspondiente:
    #Calculamos la distancia luminosa d_L =  (c/H_0) int(dz'/E(z'))
    
    d_c=np.zeros(len(E)) #Distancia comovil
    for k in range (1, len(E)):
        d_c[k] = 0.001*(C/H_0) * simps(1/E[:k],z[:k])
    dc_int = interp1d(z,d_c) #Interpolamos  
    d_L = (1+zhel) * dc_int(zcmb) #Obs, Caro multiplica por Zhel, con Zobs da un poquin mejor
    
    #magnitud aparente teorica
    muth = 25+5*np.log10(d_L) #base cosmico
       
    deltamu=muobs-muth
    transp=np.transpose(deltamu)
    aux = np.dot(Cinv,deltamu)
    chi2=np.dot(transp,aux)
    
    chis[i] = chi2/sn
#%%
plt.plot(bs,chis,'.')
index_min = np.where(chis==min(chis))[0][0]
b_posta = bs[index_min]
print(b_posta)
#Comentario, para b=0.85 aun no llegamos al minimo, hay que seguir bajando b!