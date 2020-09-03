import sympy as sym
from sympy.utilities.lambdify import lambdify
import numpy as np
import math


from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate import simps as simps
from scipy.integrate import cumtrapz as cumtrapz
import time

from scipy.interpolate import interp1d
from scipy.constants import c as c_luz #metros/segundos
c_luz_norm = c_luz/1000


#np.sqrt((H0**2* (1/(omega_m-4*math.e(3*N)*(omega_m+omega_r-1))**8(b**2)*(math.exp(5*N))*((omega_m+omega_r-1)**3) *37*math.expNn*omega_m**6-4656*math.exp(4*N)*omega_m**5*(omega_m+omega_r-1)-
#7452*math.exp(7*N)*omega_m**4*(omega_m+omega_r-1)**2-
#8692*math.exp(3*N)*omega_m**4*omega_r*(omega_m+omega_r-1)-
#4032*math.exp(2*N)*omega_m**3*omega_r**2*(omega_m+omega_r-1)+
#25408*math.exp(10*N)*omega_m**3*(omega_m+omega_r-1)**3-
#25728*math.exp(6*N)*omega_m**3*omega_r*(omega_m+omega_r-1)**2-
#17856*math.exp(5*N)*omega_m**2*omega_r**2*(omega_m+omega_r-1)**2-
#22848*math.exp(13*N)*omega_m**2*(omega_m+omega_r-1)**4+
#22016*math.exp(9*N)*omega_m**2*omega_r*(omega_m+omega_r-1)**3-
#9216*math.exp(8*N)*omega_m*omega_r**2*(omega_m+omega_r-1)**3+
#9216*math.exp(16*N)*omega_m*(omega_m+omega_r-1)**5-
#2048*math.exp(12*N)*omega_m*omega_r*(omega_m+omega_r-1)**4+
#1024*math.exp(19*N)*(omega_m+omega_r-1)**6+
#3072*math.exp(15*N)*omega_r*(omega_m+omega_r-1)**5+40*omega_m**5*omega_r)+
#((2*b*math.exp(2*N)*(omega_m+omega_r-1)**2*(-6*math.exp(N)*omega_m**2+3*math.exp(4*N)*omega_m*(omega_m+omega_r-1)+12*math.exp(7*N)*(omega_m+omega_r-1)**2+4*math.exp(3*N)*omega_r*(omega_m+omega_r-1)-7*omega_m*omega_r))/(4*math.exp(3*N)*(omega_m+omega_r-1)-omega_m)**3 * (math.exp(-3*N)-1)*omega_m+(math.exp(-4*N)-1)*omega_r+1)))


def Taylor_HS(z,omega_m,b,H0):
    '''Calculo del H(z) sugerido por Basilakos para el modelo de Hu-Sawicki
    N!=a, N=ln(a)=-ln(1+z)'''
    omega_r=0
    N = sym.Symbol('N')
    Hs_tay=N
    Hs_tay = (H0**2*(1/((omega_m-4*math.e**(3*N)*(omega_m+omega_r-1))**8) * (b**2)*(math.e**(5*N))*((omega_m+omega_r-1)**3) *(37*math.e**(N)*omega_m**6-4656*math.e**(4*N)*omega_m**5*(omega_m+omega_r-1)-
    7452*math.e**(7*N)*omega_m**4*(omega_m+omega_r-1)**2-
    8692*math.e**(3*N)*omega_m**4*omega_r*(omega_m+omega_r-1)-
    4032*math.e**(2*N)*omega_m**3*omega_r**2*(omega_m+omega_r-1)+
    25408*math.e**(10*N)*omega_m**3*(omega_m+omega_r-1)**3-
    25728*math.e**(6*N)*omega_m**3*omega_r*(omega_m+omega_r-1)**2-
    17856*math.e**(5*N)*omega_m**2*omega_r**2*(omega_m+omega_r-1)**2-
    22848*math.e**(13*N)*omega_m**2*(omega_m+omega_r-1)**4+
    22016*math.e**(9*N)*omega_m**2*omega_r*(omega_m+omega_r-1)**3-
    9216*math.e**(8*N)*omega_m*omega_r**2*(omega_m+omega_r-1)**3+
    9216*math.e**(16*N)*omega_m*(omega_m+omega_r-1)**5-
    2048*math.e**(12*N)*omega_m*omega_r*(omega_m+omega_r-1)**4+
    1024*math.e**(19*N)*(omega_m+omega_r-1)**6+
    3072*math.e**(15*N)*omega_r*(omega_m+omega_r-1)**5+40*omega_m**5*omega_r)+
    ((2*b*math.e**(2*N)*(omega_m+omega_r-1)**2*(-6*math.e**(N)*omega_m**2+3*math.e**(4*N)*omega_m*(omega_m+omega_r-1)+12*math.e**(7*N)*(omega_m+omega_r-1)**2+4*math.e**(3*N)*omega_r*(omega_m+omega_r-1)-7*omega_m*omega_r))/(4*math.e**(3*N)*(omega_m+omega_r-1)-omega_m)**3)+
    (math.e**(-3*N)-1)*omega_m + (math.e**(-4*N)-1)*omega_r+1))**(0.5)

    func = lambdify(N, Hs_tay,'numpy') # returns a numpy-ready function

    N_dato = -np.log(1+z)
    numpy_array_of_results = func(N_dato)
    return numpy_array_of_results

#%%
z_inicial = 0
z_final = 3
cantidad_zs=31
#%%
archivo_math = '/home/matias/Documents/Tesis/tesis_licenciatura/Software/Integracion numérica/Mathematica/mfile_1.5.csv'
z_math,H_math=np.loadtxt(archivo_math,unpack=True,delimiter = ',')
#%%
b = 1.5
omega = 0.24
H0 = 73.48
#cond_iniciales = condiciones_iniciales(omega,b)
zs = np.linspace(z_inicial,z_final,cantidad_zs)
H_taylor=Taylor_HS(zs,omega,b,H0)

#%%
%matplotlib qt5
plt.close()
plt.figure()
plt.grid(True)
plt.xlabel('z(redshift)')
plt.ylabel('E(z)')
plt.plot(zs,H_taylor/H0,label='Taylor')
plt.plot(z_math,H_math/H0,label='Mathematica')
plt.legend(loc='best')
#%%
plt.figure()
plt.grid(True)
plt.xlabel('z(redshift)')
plt.ylabel('Error porcentual')
plt.plot(zs,100*(1-H_taylor/H_math),label='Error porcentual')
plt.legend(loc='best')
np.mean(100*(1-(H_taylor/H_math)))
