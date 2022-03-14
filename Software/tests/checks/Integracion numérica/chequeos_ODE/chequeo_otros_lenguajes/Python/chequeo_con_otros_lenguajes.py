"""
Created on Fri Oct 18 00:38:48 2019

@author: matias
"""
#Importo librerías
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import sympy as sym
from sympy.utilities.lambdify import lambdify
import math
from scipy.integrate import solve_ivp
from scipy.integrate import simps as simps
from scipy.integrate import cumtrapz as cumtrapz
from scipy.constants import c as c_luz #metros/segundos
c_luz_norm = c_luz/1000
import time
#%%
#Defino algunas funciones que sirven papra la integración en Python
def params_fisicos_to_modelo(omega_m, b, n=1):
    '''Toma los parametros fisicos (omega_m, el parametro de distorsión b)
    y devuelve los parametros del modelo c1 y c2'''
    c_luz_norm = c_luz/1000 #km/seg
    alpha = 1 / (8315)**2
    beta = 1 / 0.13
    aux = ((100/c_luz_norm)**2 * 6 * (1 - omega_m))  / (alpha * omega_m * beta)
    c_1 =  2/b
    c_2 =  (2/(aux * b))
    Lamb = 6 * (1-omega_m) /aux
    return c_1, c_2, Lamb


def dX_dz(z, variables,*params_modelo, model='HS'):
    '''Defino el sistema de ecuaciones a resolver. El argumento params_modelo
    es una lista donde los primeros n-1 elementos son los parametros del sistema,
    mientras que el útimo argumento especifica el modelo en cuestión,
    matematicamente dado por la función gamma.'''

    x = variables[0]
    y = variables[1]
    v = variables[2]
    w = variables[3]
    r = variables[4]

    [B,D,N] = params_modelo
    gamma = lambda r,b,d: ((1+d*r) * ((1+d*r)**2 - b)) / (2*b*d*r)
    G = gamma(r,B,D) #B y D son C1 y C2

    s0 = (-w + x**2 + (1+v)*x - 2*v + 4*y) / (z+1) #Ecuacion para x
    s1 = (- (v*x*G - x*y + 4*y - 2*y*v)) / (z+1) #Ecuacion para y
    s2 = (-v * (x*G + 4 - 2*v)) / (z+1) #Ecuacion para v
    s3 = (w * (-1 + x + 2*v)) / (z+1) #Ecuacion para w
    s4 = (-(x * r * G)) / (1+z) #Ecuacion para r
    return [s0,s1,s2,s3,s4]


#%% Condiciones Iniciales
x_0 = -0.339
y_0 = 1.246
v_0 = 1.64
w_0 = 1 + x_0 + y_0 - v_0
r_0 = 41
ci = [x_0, y_0, v_0, w_0, r_0] #Condiciones iniciales
cond_iniciales = ci

#%% Cosas que tengo que definir para integrar
sistema_ec=dX_dz
max_step = 0.01
z_inicial = 0
z_final = 3
cantidad_zs = 2000
zs = np.linspace(z_inicial,z_final,cantidad_zs)

#%% Parámetros de Nunes
#H0 = 73.48
b = 2
omega = 0.24
c1,c2,_ = params_fisicos_to_modelo(omega,b) #los convierto a los del cambio de variables
print(c1,c2) #c1 = 1 #c2 = 1/19

#%% Resuelvo la ODE
params_modelo=[c1, c2, 1]
sol = solve_ivp(sistema_ec, (z_inicial,z_final),cond_iniciales, t_eval=zs,
                args=params_modelo,max_step=max_step, method='Radau')
int_v =  cumtrapz(sol.y[2]/(1+sol.t),sol.t,initial=0)
E_python =  np.exp(-int_v) * (1+sol.t)**2
#H_python = H0 * E_python
z_python = sol.t

#%%
# Importo los datos de Octave H(z) (Poner la carpeta donde están los datos)
df = pd.read_csv(
        '/home/matias/Documents/Tesis/fR-MCMC/Software/Integracion numérica/Octave/datos_octave.txt'
                 , header = None, delim_whitespace=True)
z_octave = np.array(df[0])
E_octave = np.array(df[1])

#%%
# Importo los datos de Mathematica (Poner la carpeta donde están los datos)
archivo_math = '/home/matias/Documents/Tesis/fR-MCMC/Software/Integracion numérica/Mathematica/datos_mathematica.csv'
z_math,v_math = np.loadtxt(archivo_math,unpack=True,delimiter = ',')
int_v_math =  cumtrapz(v_math/(1+z_math),z_math,initial=0)
E_math =  np.exp(-int_v_math) * (1+z_math)**2
#H_math = H0 * E_math


#%% Grafico los 3 datasets juntos
%matplotlib qt5
plt.figure(1)
plt.plot(sol.t, E_python,label='Python')
plt.plot(z_octave, E_octave,label='Octave')
plt.plot(z_math, E_math,label='Mathematica')
plt.xlabel('z(redshift)')
plt.ylabel(r'$E(z)$')
plt.legend(loc='best')
plt.grid(True)

#%% Error porcentual entre Python y Octave
f_octave = interp1d(z_octave,E_octave) #Interpolo los datos de Octave
#para poder evaluar ambos en los mismos zs.
error_octave = (1-np.array(f_octave(z_python))/E_python)
plt.close()
plt.figure(2)
plt.plot(z_python, error_octave)

plt.xlabel('z(redshift)')
plt.ylabel(r'$\frac{H_{python}-H_{octave}}{H_{python}}$', size =20)
plt.grid(True)

#%% Error porcentual entre Python y Mathematica
f_mathematica = interp1d(z_math,E_math) #Interpolo los datos de Mathematica
#para poder evaluar ambos en los mismos zs.
error_mathematica = (1-np.array(f_mathematica(z_python))/E_python)
plt.close()
plt.figure(3)
plt.plot(z_python, error_mathematica)
plt.xlabel('z(redshift)')
plt.ylabel(r'$\frac{H_{python}-H_{Mathematica}}{H_{python}}$', size =20)
plt.grid(True)
