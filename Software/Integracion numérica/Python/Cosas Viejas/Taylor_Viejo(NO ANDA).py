"""
Created on Sun Feb  2 13:28:48 2020

@author: matias
"""

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
    #gamma = lambda r,b,d: ((1+d*r) * (r * (1+d*r)**2 - b*r)) / (2*b*d*(r)**2)
    gamma = lambda r,b,d: ((1+d*r) * ((1+d*r)**2 - b)) / (2*b*d*r)
    G = gamma(r,B,D)

    s0 = (-w + x**2 + (1+v)*x - 2*v + 4*y) / (z+1) #Ecuacion para x
    s1 = (- (v*x*G - x*y + 4*y - 2*y*v)) / (z+1) #Ecuacion para y
    s2 = (-v * (x*G + 4 - 2*v)) / (z+1) #Ecuacion para v
    s3 = (w * (-1 + x + 2*v)) / (z+1) #Ecuacion para w
    s4 = (-x * r * G) / (1+z) #Ecuacion para r
    return [s0,s1,s2,s3,s4]

def Taylor_HS(z,omega_m,b,H0):
    '''Calculo del H(z) sugerido por Basilakos para el modelo de Hu-Sawicki
    N!=a, N=ln(a)=-ln(1+z)'''

    N = sym.Symbol('N')

    HL = H0 * (omega_m * (math.e**(-3*N)) + (1 - omega_m))**(0.5)

    #Calculo las derivadas
    HL_N = sym.diff(HL,N)
    HL_2N = sym.diff(HL_N,N)
    HL_3N = sym.diff(HL_2N,N)
    HL_4N = sym.diff(HL_3N,N)

    #Calculo los términos del taylor
    dE21 =  ((H0**2 * (-1 + omega_m)**2 * (6 * HL**2 + 4 * HL_N**2 + HL * (15 * HL_N + 2 * HL_2N))) / (2 * HL * (2 * HL + HL_N)**3))

    #REVISAR
    t1 = ((-1 + omega_m) * H0**2)

    t2 = (540 * t1 * HL_N**2 + 124 * HL_N**4 + 12*HL_N**3 * HL_2N
    + 3 * t1 * HL_2N * (17 * HL_2N + 4 * HL_3N) + 2 * t1 * HL_N * (129*HL_2N
    + 12*HL_3N - HL_4N))

    t3 = (84 * t1 * HL_N**3 - 53 * HL_N**5
    - 3 * HL_N**4 * HL_2N + 21 * t1 * HL_2N**3 + 3 * t1 * HL_N * HL_2N * (41*HL_2N - 4*HL_3N)
    + t1 * HL_N**2 * (217 * HL_2N - 42 * HL_3N
    + HL_4N))


    dE22 = ((H0**4 * (-1 + omega_m)**3  * (128 * HL**8 - 32 * t1 * HL_N**6
            + 32 * HL**7 * (25*HL_N + 3*HL_2N) - 2 * t1 * HL * HL_N**4 * (139*HL_N + 22*HL_2N)
            + 16 * HL**6 * (9 * t1 + 89*HL_N**2 + 12 * HL_N * HL_2N)
            + HL**2 * HL_N**2 * (-749 * t1 * HL_N**2 + 9 * HL_N**4 - 48 * t1 * HL_2N**2
            - 4 * t1 * HL_N * (74*HL_2N - 3*HL_3N)) + 8 * HL**5 * (144 * t1 * HL_N
            + 146 * HL_N**3 + 18 * HL_N**2 * HL_2N + t1 * (15*HL_2N - 6*HL_3N - 4*HL_4N))
            + 4 * HL**4 * t2 - 2 * HL**3 * t3
            )) / (4*HL**4 * (2*HL + HL_N)**8))
    #dE22=0
    #dE21=0
    EL2 = (omega_m * (math.e**(-3*N)) + (1 - omega_m))

    HS_tay = H0 * (EL2 - b * dE21 - (b**2) * dE22)**(0.5)
    func = lambdify(N, HS_tay,'numpy') # returns a numpy-ready function

    N_dato = -np.log(1+z)
    numpy_array_of_results = func(N_dato)
    return numpy_array_of_results


#%%
sistema_ec=dX_dz
z_inicial = 0
z_final = 3
cantidad_zs = 200000
max_step = 0.01
#max_step = np.inf
verbose = True

x_0 = -0.339
y_0 = 1.246
v_0 = 1.64
w_0 = 1 + x_0 + y_0 - v_0
r_0 = 41
ci = [x_0, y_0, v_0, w_0, r_0] #Condiciones iniciales
cond_iniciales = ci
#%%
b = 1
omega = 0.24
H0 = 73.48
#cond_iniciales = condiciones_iniciales(omega,b)
zs = np.linspace(z_inicial,z_final,cantidad_zs)

H_taylor=Taylor_HS(zs,omega,b,H0)

c1,c2,_ = params_fisicos_to_modelo(omega,b)
print(c1,c2)
#c1 = 1
#c2 = 1/19
params_modelo=[c1, c2, 1]
#params_modelo=[c1, 0.8538, 1]
sol = solve_ivp(sistema_ec, (z_inicial,z_final),cond_iniciales, t_eval=zs,
                args=params_modelo,max_step=max_step)#, method='BDF')

int_v =  cumtrapz(sol.y[2]/(1+sol.t),sol.t,initial=0)
E_ode =  np.exp(-int_v) * (1+sol.t)**2
H_ode = H0 * E_ode
#%%
%matplotlib qt5
plt.close()
plt.figure()
plt.grid(True)
plt.xlabel('z(redshift)')
plt.ylabel('E(z)')
plt.plot(zs,H_taylor/H0,label='Taylor')
plt.plot(zs,E_ode,label='ODE')
plt.legend(loc='best')
#%%
plt.figure()
plt.grid(True)
plt.xlabel('z(redshift)')
plt.ylabel('Error porcentual')
plt.plot(zs,100*(1-H_taylor/H_ode),label='resta')
plt.legend(loc='best')
np.mean(100*(1-(H_taylor/H_ode)))
#%%
'''SOLO ANDA PARA C1=1, CAMBIARLO!'''
def condiciones_iniciales(omega, b, h0=1, r0_param=9.840, z0=1, n=1):
    #RHS = m**2 = Lambda * H_0**2 (Que ha el imput sea Rhs/H0)

    c1,c2,lamb = params_fisicos_to_modelo(omega,b)
    #if n==1:
    #    f_0 = 1 - (c1/c2**2) * (12/omega - 9)**(-2)
    aux = c2 * r0_param * (c2*r0_param + 2*lamb)/((c2*r0_param+lamb)**2)
    y0 = r0_param * (r0_param*c2+lamb)/(6*h0**2*(r0_param*c2+2*lamb))
    v0 = r0_param/(6*h0**2)
    w0 = omega * (z0+1)**3 / (h0**2 * aux)
    r0 = ((12/omega) - 9)
    x0 = w0 + v0 - y0 - 1
    return [x0,y0,v0,w0,r0]
