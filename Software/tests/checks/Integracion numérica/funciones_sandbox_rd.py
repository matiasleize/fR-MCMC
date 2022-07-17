"""
Created on Sun Feb  2 13:28:48 2020

@author: matias
"""

import numpy as np
from numpy.linalg import inv
from matplotlib import pyplot as plt
import time
import scipy
from scipy.integrate import solve_ivp
from scipy.integrate import cumtrapz as cumtrapz
from scipy.integrate import simps as simps
from scipy.interpolate import interp1d
from scipy.constants import c as c_luz #meters/seconds
c_luz_km = c_luz/1000
#%%

def H_LCDM(z, omega_m, H_0):
    omega_r = 4.18343*10**(-5) / (H_0/100)**2
    omega_lambda = 1 - omega_m - omega_r
    H = H_0 * np.sqrt(omega_r * (1 + z)**4 + omega_m * (1 + z)**3 + omega_lambda)
    return H

def H_LCDM_a(a, omega_m, H_0):
    omega_r = 4.18343*10**(-5) / (H_0/100)**2
    omega_lambda = 1 - omega_m - omega_r
    H = H_0 * np.sqrt(omega_r * a**(-4) + omega_m * a**(-3) + omega_lambda)
    return H

H_0 = 73.48
omega_m = 0.24

#Defino el zd
def zdrag(omega_m,H_0,wb=0.0225):
    '''
    omega_b = 0.05 #(o algo asi, no lo usamos directamente)
    wb = 0.0222383 #Planck
    wb = 0.0225 #BBN
    '''
    h = H_0/100
    b1 = 0.313*(omega_m*h**2)**(-0.419)*(1+0.607*(omega_m*h**2)**(0.6748))
    b2 = 0.238*(omega_m*h**2)**0.223
    zd = (1291*(omega_m*h**2)**0.251) * (1+b1*wb**b2) /(1+0.659*(omega_m*h**2)**0.828)
    #zd =1060.31
    return zd


def r_drag(omega_m,H_0,int_z=True,z_crit=5000):
    #rd calculation:
    h = H_0/100
    zd = zdrag(omega_m,H_0)
    wb=0.0225
    #R_bar = 31500 * wb * (2.726/2.7)**(-4)
    R_bar = wb * 10**5 / 2.473
    if int_z == True:
        #Integral lineal
        zs_int_lin = np.linspace(zd,z_crit,int(10**6))
        H_int_lin = H_LCDM(zs_int_lin,omega_m,H_0)

        #cs = c_luz_km / (np.sqrt(3*(1 + R_bar/(1 + zs_int))))
        #integrando = cs/H_int
        integrando_lin = c_luz_km / (H_int_lin * np.sqrt(3*(1 + R_bar*(1+zs_int_lin)**(-1))))

        rd_lin = simps(integrando_lin,zs_int_lin)

        #Integral logaritmica
        zs_int_log = np.logspace(np.log10(z_crit),13,int(10**5))
        H_int_log = H_LCDM(zs_int_log,omega_m,H_0)

        integrando_log = c_luz_km / (H_int_log * np.sqrt(3*(1 + R_bar*(1+zs_int_log)**(-1))))

        rd_log = simps(integrando_log,zs_int_log)
        #print(len(zs_int_lin),len(zs_int_log))
    else:
        #Integral lineal
        a_crit = 1/(1+z_crit)
        adrag = 1/(1+zd)
        as_int_lin = np.linspace(a_crit,adrag,int(10**7))
        H_int_lin = H_LCDM_a(as_int_lin,omega_m,H_0)

        #cs = c_luz_km / (np.sqrt(3*(1 + R_bar/(1 + zs_int))))
        #integrando = cs/H_int
        integrando_lin = 1 / (as_int_lin**2 * H_int_lin *
                        np.sqrt(3*(1 + R_bar*as_int_lin)))
        rd_lin = c_luz_km * simps(integrando_lin,as_int_lin)

        #Integral logaritmica
        as_int_log = np.logspace(-8,np.log10(a_crit),int(10**4))
        H_int_log = H_LCDM_a(as_int_log,omega_m,H_0)

        integrando_log =  1 / (as_int_log**2 * H_int_log *
                        np.sqrt(3*(1 + R_bar*as_int_log)))

        rd_log = c_luz_km * simps(integrando_log,as_int_log)
        #print(len(as_int_lin),len(as_int_log))

    return rd_lin + rd_log

zc = 10 #funciona para todo zc :D
a=r_drag(omega_m,H_0,z_crit=zc)
b=r_drag(omega_m,H_0,int_z=False,z_crit=zc)
print(a)
print(b)
print(100*(a-b)/a)
#Error de 10**(-3)

#%% Ploteamos los integrandos
H_0 = 73.48
omega_r = 4.18343*10**(-5) / ((H_0/100)**2)
omega_m = 0.24
omega_lambda = 1-omega_m-omega_r
wb = 0.0225
ases = np.linspace(0.0001,1,1000)
zs = np.linspace(1000,100000,1000)
integrando_a =  lambda a: (c_luz_km/H_0) / (-a**2 * (np.sqrt(omega_r * a**(-4) + omega_m * a**(-3) + omega_lambda)) *
                np.sqrt(3 * (1 + (31500 * wb * (2.726/2.7)**(-4) * a))))
integrando_z =  lambda z: (c_luz_km/H_0) / (np.sqrt(omega_r * (1+z)**(4) + omega_m * (1+z)**(3) + omega_lambda) *
                np.sqrt(3 * (1 + (31500 * wb * (2.726/2.7)**(-4) * (1+z)**(-1) ))))
plt.figure(1)
plt.plot(ases,integrando_a(ases))
plt.title('Integrando en funcion de a')
plt.grid(True)
plt.xlabel('a (Factor de escala)')
plt.ylabel('Integrando')
plt.figure(2)
plt.plot(zs,integrando_z(zs))
plt.title('Integrando en funcion de z')
plt.grid(True)
plt.xlabel('z (Redshift)')
plt.ylabel('Integrando')
#%%
omegas = np.linspace(0.1,1,50)
rds = np.zeros(len(omegas))
for i,omega in enumerate(omegas):
    theta = [omega,H_0]
    rds[i]= r_drag(omega,H_0)
%matplotlib qt5
plt.title('$r_{d}$ en función de $\Omega_{m}$ ')
plt.plot(omegas,rds)
plt.grid(True)
plt.xlabel(r'$\Omega_{m}$')
plt.ylabel(r'$r_{d}$(Mpc)')
#import cobaya.theories.camb.camb()


#%%
import numpy as np
from numpy.linalg import inv
from matplotlib import pyplot as plt
import time
import scipy
from scipy.integrate import solve_ivp
from scipy.integrate import cumtrapz as cumtrapz
from scipy.integrate import simps as simps
from scipy.interpolate import interp1d
from scipy.constants import c as c_luz #meters/seconds
c_luz_km = c_luz/1000


def H_LCDM(a, omega_m, H_0):
    omega_r = 4.18343*10**(-5) * (100/H_0)**2
    omega_lambda = 1 - omega_m - omega_r
    H = H_0 * np.sqrt(omega_r * a**(-4) + omega_m * a**(-3) + omega_lambda)
    return H

H_0 = 73.48
omega_m = 0.24
import scipy.integrate as integrate

def r_drag_ints(omega_m,H_0):
    #rd calculation:
    h = H_0/100
    zd = zdrag(omega_m,H_0)
    wb=0.0225
    omega_r = 4.18343*10**(-5) * (100/H_0)**2
    omega_lambda = 1 - omega_m - omega_r
    R_bar = 31500 * wb * (2.726/2.7)**(-4)

    #Integral lineal
    adrag = 1/(1+zd)
    as_int_lin = np.linspace(adrag,10**(-8),int(10**7))
    H_int_lin = H_LCDM_a(as_int_lin,omega_m,H_0)

    #cs = c_luz_km / (np.sqrt(3*(1 + R_bar/(1 + zs_int))))
    #integrando = cs/H_int
    integrando_lin = 1 / (-as_int_lin**2 * H_int_lin *
                    np.sqrt(3*(1 + R_bar*as_int_lin)))
    rd_simps = c_luz_km * simps(integrando_lin,as_int_lin)

    #Integramos tomando funciones
    integrando =  lambda a: 1 / (-a**2 * (np.sqrt(omega_r * a**(-4) + omega_m
                    * a**(-3) + omega_lambda)) *
                    np.sqrt(3 * (1 + (31500 * wb * (2.726/2.7)**(-4) * a))))

    #Usando quad
    I_quad_gauss =  integrate.quad(integrando, adrag , 0)[0]
    rd_quad_gauss = (c_luz_km/H_0) * I_quad_gauss

    #Usando quadrature (Gausssian quadrature)
    I_quad =  integrate.quadrature(integrando, adrag, 0)[0]
    rd_quad = (c_luz_km/H_0) * I_quad

    #Usando Romberg
    I_romberg = integrate.romberg(integrando, adrag, 10**(-8))
    rd_romberg = (c_luz_km/H_0) * I_romberg

    return rd_simps, rd_quad, rd_quad_gauss, rd_romberg
print(r_drag_ints(omega_m,H_0))

#%%
import camb
from camb import model, initialpower
#Set up a new set of parameters for CAMB
#calculate results for these parameters
H_0 = 65
def r_drag_camb(omega_m,H_0,wb=0.0225):
    pars = camb.CAMBparams()
    h = (H_0/100)
    pars.set_cosmology(H0=H_0, ombh2=wb, omch2=omega_m*h**2-wb)
    results = camb.get_background(pars)
    rd = results.get_derived_params()['rdrag']
    #print('Derived parameter dictionary: %s'%results.get_derived_params()['rdrag'])
    return rd

a=r_drag_camb(omega_m=0.24,H_0=73.48)
b=r_drag(omega_m=0.24,H_0=73.48)

100*(np.abs(a-b)/a)
r_drag_camb(0.27,72,wb=0.0448*(0.72**2))
#%%
omegas = np.linspace(0.1,0.99,50)
rds_int = np.zeros(len(omegas))
rds_camb = np.zeros(len(omegas))
for i,omega in enumerate(omegas):
    theta = [omega,H_0]
    rds_camb[i] = r_drag_camb(omega,H_0)
    rds_int[i] = r_drag(omega,H_0)
%matplotlib qt5
plt.close()
plt.figure(1)
plt.title('$r_{d}$ en función de $\Omega_{m}, H0 = 65$ ')
plt.plot(omegas,rds_camb,label='camb')
plt.plot(omegas,rds_int,label='integral')
plt.grid(True)
plt.xlabel(r'$\Omega_{m}$')
plt.ylabel(r'$r_{d}$(Mpc)')
plt.legend()
plt.figure(2)
plt.grid(True)
plt.plot(omegas,((rds_camb-rds_int)/rds_camb)*100)
plt.xlabel(r'$\Omega_{m}$')
plt.ylabel(r'$error %$(Mpc)')
#%%%
import numpy as np
from scipy.integrate import simps as simps
xs = np.linspace(1,10**6,int(10**7))
output = simps(np.exp(-xs),xs)
output
real = 1/np.e

100*(1-output/real)
