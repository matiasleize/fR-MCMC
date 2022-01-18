"""
Created on Sun Feb  2 13:28:48 2020

@author: matias
"""
import numpy as np
from numpy.linalg import inv
from matplotlib import pyplot as plt
import scipy.integrate as integrate
import camb
from camb import model, initialpower
import time
import scipy
from scipy.integrate import solve_ivp
from scipy.integrate import cumtrapz as cumtrapz
from scipy.integrate import simps as simps
from scipy.interpolate import interp1d
from scipy.constants import c as c_luz #metros/segundos
c_luz_km = c_luz/1000



def H_LCDM(z, omega_m, H_0):
    omega_lambda = 1 - omega_m
    H = H_0 * np.sqrt(omega_m * (1+z)**(3) + omega_lambda)
    return H

def E_LCDM(z, omega_m):
    omega_lambda = 1 - omega_m
    E = np.sqrt(omega_m * (1+z)**(3) + omega_lambda)
    return E

H_0 = 73.48
omega_m = 0.262 #########


pars = camb.CAMBparams()
omega_bh2 = 0.0225 #BBN
pars.set_cosmology(H0=H_0, ombh2=omega_bh2, omch2=omega_m*(H_0/100)**2-omega_bh2)
results = camb.get_background(pars)
z = np.linspace(0,4,100)
DA = results.angular_diameter_distance(z)

INT = 1/E_LCDM(z,omega_m)
DA_int = (1+z)**(-1) * (c_luz_km/H_0) * cumtrapz(INT,z,initial=0)
#%matplotlib qt5
plt.figure(1)
plt.plot(z, DA,label='CAMB')
plt.plot(z, DA_int,label='Python')
plt.xlabel('$z$')
plt.ylabel(r'$D_A [\rm{Mpc}]$')
plt.title('Angular diameter distance')
plt.grid()
plt.legend()
plt.ylim([0,2000])
plt.xlim([0,4]);
#%%
mean_Da = (DA+DA_int)/2
error = 100* (DA[1:]-DA_int[1:])/mean_Da[1:]
plt.figure(2)
plt.plot(z[1:], error,label='Porcentual error')
plt.xlabel('$z$')
plt.ylabel(r'$eD_A}$')
plt.title('Error on the Angular Diameter Distance ($\Omega_{m}=0.262$)')#######
plt.grid()
plt.legend()
plt.xlim([0,4]);
plt.savefig('/home/matias/omega_m=0.262.png')#######
