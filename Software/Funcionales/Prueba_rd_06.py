"""
Created on Sun Feb  2 13:28:48 2020

@author: matias
"""
import numpy as np
from numpy.linalg import inv
from matplotlib import pyplot as plt
import time
import camb
from camb import model, initialpower
from scipy.integrate import solve_ivp
from scipy.integrate import cumtrapz as cumtrapz
from scipy.integrate import simps as simps
from scipy.interpolate import interp1d
from scipy.constants import c as c_luz #metros/segundos
c_luz_km = c_luz/1000


# BAO
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

def H_LCDM_rad(z, omega_m, H_0):
    omega_r = 4.18343*10**(-5) / (H_0/100)**2
    omega_lambda = 1 - omega_m - omega_r
    H = H_0 * np.sqrt(omega_r * (1 + z)**4 + omega_m * (1 + z)**3 + omega_lambda)
    return H

def r_drag(omega_m,H_0,wb = 0.0225, int_z=True): #wb x default tomo el de BBN.
    #Calculo del rd:
    h = H_0/100
    zd = zdrag(omega_m,H_0)
    #R_bar = 31500 * wb * (2.726/2.7)**(-4)
    R_bar = wb * 10**5 / 2.473

#omega_b va como h^(-2)
#wb=omega_b * h**2 no depende de h

    #Integral logaritmica
    zs_int_log = np.logspace(np.log10(zd),13,int(10**5))
    H_int_log = H_LCDM_rad(zs_int_log,omega_m,H_0)

    integrando_log = c_luz_km / (H_int_log * np.sqrt(3*(1 + R_bar*(1+zs_int_log)**(-1))))

    rd_log = simps(integrando_log,zs_int_log)
    return rd_log

def r_drag_camb(omega_m,H_0,wb = 0.0225):
    pars = camb.CAMBparams()
    h = (H_0/100)
    pars.set_cosmology(H0=H_0, ombh2=wb, omch2=omega_m*h**2-wb)
    results = camb.get_background(pars)
    rd = results.get_derived_params()['rdrag']
    #print('Derived parameter dictionary: %s'%results.get_derived_params()['rdrag'])
    return rd

#Ref 15 de caro
omega_m = 0.31
h = 0.67
wb = 0.048*h**2
r_drag_camb(omega_m,h*100,wb=wb) #148.66089043939596
r_drag(omega_m,h*100,wb=wb) #147.52197078103742
#Deberia ser 148.69 Mpc

#Ref 16 de caro
omega_m = 0.31
h = 0.676
wb = 0.022
r_drag_camb(omega_m,h*100,wb=wb) #147.59690329061232
r_drag(omega_m,h*100,wb=wb) #146.48214042341743
#Deberia ser 147,78 Mpc

#Ref 17 de caro
omega_m = 0.25
h = 0.7
wb = 0.044*(h**2)
r_drag_camb(omega_m,h*100,wb=wb) #153.41991935756414
r_drag(omega_m,h*100,wb=wb) #152.31309641483853
#Deberia ser 153.44 Mpc

#Ref 79 de caro
omega_m = 0.27
h = 0.71
wb = 0.0448*(h**2)
r_drag_camb(omega_m,h*100,wb=wb) #148.5959478254231
r_drag(omega_m,h*100,wb=wb) #147.51886411943823
#Deberia ser 148.6 Mpc


#Ref 80 de caro
omega_m = 0.27
h = 0.71
wb = 0.0448*(h**2)
r_drag_camb(omega_m,h*100,wb=wb) #148.5959478254231
r_drag(omega_m,h*100,wb=wb) #146.06297885462257
#Deberia ser 147.78

#Ref 81 y 82 de caro
omega_m = 0.3147
h = 0.6731
wb = 0.02222
r_drag_camb(omega_m,h*100,wb=wb) #147.16624708118295
r_drag(omega_m,h*100,wb=wb) #146.06297885462257
#Deberia ser 147.33
