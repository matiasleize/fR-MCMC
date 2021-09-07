"""
Created on Sun Feb  2 13:28:48 2020

@author: matias
"""
import numpy as np
#from numba import jit
#import camb #Para que ande en el DF
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz as cumtrapz
from scipy.integrate import simps as simps
from scipy.integrate import quad as quad

from scipy.constants import c as c_luz #metros/segundos
c_luz_km = c_luz/1000;

import sys
import os
from os.path import join as osjoin
from pc_path import definir_path
path_git, path_datos_global = definir_path()
os.chdir(path_git)
sys.path.append('./Software/Funcionales/')

from funciones_int import Hubble_teorico
from funciones_LambdaCDM import H_LCDM_rad

#ORDEN DE PRESENTACION DE LOS PARAMETROS: omega_m,b,H_0,n

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

def r_drag_viejo(omega_m,H_0,wb = 0.0225, int_z=True): #wb x default tomo el de BBN.
    #Calculo del rd:
    h = H_0/100
    zd = zdrag(omega_m,H_0)
    #R_bar = 31500 * wb * (2.726/2.7)**(-4)
    R_bar = wb * 10**5 / 2.473

    #Integral logaritmica
    zs_int_log = np.logspace(np.log10(zd),13,int(10**5))
    H_int_log = H_LCDM_rad(zs_int_log,omega_m,H_0)

    integrando_log = c_luz_km / (H_int_log * np.sqrt(3*(1 + R_bar*(1+zs_int_log)**(-1))))

    rd_log = simps(integrando_log,zs_int_log)
    return rd_log

#@jit
def integrand(z, Om_m_0, H_0, wb):
    R_bar = wb * 10**5 / 2.473

    Om_r = 4.18343*10**(-5) / (H_0/100)**2
    Om_Lambda = 1 - Om_m_0 - Om_r
    H = H_0 * ((Om_r * (1 + z)**4 + Om_m_0 * (1 + z)**3 + Om_Lambda) ** (1/2))
    return c_luz_km/(H * (3*(1 + R_bar*(1+z)**(-1)))**(1/2))

def r_drag(omega_m,H_0,wb = 0.0225, int_z=True): #wb x default tomo el de BBN.
    #Calculo del rd:
    h = H_0/100
    zd = zdrag(omega_m,H_0)
    #R_bar = 31500 * wb * (2.726/2.7)**(-4)
    R_bar = wb * 10**5 / 2.473


   # Calculo del rd:
    zd = zdrag(omega_m, H_0)
    # zd = 1000
    R_bar = wb * 10**5 / 2.473

    rd_log, _ = quad(lambda z: integrand(z, omega_m, H_0, wb), zd, np.inf)

    return rd_log

def Hs_to_Ds(Hs_interpolado, int_inv_Hs_interpolado, z_data, index):
    if index == 4: #H
        output = Hs_interpolado(z_data)

    elif index == 1: #DH
        output = c_luz_km * (Hs_interpolado(z_data))**(-1)

    else:
        INT = int_inv_Hs_interpolado(z_data)

        if index == 0: #DA
            output = (c_luz_km/(1 + z_data)) * INT

        elif index == 2: #DM
            #output = (1 + z_data) * DA
            output = c_luz_km * INT

        elif index == 3: #DV
            #output = (((1 +z_data) * DA)**2 * c_luz_km * z_data * (Hs**(-1))) ** (1/3)
            output = c_luz_km * (INT**2 * z_data * (Hs_interpolado(z_data)**(-1))) ** (1/3)

    return output

def Ds_to_obs_final(zs, Dist, rd, index):
    if index == 4: #H
        output = Dist*rd
    else: #Todas las otras distancias
        output = Dist/rd
    return output

#%%
if __name__ == '__main__':
    '''Actualizar el ejemplo con estas funciones!'''
    import sys
    import os
    from os.path import join as osjoin
    from pc_path import definir_path
    path_git, path_datos_global = definir_path()
    os.chdir(path_git)
    sys.path.append('./Software/Funcionales/')
    from funciones_data import leer_data_BAO
    #%% BAO
    os.chdir(path_git+'/Software/Estadística/Datos/BAO/')
#    os.chdir(path_git+'/Software/Estadística/Datos/BAO/Datos_sin_nuevos')
    dataset_BAO = []
    archivo_BAO = ['datos_BAO_da.txt','datos_BAO_dh.txt','datos_BAO_dm.txt',
                    'datos_BAO_dv.txt','datos_BAO_H.txt']
    for i in range(5):
        aux = leer_data_BAO(archivo_BAO[i])
        dataset_BAO.append(aux)

    [omega_m,b,H_0] = [0.28,1,66.012] #Usando los datos nuevos
    theta = [omega_m,b,H_0]
    #params_to_chi2_BAO(theta,1, dataset_BAO,model='EXP')




    omega_m = 0.3
    H_0
    r_drag(omega_m,H_0,wb = 0.0225)
