"""
Functions related to BAO data.
"""
import numpy as np
from numba import jit
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz as cumtrapz
from scipy.integrate import simps as simps
from scipy.integrate import quad as quad

from scipy.constants import c as c_luz #meters/seconds
c_luz_km = c_luz/1000

import os
import git
path_git = git.Repo('.', search_parent_directories=True).working_tree_dir
path_global = os.path.dirname(path_git)
os.chdir(path_git)
os.sys.path.append('./fr_mcmc/utils/')
from LambdaCDM import H_LCDM_rad

#Parameters order: omega_m,b,H_0,n

def zdrag(omega_m,H_0,wb=0.0225):
    '''
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
    #rd calculation:
    h = H_0/100
    zd = zdrag(omega_m,H_0)
    #R_bar = 31500 * wb * (2.726/2.7)**(-4)
    R_bar = wb * 10**5 / 2.473

    #Logarithmic integration
    zs_int_log = np.logspace(np.log10(zd),13,int(10**5))
    H_int_log = H_LCDM_rad(zs_int_log,omega_m,H_0)

    integrando_log = c_luz_km / (H_int_log * np.sqrt(3*(1 + R_bar*(1+zs_int_log)**(-1))))

    rd_log = simps(integrando_log,zs_int_log)
    return rd_log

@jit
def integrand(z, Om_m_0, H_0, wb):
    R_bar = wb * 10**5 / 2.473

    Om_r = 4.18343*10**(-5) / (H_0/100)**2
    Om_Lambda = 1 - Om_m_0 - Om_r
    H = H_0 * ((Om_r * (1 + z)**4 + Om_m_0 * (1 + z)**3 + Om_Lambda) ** (1/2))
    return c_luz_km/(H * (3*(1 + R_bar*(1+z)**(-1)))**(1/2))


def r_drag(omega_m,H_0,wb = 0.0225, int_z=True): #wb of BBN as default.
    #rd calculation:
    h = H_0/100
    zd = zdrag(omega_m,H_0)
    #R_bar = 31500 * wb * (2.726/2.7)**(-4)
    R_bar = wb * 10**5 / 2.473


    #zd calculation:
    zd = zdrag(omega_m, H_0)
    # zd = 1000
    R_bar = wb * 10**5 / 2.473

    rd_log, _ = quad(lambda z: integrand(z, omega_m, H_0, wb), zd, np.inf)

    return rd_log

def Hs_to_Ds(Hs_interpol, int_inv_Hs_interpol, z_data, index):
    if index == 4: #H
        output = Hs_interpol(z_data)

    elif index == 1: #DH
        output = c_luz_km * (Hs_interpol(z_data))**(-1)

    else:
        INT = int_inv_Hs_interpol(z_data)

        if index == 0: #DA
            output = (c_luz_km/(1 + z_data)) * INT

        elif index == 2: #DM
            #output = (1 + z_data) * DA
            output = c_luz_km * INT

        elif index == 3: #DV
            #output = (((1 +z_data) * DA)**2 * c_luz_km * z_data * (Hs**(-1))) ** (1/3)
            output = c_luz_km * (INT**2 * z_data * (Hs_interpol(z_data)**(-1))) ** (1/3)

    return output

def Ds_to_obs_final(zs, Dist, rd, index):
    if index == 4: #H
        output = Dist*rd
    else: #Every distances
        output = Dist/rd
    return output

#%%
if __name__ == '__main__':
    '''TODO: update the example with these functions'''
    import os
    import git
    path_git = git.Repo('.', search_parent_directories=True).working_tree_dir
    path_global = os.path.dirname(path_git)
    os.chdir(path_git)
    sys.path.append('./fr_mcmc/utils/')
    from data import read_data_BAO
    #%% BAO
    os.chdir(path_git+'/fr_mcmc/source/BAO')
    dataset_BAO = []
    file_BAO = ['BAO_data_da.txt','BAO_data_dh.txt','BAO_data_dm.txt',
                    'BAO_data_dv.txt','BAO_data_H.txt']
    for i in range(5):
        aux = read_data_BAO(file_BAO[i])
        dataset_BAO.append(aux)

    [omega_m,b,H_0] = [0.28,1,66.012]
    theta = [omega_m,b,H_0]
    params_to_chi2_BAO(theta,1, dataset_BAO,model='EXP')
    r_drag(omega_m,H_0,wb = 0.0225)
