"""
Functions related to SnIa data.
"""

import numpy as np

import os
import git
path_git = git.Repo('.', search_parent_directories=True).working_tree_dir
path_global = os.path.dirname(path_git)

os.chdir(path_git)
os.sys.path.append('./fr_mcmc/utils/')

from LambdaCDM import H_LCDM

from scipy.interpolate import interp1d
from scipy.constants import c as c_luz #meters/seconds
c_luz_km = c_luz/1000; #kilometers/seconds
#Parameters order: Mabs,omega_m,b,H_0,n

def aparent_magnitude_th(int_inv_Hs_interpol, zcmb, zhel):
    '''
    Given an interpolate function of 1/H and arrays for zcmb and zhel,
    this function returns the theoretical expression for the distance modulus (mu)
    muth = 25 + 5 * log_{10}(d_L),
    where d_L = (c/H_0) (1+z) int(dz'/E(z')).
    '''

    dc_int =  c_luz_km * int_inv_Hs_interpol(zcmb)
    d_L = (1 + zhel) * dc_int
    muth = 25.0 + 5.0 * np.log10(d_L)
    return muth

def chi2_supernovae(muth, muobs, C_invertida):
    '''This function estimates the value of the statistic chi squared
    for the Supernovae data.'''
    deltamu = muth - muobs #row vector
    transp = np.transpose(deltamu) #column vector
    aux = np.dot(C_invertida,transp) #column vector
    chi2 = np.dot(deltamu,aux) #scalar
    return chi2


#%%
if __name__ == '__main__':


    import numpy as np
    np.random.seed(42)
    from matplotlib import pyplot as plt

    import os
    import git
    path_git = git.Repo('.', search_parent_directories=True).working_tree_dir
    path_global = os.path.dirname(path_git)
    os.chdir(path_git)
    sys.path.append('./fr_mcmc/utils/')
    from data import read_data_pantheon_2
    from data import read_data_pantheon
    #Parameters order: Mabs,omega_m,b,H_0,n

    #Fixed parameters for testing:
    M_true = -19.2
    omega_m_true = 0.3
    b_true = 0.5
    H0_true =  73.48 #(km/seg)/Mpc
    alpha_true = 0.154
    beta_true = 3.02
    gamma_true = 0.053
    n = 1

    fixed_params = [H0_true,n]
    theta = [M_true,omega_m_true,b_true]

    #%% SN data
    os.chdir(path_git+'/fr_mcmc/source/Pantheon')

    _, zcmb, zhel, Cinv, mb0, x1, cor, hmass = read_data_pantheon_2(
                'lcparam_full_long_zhel.txt','ancillary_g10.txt')
    zcmb_1,zhel_1, Cinv_1, mb_1 = read_data_pantheon('lcparam_full_long_zhel.txt')

    params_to_chi2(theta, fixed_params, zcmb, zhel, Cinv,mb_1)
    #%%
    np.all(zhel_1==zhel)
    np.where(zcmb_1==zcmb)

    alpha_0=0.154
    beta_0=3.02
    gamma_0=0.053
    mstep0=10.13
    tau0=0.001
    from matplotlib import pyplot as plt
    plt.figure()
    plt.plot(hmass,gamma_0*np.power((1.+np.exp((mstep0-hmass)/tau0)),-1),'.')
    plt.plot(hmass,gamma_0*np.heaviside(hmass-mstep0, 1),'.')
    plt.grid(True)

    plt.figure()
    plt.plot(zcmb-zcmb_1,'.')
    plt.grid(True)
