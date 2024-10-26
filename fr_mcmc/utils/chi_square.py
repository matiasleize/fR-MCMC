"""
Definition of the log likelihood distribution and the chi_square in terms of
the parameters of the model and the datasets which are use. 
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz as cumtrapz
from scipy.constants import c as c_luz #meters/seconds
c_luz_km = c_luz/1000

import os
import git
path_git = git.Repo('.', search_parent_directories=True).working_tree_dir

os.chdir(path_git); os.sys.path.append('./fr_mcmc/utils/')
from solve_sys import Hubble_th
from supernovae import aparent_magnitude_th, chi2_supernovae
from BAO import r_drag, Hs_to_Ds, Ds_to_obs_final
from AGN import zs_2_logDlH0

def chi2_without_cov(teo, data, errors_cuad):
    '''
    Calculate chi square assuming no correlation.

    teo (array): Theoretical prediction of the model.
    data (array): Observational data to compare with the model.
    errors_cuad (array): The square of the errors of the data.

    '''

    chi2 = np.sum((data-teo)**2/errors_cuad)
    return chi2

def all_parameters(theta, fixed_params, index):
    '''
    Auxiliary function that reads and organizes fixed and variable parameters into one 
    list according to the index criteria.

    theta: object with variable parameters.
    fixed_params: object with fixed parameters.
    index (int): indicate if the parameters are fixed or variable.
    
    '''

    if index == 4:
        [Mabs, omega_m, b, H_0] = theta
        _ = fixed_params

    elif index == 31:
        [omega_m, b, H_0] = theta
        Mabs = fixed_params

    elif index == 32:
        [Mabs, omega_m, H_0] = theta
        b = fixed_params

    elif index == 33:
        [Mabs, omega_m, b] = theta
        H_0 = fixed_params

    elif index == 21:
        [omega_m, b] = theta
        [Mabs, H_0] = fixed_params

    elif index == 22:
        [omega_m, H_0] = theta
        [Mabs, b] = fixed_params

    elif index == 23:
        [Mabs, omega_m] = theta
        [b, H_0] = fixed_params

    elif index == 1:
        omega_m = theta
        [Mabs, b, H_0] = fixed_params


    return [Mabs, omega_m, b, H_0]


def params_to_chi2(theta, fixed_params, index=0,
                   dataset_SN_plus_shoes=None, dataset_SN_plus=None,
                   dataset_SN=None, dataset_CC=None,
                   dataset_BAO=None, dataset_AGN=None, H0_Riess=False,
                   num_z_points=int(10**5), model='HS',n=1,
                   nuisance_2 = False, enlarged_errors=False,
                   all_analytic=False):
    '''
    Given the free parameters of the model, return chi square for the data.
    
    theta: object with variable parameters.
    fixed_params: object with fixed parameters.
    index (int): indicate if the parameters are fixed or variable.

    dataset_SN:
    dataset_CC:
    dataset_BAO: This data goes up to z=7.4 aproximately. Don't integrate with z less than that!
    dataset_AGN:
    H0_Riess:

    num_z_points:
    model (str): cosmological model ('LCDM', 'HS', 'EXP').
    n (int): (1, 2)
    nuisance_2 (bool):
    enlarged_errors (bool):
    all_analytic (bool):
    '''

    chi2_SN = 0
    chi2_CC = 0
    chi2_BAO = 0
    chi2_AGN = 0
    chi2_H0 =  0

    [Mabs, omega_m, b, H_0] = all_parameters(theta, fixed_params, index)

    physical_params = [omega_m,b,H_0]
    zs_model, Hs_model = Hubble_th(physical_params, n=n, model=model,
                                z_min=0, z_max=10, num_z_points=num_z_points,
                                all_analytic=all_analytic)

    if (dataset_CC != None or dataset_BAO != None or dataset_AGN != None):
        Hs_interpol = interp1d(zs_model, Hs_model)

    if (dataset_SN_plus_shoes != None or dataset_SN_plus != None or
        dataset_SN != None or dataset_BAO != None or dataset_AGN != None):
        int_inv_Hs = cumtrapz(Hs_model**(-1), zs_model, initial=0)
        int_inv_Hs_interpol = interp1d(zs_model, int_inv_Hs)

    if dataset_SN_plus_shoes != None:
        zhd, zhel, mb, mu_shoes, Cinv, is_cal = dataset_SN_plus_shoes #Import the data
        muobs = mb - Mabs
        muth_num = aparent_magnitude_th(int_inv_Hs_interpol, zhd, zhel) #Numeric prediction of mu
        muth = muth_num*(-is_cal + 1) + mu_shoes*(is_cal) #Merge num predicion with mu_shoes
        chi2_SN = chi2_supernovae(muth, muobs, Cinv)

    if dataset_SN_plus != None:
        zhd, zhel, Cinv, mb = dataset_SN_plus #Import the data
        muth = aparent_magnitude_th(int_inv_Hs_interpol, zhd, zhel)
        muobs =  mb - Mabs
        chi2_SN = chi2_supernovae(muth, muobs, Cinv)

    if dataset_SN != None:
        zcmb, zhel, Cinv, mb = dataset_SN #Import the data
        muth = aparent_magnitude_th(int_inv_Hs_interpol, zcmb, zhel)
        muobs =  mb - Mabs
        chi2_SN = chi2_supernovae(muth, muobs, Cinv)

    if dataset_CC != None:
        z_data, H_data, dH = dataset_CC #Import the data
        H_teo = Hs_interpol(z_data)
        chi2_CC = chi2_without_cov(H_teo, H_data, dH**2)

    if dataset_BAO != None:
        WB_BBN = 0.02218 #± 0.00055 #Baryon density (eq. 2.8 arXiv:2404.03002)
        num_datasets=5
        chies_BAO = np.zeros(num_datasets)
        for i in range(num_datasets): # For each datatype
            (z_data_BAO, data_values, data_error_cuad) = dataset_BAO[i]
            if i==0: #Da entry
                rd = r_drag(omega_m, H_0, WB_BBN) # rd calculation
                theoretical_distances = Hs_to_Ds(Hs_interpol, int_inv_Hs_interpol, z_data_BAO, i)
                output_th = Ds_to_obs_final(zs_model, theoretical_distances, rd, i)
            else: #If not..
                theoretical_distances = Hs_to_Ds(Hs_interpol, int_inv_Hs_interpol, z_data_BAO, i)
                output_th = np.zeros(len(z_data_BAO))
                for j in range(len(z_data_BAO)): # For each datatype
                     rd = r_drag(omega_m, H_0, WB_BBN) #rd calculation
                     output_th[j] = Ds_to_obs_final(zs_model,theoretical_distances[j],rd,i)
            #Chi square calculation for each datatype (i)
            chies_BAO[i] = chi2_without_cov(output_th,data_values,data_error_cuad)


        if np.isnan(sum(chies_BAO))==True:
            print('There are some errors!')
            print(omega_m,H_0,rd)

        chi2_BAO = np.sum(chies_BAO)


    if dataset_AGN != None:
        z_data, logFuv, eFuv, logFx, eFx  = dataset_AGN #Import the data

        if nuisance_2 == True: #Deprecated
            beta = 8.513
            ebeta = 0.437
            gamma = 0.622
            egamma = 0.014
        elif enlarged_errors == True:
            beta = 7.735
            ebeta = 0.6
            gamma = 0.648
            egamma = 0.007
        else: #Standard case
            beta = 7.735
            ebeta = 0.244
            gamma = 0.648
            egamma = 0.007

        DlH0_teo = zs_2_logDlH0(int_inv_Hs_interpol(z_data)*H_0,z_data)
        DlH0_obs =  np.log10(3.24) - 25 + (logFx - gamma * logFuv - beta) / (2*gamma - 2)

        df_dgamma =  (-logFx+beta+logFuv) / (2*(gamma-1)**2)
        eDlH0_cuad = (eFx**2 + gamma**2 * eFuv**2 + ebeta**2)/ (2*gamma - 2)**2 + (df_dgamma)**2 * egamma**2 #Square of the errors

        chi2_AGN = chi2_without_cov(DlH0_teo, DlH0_obs, eDlH0_cuad)

    if H0_Riess == True:
        chi2_H0 = ((Hs_model[0]-73.48)/1.66)**2

    return chi2_SN + chi2_CC + chi2_AGN + chi2_BAO + chi2_H0

def log_likelihood(*args, **kargs):  
    '''
    Return the log likelihood in terms of the chi square.
    '''

    return -0.5 * params_to_chi2(*args, **kargs)

#%%
if __name__ == '__main__':
    from matplotlib import pyplot as plt

    path_git = git.Repo('.', search_parent_directories=True).working_tree_dir
    os.chdir(path_git); os.sys.path.append('./fr_mcmc/utils/')
    from data import read_data_pantheon_plus_shoes, read_data_pantheon_plus, read_data_pantheon,\
                     read_data_chronometers, read_data_BAO, read_data_AGN

    # Pantheon plus + SH0ES
    os.chdir(path_git+'/fr_mcmc/source/Pantheon_plus_shoes')
    ds_SN_plus_shoes = read_data_pantheon_plus_shoes('Pantheon+SH0ES.dat',
                                    'Pantheon+SH0ES_STAT+SYS.cov')

    # Pantheon plus
    os.chdir(path_git+'/fr_mcmc/source/Pantheon_plus_shoes')
    ds_SN_plus = read_data_pantheon_plus('Pantheon+SH0ES.dat',
                                'covmat_pantheon_plus_only.npz')

    # Pantheon
    os.chdir(path_git+'/fr_mcmc/source/Pantheon/')
    ds_SN = read_data_pantheon('lcparam_full_long_zhel.txt')

    # Cosmic Chronometers
    os.chdir(path_git+'/fr_mcmc/source/CC/')
    ds_CC = read_data_chronometers('chronometers_data.txt')

    # BAO
    os.chdir(path_git+'/fr_mcmc/source/BAO/')
    ds_BAO = []
    files_BAO = ['BAO_data_da.txt','BAO_data_dh.txt','BAO_data_dm.txt',
                    'BAO_data_dv.txt','BAO_data_H.txt']
    for i in range(5):
        aux = read_data_BAO(files_BAO[i])
        ds_BAO.append(aux)

    # AGN
    os.chdir(path_git+'/fr_mcmc/source/AGN')
    ds_AGN = read_data_AGN('table3.dat')


    #%%
    chi2 = params_to_chi2([-19.37, 0.3, 70], 0.01, index=32,
                    dataset_SN_plus_shoes = ds_SN_plus_shoes,
                    dataset_SN_plus = ds_SN_plus,
                    dataset_SN = ds_SN,
                    dataset_CC = ds_CC,
                    dataset_BAO = ds_BAO,
                    dataset_AGN = ds_AGN,
                    H0_Riess = True,
                    model = 'HS'
                    )
    print(chi2)
