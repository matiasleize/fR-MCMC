"""
Definition of the log likelihood distribution and the chi_square in terms of
the parameters of the model and the datasets which are use. 
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid as cumulative_trapezoid
from scipy.constants import c as c_light #meters/seconds
c_light_km = c_light/1000

import os
import git
path_git = git.Repo('.', search_parent_directories=True).working_tree_dir
os.chdir(path_git); os.sys.path.append('./fr_mcmc/utils/')

from LambdaCDM import H_LCDM
from solve_sys import Hubble_th
from ML import H_ML, ML_limits
from supernovae import aparent_magnitude_th, chi2_supernovae
from BAO import r_drag, Hs_to_Ds, Ds_to_obs_final
from AGN import zs_2_logDlH0
from constants import OMEGA_R_0, WB_BBN
#from ML import H_ML

def chi2_without_cov(teo, data, errors_cuad):
    '''
    Calculate chi square assuming no correlation.

    teo (array): Theoretical prediction of the model.
    data (array): Observational data to compare with the model.
    errors_cuad (array): The square of the errors of the data.

    '''

    chi2 = np.sum((data-teo)**2/errors_cuad)
    return chi2


def all_parameters(theta, fixed_params, model, index):
    '''
    Auxiliary function that reads and organizes fixed and variable parameters into one 
    list according to the index criteria.

    theta: object with variable parameters.
    fixed_params: object with fixed parameters.
    index (int): indicate if the parameters are fixed or variable.
    
    '''

    if model == 'LCDM':
        if index == 4:
            [Mabs, bao_param, omega_m, H_0] = theta
            _ = fixed_params
        elif index == 31:
            [Mabs, omega_m, H_0] = theta
            bao_param = fixed_params
        elif index == 32:
            [bao_param, omega_m, H_0] = theta
            Mabs = fixed_params
        elif index == 21:
            [omega_m, H_0] = theta
            [Mabs, bao_param] = fixed_params
        else:
            raise ValueError('Introduce a valid index for LCDM!')
        return [Mabs, bao_param, omega_m, H_0]

    elif (model == 'HS' or model == 'ST' or model == 'EXP'):
        if index == 5:
            [Mabs, bao_param, omega_m, b, H_0] = theta
            _ = fixed_params
        elif index == 41:
            [Mabs, bao_param, b, H_0] = theta
            omega_m = fixed_params
        elif index == 42:
            [Mabs, omega_m, b, H_0] = theta
            bao_param = fixed_params
        elif index == 43:
            [bao_param, omega_m, b, H_0] = theta
            Mabs = fixed_params
        elif index == 31:
            [Mabs, b, H_0] = theta
            bao_param, omega_m = fixed_params
        elif index == 32:
            [bao_param, b, H_0] = theta
            Mabs, omega_m = fixed_params
        elif index == 33:
            [omega_m, b, H_0] = theta
            [Mabs, bao_param] = fixed_params
    else:
        raise ValueError('Introduce a valid index for HS, ST and EXP!')
    return [Mabs, bao_param, omega_m, b, H_0]

def params_to_chi2(theta, fixed_params, index=0,
                   dataset_SN_plus_shoes=None, dataset_SN_plus=None,
                   dataset_SN=None, dataset_CC=None,
                   dataset_BAO=None, dataset_DESI=None, dataset_BAO_full=None,
                   dataset_AGN=None, H0_Riess=False,
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
    dataset_DESI:
    dataset_BAO_full:
    dataset_AGN:
    H0_Riess:

    num_z_points:
    model (str): cosmological model ('LCDM', 'HS', 'ST', 'EXP').
    n (int): (1, 2)
    nuisance_2 (bool):
    enlarged_errors (bool):
    all_analytic (bool):
    '''

    chi2_SN = 0
    chi2_CC = 0
    chi2_BAO = 0
    chi2_DESI = 0
    chi2_BAO_full = 0
    chi2_AGN = 0
    chi2_H0 =  0

    ML_bool = True
    if model == 'LCDM':
        [Mabs, bao_param, omega_m, H_0] = all_parameters(theta, fixed_params, model, index)
        zs_model = np.linspace(0, 10, num_z_points)
        Hs_model = H_LCDM(zs_model, omega_m, H_0)

    elif (model == 'HS' or model == 'ST' or model == 'EXP'):
        [Mabs, bao_param, omega_m, b, H_0] = all_parameters(theta, fixed_params, model, index)
        physical_params = [omega_m, b, H_0]
        
        if (model == 'HS' or model == 'ST') and ML_bool == True:    
            Om_m_0_min, Om_m_0_max, b_min, b_max = ML_limits(model)
            #print(Om_m_0_min, Om_m_0_max, b_min, b_max)

            #If parameters are inside the ML training..
            if (Om_m_0_min < omega_m < Om_m_0_max) and (b_min < b < b_max):
                zs_model = np.linspace(0, 10, num_z_points)
                Hs_model = H_ML(zs_model, [b, omega_m, H_0, 0], model=model)
            else:
                try:
                    zs_model, Hs_model = Hubble_th(physical_params, n=n, model=model,
                                                z_min=0, z_max=10, num_z_points=num_z_points,
                                                all_analytic=all_analytic)
                except Exception as e:
                    # If integration fails, reject the step
                    return -np.inf

        else:
            try:
                zs_model, Hs_model = Hubble_th(physical_params, n=n, model=model,
                                            z_min=0, z_max=10, num_z_points=num_z_points,
                                            all_analytic=all_analytic)
            except Exception as e:
                # If integration fails, reject the step
                return -np.inf

    if (dataset_CC != None or dataset_BAO != None or dataset_DESI != None or 
        dataset_BAO_full != None or dataset_AGN != None):
        Hs_interp = interp1d(zs_model, Hs_model)

    if (dataset_SN_plus_shoes != None or dataset_SN_plus != None or
        dataset_SN != None or dataset_BAO != None or dataset_AGN != None or 
        dataset_DESI != None or dataset_BAO_full != None):
        int_inv_Hs = cumulative_trapezoid(Hs_model**(-1), zs_model, initial=0)
        int_inv_Hs_interp = interp1d(zs_model, int_inv_Hs)

    if (dataset_BAO != None or dataset_DESI != None or 
        dataset_BAO_full != None):
        #rd = bao_param

        #wb = bao_param
        #rd = r_drag(Omega_m_LCDM, H_0, wb) #rd calculation

        rd = r_drag(omega_m, H_0, WB_BBN) #rd calculation

    if dataset_SN_plus_shoes != None:
        zhd, zhel, mb, mu_shoes, Cinv, is_cal = dataset_SN_plus_shoes #Import the data
        muobs = mb - Mabs
        muth_num = aparent_magnitude_th(int_inv_Hs_interp, zhd, zhel) #Numeric prediction of mu
        muth = muth_num*(-is_cal + 1) + mu_shoes*(is_cal) #Merge num predicion with mu_shoes
        chi2_SN = chi2_supernovae(muth, muobs, Cinv)

    if dataset_SN != None:
        zcmb, zhel, Cinv, mb = dataset_SN #Import the data
        muth = aparent_magnitude_th(int_inv_Hs_interp, zcmb, zhel)
        muobs =  mb - Mabs
        chi2_SN = chi2_supernovae(muth, muobs, Cinv)

    if dataset_CC != None:
        z_data, H_data, dH = dataset_CC #Import the data
        H_teo = Hs_interp(z_data)
        chi2_CC = chi2_without_cov(H_teo, H_data, dH**2)
    
    if dataset_BAO != None:
        num_datasets=5

        chies_BAO = np.zeros(num_datasets)
        for i in range(num_datasets): # For each datatype
            (z_data_BAO, data_values, data_squared_errors) = dataset_BAO[i]
            if i==0: #Da entry
                rd = r_drag(omega_m, H_0, bao_param) # rd calculation
                theoretical_distances = Hs_to_Ds(Hs_interp, int_inv_Hs_interp, z_data_BAO, i)
                output_th = Ds_to_obs_final(theoretical_distances, rd, i)
            else: #If not..
                theoretical_distances = Hs_to_Ds(Hs_interp, int_inv_Hs_interp, z_data_BAO, i)
                output_th = np.zeros(len(z_data_BAO))
                for j in range(len(z_data_BAO)): # For each datatype
                    output_th[j] = Ds_to_obs_final(theoretical_distances[j] ,rd, i)
            #Chi square calculation for each datatype (i)
            chies_BAO[i] = chi2_without_cov(output_th, data_values, data_squared_errors)


        if np.isnan(sum(chies_BAO))==True:
            print('There are some errors!')
            print(omega_m,H_0,rd)

        chi2_BAO = np.sum(chies_BAO)
    
    if dataset_BAO_full != None:
        (set_1, set_2) = dataset_BAO_full
        z_data_BAO, data_values, data_squared_errors, index = set_1
        chies_BAO_full_1 = np.zeros(len(z_data_BAO))
        for i in range(len(chies_BAO_full_1)): # For each datatype
            theoretical_distances = Hs_to_Ds(Hs_interp, int_inv_Hs_interp, z_data_BAO[i], index[i])
            output_th = Ds_to_obs_final(theoretical_distances, rd, index[i])
            #Chi square calculation for each datatype (i)
            chies_BAO_full_1[i] = chi2_without_cov(output_th, data_values[i], data_squared_errors[i])        

        z_eff_2, data_dm_rd, errors_dm_rd, data_dh_rd, errors_dh_rd, rho = set_2
        chies_BAO_full_2 = np.zeros(len(z_eff_2))
        for j in range(len(chies_BAO_full_2)): # For each datatype
            aux = Hs_to_Ds(Hs_interp, int_inv_Hs_interp, z_eff_2[j], 1)
            dh_th = Ds_to_obs_final(aux, rd, 1) # dh/rd

            aux = Hs_to_Ds(Hs_interp, int_inv_Hs_interp, z_eff_2[j], 2)
            dm_th = Ds_to_obs_final(aux, rd, 2) # dm/rd

            delta_dh = dh_th - data_dh_rd[j]
            delta_dm = dm_th - data_dm_rd[j]
            Cov_mat = np.array([[errors_dh_rd[j]**2, errors_dh_rd[j]*errors_dm_rd[j]*rho[j]],\
                                [errors_dh_rd[j]*errors_dm_rd[j]*rho[j], errors_dm_rd[j]**2]])
            C_inv = np.linalg.inv(Cov_mat)

            delta = np.array([delta_dh, delta_dm]) #row vector
            transp = np.transpose(delta) #column vector
            aux_chi = np.dot(C_inv, transp) #column vector
            chies_BAO_full_2[j] = np.dot(delta, aux_chi) #scalar    
        
        chi2_BAO_full = np.sum(chies_BAO_full_1) + np.sum(chies_BAO_full_2)


    if dataset_DESI != None:
        
        (set_1, set_2) = dataset_DESI
        z_eff_1, data_dm_rd, errors_dm_rd, data_dh_rd, errors_dh_rd, rho = set_1 
        z_eff_2, data_dv_rd, errors_dv_rd = set_2

        #index: 1 (DH)
        #index: 2 (DM)
        #index: 3 (DV)

        #DM_DH
        chi2_dm_dh = np.zeros(len(z_eff_1))

        for j in range(len(chi2_dm_dh)): # For each datatype

            aux = Hs_to_Ds(Hs_interp, int_inv_Hs_interp, z_eff_1[j], 1)
            dh_th = Ds_to_obs_final(aux, rd, 1) # dh/rd

            aux = Hs_to_Ds(Hs_interp, int_inv_Hs_interp, z_eff_1[j], 2)
            dm_th = Ds_to_obs_final(aux, rd, 2) # dm/rd

            delta_dh = dh_th - data_dh_rd[j]
            delta_dm = dm_th - data_dm_rd[j]
            Cov_mat = np.array([[errors_dh_rd[j]**2, errors_dh_rd[j]*errors_dm_rd[j]*rho[j]],\
                                [errors_dh_rd[j]*errors_dm_rd[j]*rho[j], errors_dm_rd[j]**2]])
            C_inv = np.linalg.inv(Cov_mat)

            delta = np.array([delta_dh, delta_dm]) #row vector
            transp = np.transpose(delta) #column vector
            aux_chi = np.dot(C_inv, transp) #column vector
            chi2_dm_dh[j] = np.dot(delta, aux_chi) #scalar

        chi2_DESI_1 = np.sum(chi2_dm_dh)

        #DV
        dv_th_rd = np.zeros(len(z_eff_2))
        for j in range(len(dv_th_rd)): # For each datatype
            aux = Hs_to_Ds(Hs_interp, int_inv_Hs_interp, z_eff_2[j], 3)
            dv_th_rd[j] = Ds_to_obs_final(aux, rd, 3) # dv/rd

        #Chi square calculation for each datatype (i)
        chi2_DESI_2 = chi2_without_cov(dv_th_rd, data_dv_rd, errors_dv_rd**2)

        chi2_DESI = chi2_DESI_1 + chi2_DESI_2

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

        DlH0_teo = zs_2_logDlH0(int_inv_Hs_interp(z_data)*H_0,z_data)
        DlH0_obs =  np.log10(3.24) - 25 + (logFx - gamma * logFuv - beta) / (2*gamma - 2)

        df_dgamma =  (-logFx+beta+logFuv) / (2*(gamma-1)**2)
        eDlH0_cuad = (eFx**2 + gamma**2 * eFuv**2 + ebeta**2)/ (2*gamma - 2)**2 + \
                     (df_dgamma)**2 * egamma**2 #Square of the errors

        chi2_AGN = chi2_without_cov(DlH0_teo, DlH0_obs, eDlH0_cuad)

    if H0_Riess == True:
        chi2_H0 = ((Hs_model[0]-73.48)/1.66)**2

    return chi2_SN + chi2_CC + chi2_AGN + chi2_BAO + chi2_DESI + chi2_BAO_full + chi2_H0

def log_likelihood(*args, **kargs):  
    '''
    Return the log likelihood in terms of the chi square.
    '''
    return -0.5 * params_to_chi2(*args, **kargs)

#%%
if __name__ == '__main__':
    from sympy import symbols, latex
    from matplotlib import pyplot as plt
    
    path_git = git.Repo('.', search_parent_directories=True).working_tree_dir
    os.chdir(path_git); os.sys.path.append('./fr_mcmc/utils/')
    from data import read_data_pantheon_plus_shoes, read_data_pantheon_plus, \
                     read_data_pantheon, read_data_chronometers, read_data_BAO, \
                     read_data_AGN, read_data_DESI, read_data_BAO_full
    
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

    # BAO full
    os.chdir(path_git+'/fr_mcmc/source/BAO_full/')
    ds_BAO_full = read_data_BAO_full('BAO_full_1.csv','BAO_full_2.csv')

    # DESI
    os.chdir(path_git+'/fr_mcmc/source/DESI/')
    ds_DESI = read_data_DESI('DESI_data_dm_dh.txt','DESI_data_dv.txt')

    # AGN
    os.chdir(path_git+'/fr_mcmc/source/AGN')
    ds_AGN = read_data_AGN('table3.dat')

    #%%

    # Define your variables
    chi2_LCDM = symbols('chi2_LCDM')
    chi2_HS = symbols('chi2_HS')
    chi2_ST = symbols('chi2_ST')
    chi2_EXP = symbols('chi2_EXP')

    chi2_lcdm_value = params_to_chi2([-19.37, 147, 0.3, 70], None, index=4,
                        dataset_SN_plus_shoes = ds_SN_plus_shoes,
                        dataset_SN_plus = ds_SN_plus,
                        dataset_SN = ds_SN,
                        dataset_CC = ds_CC,
                        dataset_BAO = ds_BAO,
                        dataset_BAO_full = ds_BAO_full,
                        dataset_AGN = ds_AGN,
                        dataset_DESI = ds_DESI,
                        H0_Riess = True,
                        model = 'LCDM'
                        )
                        
    print(r'$\\chi_{\rm \Lambda CDM}^{2}$:', chi2_lcdm_value)

    chi2_HS_value = params_to_chi2([-19.37, 147, 0.3, 0.1, 70], None, index=5,
                    dataset_SN_plus_shoes = ds_SN_plus_shoes,
                    dataset_SN_plus = ds_SN_plus,
                    dataset_SN = ds_SN,
                    dataset_CC = ds_CC,
                    dataset_BAO = ds_BAO,
                    dataset_BAO_full = ds_BAO_full,
                    dataset_AGN = ds_AGN,
                    dataset_DESI = ds_DESI,
                    H0_Riess = True,
                    model = 'HS'
                    )
    print(r'$\\chi_{\rm HS}^{2}$', chi2_HS_value)

    chi2_ST_value = params_to_chi2([-19.37, 147, 0.3, 0.1, 70], None, index=5,
                dataset_SN_plus_shoes = ds_SN_plus_shoes,
                dataset_SN_plus = ds_SN_plus,
                dataset_SN = ds_SN,
                dataset_CC = ds_CC,
                dataset_BAO = ds_BAO,
                dataset_BAO_full = ds_BAO_full,
                dataset_AGN = ds_AGN,
                dataset_DESI = ds_DESI,
                H0_Riess = True,
                model = 'ST'
                )
    print(r'$\\chi_{\rm ST}^{2}$:', chi2_ST_value)

    chi2_EXP_value = params_to_chi2([-19.37, 147, 0.3, 0.1, 70], None, index=5,
                dataset_SN_plus_shoes = ds_SN_plus_shoes,
                dataset_SN_plus = ds_SN_plus,
                dataset_SN = ds_SN,
                dataset_CC = ds_CC,
                dataset_BAO = ds_BAO,
                dataset_BAO_full = ds_BAO_full,
                dataset_AGN = ds_AGN,
                dataset_DESI = ds_DESI,
                H0_Riess = True,
                model = 'EXP'
                )
    print(r'$\\chi_{\rm EXP}^{2}$:', chi2_EXP_value)
    