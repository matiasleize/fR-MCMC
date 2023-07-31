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

def chi2_without_cov(teo, data, errores_cuad):
    '''
    Calculate chi square assuming no correlation.

    teo (array): Theoretical prediction of the model.
    data (array): Observational data to compare with the model.
    errores_cuad (array): The square of the errors of the data.

    '''

    chi2 = np.sum((data-teo)**2/errores_cuad)
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

    if (dataset_SN != None or dataset_BAO != None or dataset_AGN != None):
        int_inv_Hs = cumtrapz(Hs_model**(-1), zs_model, initial=0)
        int_inv_Hs_interpol = interp1d(zs_model, int_inv_Hs)

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
        num_datasets=5
        chies_BAO = np.zeros(num_datasets)
        for i in range(num_datasets): # For each datatype
            (z_data_BAO, data_values, data_error_cuad,wb_fid) = dataset_BAO[i]
            if i==0: #Da entry
                rd = r_drag(omega_m,H_0,wb_fid) # rd calculation
                distancias_teoricas = Hs_to_Ds(Hs_interpol, int_inv_Hs_interpol, z_data_BAO, i)
                output_th = Ds_to_obs_final(zs_model, distancias_teoricas, rd, i)
            else: #If not..
                distancias_teoricas = Hs_to_Ds(Hs_interpol, int_inv_Hs_interpol, z_data_BAO, i)
                output_th = np.zeros(len(z_data_BAO))
                for j in range(len(z_data_BAO)): # For each datatype
                     rd = r_drag(omega_m,H_0,wb_fid[j]) #rd calculation
                     output_th[j] = Ds_to_obs_final(zs_model,distancias_teoricas[j],rd,i)
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
    from data import leer_data_pantheon, leer_data_cronometros, leer_data_BAO, leer_data_AGN

    # Supernovas
    os.chdir(path_git+'/fr_mcmc/source/Pantheon/')
    ds_SN = leer_data_pantheon('lcparam_full_long_zhel.txt')

    # Cronómetros
    os.chdir(path_git+'/fr_mcmc/source/CC/')
    ds_CC = leer_data_cronometros('chronometers_data.txt')

    # BAO
    os.chdir(path_git+'/fr_mcmc/source/BAO/')
    ds_BAO = []
    archivos_BAO = ['BAO_data_da.txt','BAO_data_dh.txt','BAO_data_dm.txt',
                    'BAO_data_dv.txt','BAO_data_H.txt']
    for i in range(5):
        aux = leer_data_BAO(archivos_BAO[i])
        ds_BAO.append(aux)

    # AGN
    os.chdir(path_git+'/fr_mcmc/source/AGN')
    ds_AGN = leer_data_AGN('table3.dat')


    #%%
    a = params_to_chi2([-19.37, 0.3, 70], 0.01, index=32,
                    #dataset_SN = ds_SN,
                    #dataset_CC = ds_CC,
                    dataset_BAO = ds_BAO,
                    #dataset_AGN = ds_AGN,
                    H0_Riess = True,
                    model = 'HS'
                    )
    print(a)
    #%%
    from scipy.stats import chi2
    N = len(ds_SN[0])
    P = 3
    df = N - P
    x = np.linspace(0,2000, 10**5)
    y = chi2.pdf(x, df, loc=0, scale=1)
    plt.vlines(a,0,np.max(y),'r')
    plt.plot(x,y)

    #%%
    bs = np.linspace(0,2,22)
    chies_HS = np.zeros(len(bs))
    chies_EXP = np.zeros(len(bs))
    for (i,b) in enumerate(bs):
        chies_EXP[i] = params_to_chi2([0.325,b,68], -19.35, index=31,
                        #dataset_SN = ds_SN,
                        dataset_CC = ds_CC,
                        #dataset_BAO = ds_BAO,
                        #dataset_AGN = ds_AGN,
                        H0_Riess = True,
                        model = 'EXP'
                        )
        chies_HS[i] = params_to_chi2([0.325,b,68], -19.35, index=31,
                        #dataset_SN = ds_SN,
                        dataset_CC = ds_CC,
                        #dataset_BAO = ds_BAO,
                        #dataset_AGN = ds_AGN,
                        H0_Riess = True,
                        model = 'HS'
                        )
        print(i)

        plt.figure()
        plt.title('CC+H0')
        plt.ylabel(r'$\chi^2$')
        plt.xlabel('b')
        plt.grid(True)
        plt.plot(bs,chies_HS,label = 'Hu-Sawicki model')
        plt.plot(bs,chies_EXP,label = 'Exponencial model')
        plt.legend()

    #%%
    bs = np.linspace(0.1,4,100)
    chis_1 = np.zeros(len(bs))
    chis_2 = np.zeros(len(bs))
    chis_3 = np.zeros(len(bs))
    chis_4 = np.zeros(len(bs))
    chis_5 = np.zeros(len(bs))
    chis_6 = np.zeros(len(bs))
    for (i,b) in enumerate(bs):
        #print(i,b)
        chis_1[i] = params_to_chi2([-19.41, 0.352, b, 62], 0, index=4,
                    dataset_SN = ds_SN,
                    dataset_CC = ds_CC,
                    dataset_BAO = ds_BAO,
                    #dataset_AGN = ds_AGN,
                    #H0_Riess = True,
                    model = 'EXP'
                    )
        chis_2[i] = params_to_chi2([-19.41, 0.352, b, 63], 0, index=4,
                    dataset_SN = ds_SN,
                    dataset_CC = ds_CC,
                    dataset_BAO = ds_BAO,
                    #dataset_AGN = ds_AGN,
                    #H0_Riess = True,
                    model = 'EXP'
                    )

        chis_3[i] = params_to_chi2([-19.41, 0.352, b, 64], 0, index=4,
                    dataset_SN = ds_SN,
                    dataset_CC = ds_CC,
                    dataset_BAO = ds_BAO,
                    #dataset_AGN = ds_AGN,
                    #H0_Riess = True,
                    model = 'EXP'
                    )
        chis_4[i] = params_to_chi2([-19.41, 0.352, b, 65], 0, index=4,
                    dataset_SN = ds_SN,
                    dataset_CC = ds_CC,
                    dataset_BAO = ds_BAO,
                    #dataset_AGN = ds_AGN,
                    #H0_Riess = True,
                    model = 'EXP'
                    )
        chis_5[i] = params_to_chi2([-19.41, 0.352, b, 66], 0, index=4,
                    dataset_SN = ds_SN,
                    dataset_CC = ds_CC,
                    dataset_BAO = ds_BAO,
                    #dataset_AGN = ds_AGN,
                    #H0_Riess = True,
                    model = 'EXP'
                    )
        chis_6[i] = params_to_chi2([-19.41, 0.352, b, 67], 0, index=4,
                    dataset_SN = ds_SN,
                    dataset_CC = ds_CC,
                    dataset_BAO = ds_BAO,
                    #dataset_AGN = ds_AGN,
                    #H0_Riess = True,
                    model = 'EXP'
                    )
    #1077.8293845284927/(1048+20+len(ds_CC[0])-4)
    #%%
    plt.title('EXP: CC+SN+BAO, omega_m=0.352, M=-19.41')
    plt.grid(True)
    plt.plot(bs,chis_1,label='H0=62')
    plt.plot(bs,chis_2,label='H0=63')
    plt.plot(bs,chis_3,label='H0=64')
    plt.plot(bs,chis_4,label='H0=65')
    plt.plot(bs,chis_5,label='H0=66')
    plt.plot(bs,chis_6,label='H0=67')
    plt.ylabel(r'$\chi^2$')
    plt.xlabel('b')
    plt.legend()