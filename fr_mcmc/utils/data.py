"""
Functions related to data management
"""

import numpy as np
from numpy.linalg import inv
import numpy.ma as ma

def read_data_pantheon(file_pantheon, masked = False, min_z = 0, max_z = 30):

    '''
    Takes Pantheon data and extracts the data from the zcmb and zhel 
    redshifts, its error dz, in addition to the data of the apparent magnitude
    with its error: mb and dm. With the errors of the apparent magnitude 
    builds the associated correlation matrix. The function returns the
    information of the redshifts, the apparent magnitude 
    and the correlation matrix inverse.
    '''

    # Read text with data
    zcmb,zhel,dz,mb,dmb=np.loadtxt(file_pantheon
                                   , usecols=(1,2,3,4,5),unpack=True)
    #Create the diagonal matrx with m_B uncertainties (it depends on alpha and beta).
    Dstat=np.diag(dmb**2.)

    # Read data and create the matrix with sistematic errors of NxN
    sn=len(zcmb)
    Csys=np.loadtxt('lcparam_full_long_sys.txt',unpack=True)
    Csys=Csys.reshape(sn,sn)
    #We made the final covariance matrix..
    Ccov=Csys+Dstat


    if masked == True:
        mask = ma.masked_where((zcmb <= max_z) & ((zcmb >= min_z)) , zcmb).mask
        mask_1 = mask[np.newaxis, :] & mask[:, np.newaxis]

        zhel = zhel[mask]
        mb = mb[mask]
        Ccov = Ccov[mask_1]
        Ccov = Ccov.reshape(len(zhel),len(zhel))
        zcmb = zcmb[mask]

    #.. and finally we invert it
    Cinv=inv(Ccov)
    return zcmb, zhel, Cinv, mb

def read_data_pantheon_2(file_pantheon,file_pantheon_2):
    '''Idem read_data_pantheon, apart from importing nuisance parameters.'''
    # Read text with data
    zcmb0,zhel0,dz0,mb0,dmb0=np.loadtxt(file_pantheon
                    , usecols=(1,2,3,4,5),unpack=True)
    zcmb_1,hmass,x1,cor=np.loadtxt(file_pantheon_2,usecols=(7,13,20,22),
                        unpack=True)
    #Create the diagonal matrx with m_B uncertainties (it depends on alpha and beta).
    Dstat=np.diag(dmb0**2.)

    # Read data and create the matrix with sistematic errors of NxN
    sn=len(zcmb0)
    Csys=np.loadtxt('lcparam_full_long_sys.txt',unpack=True)
    Csys=Csys.reshape(sn,sn)
    #We made the final covariance matrix and then we invert it.
    Ccov=Csys+Dstat
    Cinv=inv(Ccov)

    return zcmb0, zcmb_1, zhel0, Cinv, mb0, x1, cor, hmass



def read_data_chronometers(file_chronometers):
    # Read text with data
    z, h, dh = np.loadtxt(file_chronometers, usecols=(0,1,2), unpack=True)
    return z, h, dh

def read_data_BAO(file_BAO):
    z, data_values, errors_est, errors_sist, wb_fid = np.loadtxt(file_BAO,
    usecols=(0,1,2,3,4), skiprows=1,unpack=True)
    total_errors_cuad = errors_est**2 + errors_sist**2
    return z, data_values, total_errors_cuad, wb_fid

def read_data_AGN(file_AGN):
    z, Fuv, eFuv, Fx, eFx = np.loadtxt(file_AGN,
    usecols=(3,4,5,6,7), unpack=True)
    arr1inds = z.argsort()
    sorted_z = z[arr1inds]
    sorted_Fuv = Fuv[arr1inds]
    sorted_eFuv = eFuv[arr1inds]
    sorted_Fx = Fx[arr1inds]
    sorted_eFx = eFx[arr1inds]

    return sorted_z, sorted_Fuv, sorted_eFuv, sorted_Fx, sorted_eFx

def read_data_BAO_odintsov(file_BAO_odintsov):
    # Read text with data
    z, h, dh, rd_fid = np.loadtxt(file_BAO_odintsov, usecols=(0,1,2,3), unpack=True)
    return z, h, dh, rd_fid
#%%
if __name__ == '__main__':
    import pandas as pd
    import os
    import git
    path_git = git.Repo('.', search_parent_directories=True).working_tree_dir
    os.chdir(path_git)

    #%% AGN
    os.chdir(path_git+'/fr_mcmc/source/AGN')
    aux = read_data_AGN('table3.dat')

    #%% Supernovae
    os.chdir(path_git+'/fr_mcmc/source/Pantheon')
    zcmb, zhel, Cinv, mb = read_data_pantheon('lcparam_full_long_zhel.txt')
    #zcmb, zhel, Cinv, mb

    #%% Cosmic chronometers
    os.chdir(path_git+'/fr_mcmc/source/CC')
#    z_data, H_data, dH  = read_data_chronometers('chronometers_data.txt')
    z_data, H_data, dH  = read_data_chronometers('chronometers_data_nunes.txt')

    #%% BAO
    os.chdir(path_git+'/fr_mcmc/source/BAO')
    file_BAO='BAO_data_da.txt'

    z, data_values, data_error_cuad = read_data_BAO(file_BAO)
    #z, data_values, data_error_cuad
    
    #%%
    os.chdir(path_git+'/fr_mcmc/source/BAO')
    file_BAO='BAO_data.txt'
    df = pd.read_csv(file_BAO,sep='\t')
    z_data = df.to_numpy()[:,0]
    data_values = df.to_numpy()[:,1]
    errors_est = df.to_numpy()[:,2]
    errors_sist = df.to_numpy()[:,3]
    total_errors_cuad = errors_est**2 + errors_sist**2
    df.to_numpy()[:,4]
