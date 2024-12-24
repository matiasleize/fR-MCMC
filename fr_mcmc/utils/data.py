"""
Functions related to data management
"""

import numpy as np
from numpy.linalg import inv
import numpy.ma as ma
import pandas as pd

def read_data_pantheon_plus_shoes(file_pantheon_plus,file_pantheon_plus_shoes_cov):

    '''
    Takes Pantheon+ data and extracts the data from the zhd and zhel 
    redshifts, its error dz, in addition to the data of the apparent magnitude
    with its error: mb and dm. With the errors of the apparent magnitude 
    builds the associated correlation matrix. The function returns the
    information of the redshifts, the apparent magnitude 
    and the correlation matrix inverse.
    '''

    # Read text with data
    df = pd.read_csv(file_pantheon_plus,sep='\s+')

    #################################################################################+
    # NEW for PPS: We mask the data with zHD<0.01 and IS_CALIBRATOR==False
    ww = (df['zHD']>0.01) | (np.array(df['IS_CALIBRATOR'],dtype=bool))
    zhd = df['zHD'][ww]
    zhel = df['zHEL'][ww]
    mb = df['m_b_corr'][ww]
    ceph_dist = df['CEPH_DIST'][ww] #FIXED BUG, before it read: 'mu_shoes = df['MU_SH0ES']' which was wrong!
    is_cal = df['IS_CALIBRATOR'][ww]
    #################################################################################
    
    #Load the covariance matrix elements
    #Ccov=np.loadtxt(file_pantheon_plus_shoes_cov,unpack=True)
    #Ccov=Ccov[1:] #The first element is the total number of row/columns
    #We made the final covariance matrix..
    sn=len(df['zHD'])
    #Ccov=Ccov.reshape(sn,sn)

    f = open(file_pantheon_plus_shoes_cov)
    line = f.readline()
    n = int(len(zhd))
    Cov_PANplus = np.zeros((n,n))
    ii = -1
    jj = -1
    mine = 999
    maxe = -999
    for i in range(sn):
        jj = -1
        if ww[i]:
            ii += 1
        for j in range(sn):
            if ww[j]:
                jj += 1
            val = float(f.readline())
            if ww[i] and ww[j]:
                Cov_PANplus[ii,jj] = val
    f.close()

    #.. and finally we invert it
    Cinv = np.linalg.inv(Cov_PANplus)

    return zhd, zhel, mb, ceph_dist, Cinv, is_cal
    

def read_data_pantheon_plus(file_pantheon_plus,file_pantheon_plus_cov):

    '''
    Takes Pantheon+ data and extracts the data from the zhd and zhel 
    redshifts, its error dz, in addition to the data of the apparent magnitude
    with its error: mb and dm. With the errors of the apparent magnitude 
    builds the associated correlation matrix. The function returns the
    information of the redshifts, the apparent magnitude 
    and the correlation matrix inverse.
    '''

    # Read text with data

    df = pd.read_csv(file_pantheon_plus,sep='\s+')
    ww = (df['zHD']>0.01) | (np.array(df['IS_CALIBRATOR'],dtype=bool))

    zhd = df['zHD'][ww]
    zhel = df['zHEL'][ww]
    mb = df['m_b_corr'][ww]

    Ccov=np.load(file_pantheon_plus_cov)['arr_0']
    Cinv=inv(Ccov)

    return zhd, zhel, Cinv, mb

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
    z, data_values, errors_est, errors_sist = np.loadtxt(file_BAO,
    usecols=(0,1,2,3), skiprows=1,unpack=True)
    total_errors_cuad = errors_est**2 + errors_sist**2
    return z, data_values, total_errors_cuad

def read_data_DESI(file_DESI_1, file_DESI_2):
    # Read text with data
    z_eff_1, data_dm_rd, errors_dm_rd, data_dh_rd, errors_dh_rd, rho = np.loadtxt(file_DESI_1,
                                                                     usecols=(0,1,2,3,4,5),
                                                                     skiprows=1, unpack=True)
    
    set_1 = z_eff_1, data_dm_rd, errors_dm_rd, data_dh_rd, errors_dh_rd, rho 

    # Read text with data
    z_eff_2, data_dv_rd, errors_dv_rd = np.loadtxt(file_DESI_2,
                                                usecols=(0,1,2),
                                                skiprows=1, unpack=True)
    set_2 = z_eff_2, data_dv_rd, errors_dv_rd
    return [set_1, set_2]


def read_data_BAO_full(file_BAO_full_1, file_BAO_full_2):
    # Read text with data
    df_1 = pd.read_csv(file_BAO_full_1)
    df_2 = pd.read_csv(file_BAO_full_2)

    total_squared_errors_1 = df_1['Stat_error']**2 + df_1['Sist_error']**2

    # Set data
    set_1 = df_1['z'] , df_1['Dist'], total_squared_errors_1, df_1['index']
    set_2 = df_2['z_eff'], df_2['Dm_rd'], df_2['error_Dm_rd'], df_2['Dh_rd'], df_2['error_Dh_rd'], df_2['rho']

    return [set_1, set_2]


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

    #%% Pantheon plus
    os.chdir(path_git+'/fr_mcmc/source/Pantheon_plus_shoes')
    zhd, zhel, Cinv, mb = read_data_pantheon_plus('Pantheon+SH0ES.dat',
                            'covmat_pantheon_plus_only.npz')

    #%% Pantheon plus + SH0ES
    os.chdir(path_git+'/fr_mcmc/source/Pantheon_plus_shoes')
    zhd, zhel, mb, ceph_dist, Cinv, is_cal = read_data_pantheon_plus_shoes('Pantheon+SH0ES.dat',
                                    'Pantheon+SH0ES_STAT+SYS.cov')

    #%% Pantheon
    os.chdir(path_git+'/fr_mcmc/source/Pantheon')
    zcmb, zhel, Cinv, mb = read_data_pantheon('lcparam_full_long_zhel.txt')

    #%% AGN
    os.chdir(path_git+'/fr_mcmc/source/AGN')
    aux = read_data_AGN('table3.dat')

    #%% Cosmic chronometers
    os.chdir(path_git+'/fr_mcmc/source/CC')
    z_data, H_data, dH  = read_data_chronometers('chronometers_data.txt')

    #%% BAO
    os.chdir(path_git+'/fr_mcmc/source/BAO')
    file_BAO='BAO_data_da.txt'
    z, data_values, total_errors_cuad = read_data_BAO(file_BAO)
    
    #%% BAO full
    os.chdir(path_git+'/fr_mcmc/source/BAO_full/')
    ds_BAO_full = read_data_BAO_full('BAO_full_1.csv','BAO_full_2.csv')
    print(ds_BAO_full)

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